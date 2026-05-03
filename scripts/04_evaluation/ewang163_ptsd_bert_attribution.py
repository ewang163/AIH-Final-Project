"""
ewang163_ptsd_bert_attribution.py
=================================
Integrated Gradients attribution analysis for BioClinicalBERT.

Mirrors `ewang163_ptsd_attribution_v2.py` (the Longformer attribution) but
adapted for BioClinicalBERT's 512-token context and standard
(non-hybrid) attention.

Two attribution slices are produced from the same 50-patient sample of
high-confidence true positives (top decile of predicted probabilities):

  1. **truncated** — IG on the first 512 tokens. Direct counterpart to the
     model's truncated inference mode. Section attribution will skew toward
     the early sections (HPI, often Social History), reflecting what the
     truncated model can actually see.

  2. **per-window (chunk-pool)** — IG run on the *highest-scoring window*
     of each note under the chunk-pool inference. This captures what the
     chunk-pool variant relies on when its max-pooled prediction comes
     from a window deep in the note. The chosen window's character span is
     mapped back to source-note sections using offset mappings so that
     section-share aggregation stays comparable to the Longformer attribution.

For each slice the script writes:
  results/attribution/ewang163_attribution_by_section_bert_{slice}.csv
  results/attribution/ewang163_top_attributed_words_bert_{slice}.csv
  results/attribution/ewang163_attribution_failures_bert_{slice}.log

Submit via SLURM:
  sbatch scripts/04_evaluation/ewang163_ptsd_bert_attribution.sh
"""

import csv
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

csv.field_size_limit(sys.maxsize)

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC               = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR         = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS         = f'{STUDENT_DIR}/data/splits'
MODEL_DIR           = f'{STUDENT_DIR}/models'
RESULTS_ATTRIBUTION = f'{STUDENT_DIR}/results/attribution'

TEST_PARQUET = f'{DATA_SPLITS}/ewang163_split_test.parquet'
DISCHARGE_F  = f'{MIMIC}/note/2.2/discharge.csv'
BERT_DIR     = f'{MODEL_DIR}/ewang163_bioclinbert_best'

MAX_LEN     = 512
STRIDE      = 256
N_SAMPLES   = 50
N_IG_STEPS  = 20
IG_BATCH_SIZE = 1
TOP_K       = 50

# Section parsing
SECTION_HEADER_RE = re.compile(r'^([A-Z][A-Za-z /&\-]+):[ ]*$', re.MULTILINE)
INCLUDE_SECTIONS  = {'history of present illness', 'social history',
                     'past medical history', 'brief hospital course'}
SECTION_ORDER     = ['history of present illness', 'social history',
                     'past medical history', 'brief hospital course']


def parse_section_spans(text):
    if not text:
        return {}
    headers = []
    for m in SECTION_HEADER_RE.finditer(text):
        headers.append((m.start(), m.end(), m.group(1).strip().lower()))
    spans = {}
    for i, (s, e, name) in enumerate(headers):
        if name not in INCLUDE_SECTIONS:
            continue
        body_s = e
        body_e = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[body_s:body_e].strip()
        if body and name not in spans:
            spans[name] = body
    return spans


def build_section_labeled_text(raw_text):
    """Return (concatenated_text, section_ranges)."""
    sections = parse_section_spans(raw_text)
    parts, ranges, off = [], [], 0
    for name in SECTION_ORDER:
        if name not in sections:
            continue
        body = sections[name]
        parts.append(body)
        ranges.append((name, off, off + len(body)))
        off += len(body) + 2  # '\n\n' separator
    return '\n\n'.join(parts), ranges


def map_token_to_section(s, e, ranges):
    for name, ss, se in ranges:
        if s >= ss and e <= se:
            return name
    return 'unknown'


# ── Captum wrapper for BERT ───────────────────────────────────────────────
class BertEmbedForward(torch.nn.Module):
    """Forward (input_embeds, attention_mask) → positive-class logit.
    BERT does NOT need a global_attention_mask, so this is simpler than
    the Longformer wrapper."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_embeds, attention_mask):
        out = self.model(inputs_embeds=input_embeds,
                         attention_mask=attention_mask)
        return out.logits[:, 1]


def chunkpool_top_window(model, tokenizer, text, device,
                         chunk_len=MAX_LEN, stride=STRIDE):
    """Return (top_window_input_ids, attention_mask, char_offsets) for the
    highest-scoring 512-token window, using the same chunking the inference
    pipeline uses."""
    tokens = tokenizer(text, return_tensors='pt', truncation=False,
                       add_special_tokens=False, return_offsets_mapping=True)
    ids_full = tokens['input_ids'].squeeze(0)
    offs_full = tokens['offset_mapping'].squeeze(0).tolist()
    n_tok = len(ids_full)
    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102
    usable = chunk_len - 2

    if n_tok <= usable:
        enc = tokenizer(text, max_length=chunk_len, padding='max_length',
                        truncation=True, return_tensors='pt',
                        return_offsets_mapping=True)
        return (enc['input_ids'].to(device),
                enc['attention_mask'].to(device),
                enc['offset_mapping'][0].tolist())

    best_prob, best_payload = -1.0, None
    start = 0
    while start < n_tok:
        end = min(start + usable, n_tok)
        chunk = torch.cat([
            torch.tensor([cls_id]),
            ids_full[start:end],
            torch.tensor([sep_id]),
        ])
        am = torch.ones(len(chunk), dtype=torch.long)
        offsets = [(0, 0)] + offs_full[start:end] + [(0, 0)]
        pad = chunk_len - len(chunk)
        if pad > 0:
            chunk = torch.cat([chunk, torch.zeros(pad, dtype=torch.long)])
            am = torch.cat([am, torch.zeros(pad, dtype=torch.long)])
            offsets = offsets + [(0, 0)] * pad
        chunk = chunk.unsqueeze(0).to(device)
        am = am.unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda',
                                                  enabled=(device.type == 'cuda')):
            logits = model(input_ids=chunk, attention_mask=am).logits
        prob = float(F.softmax(logits.float(), dim=-1)[0, 1])
        if prob > best_prob:
            best_prob = prob
            best_payload = (chunk, am, offsets)
        if end >= n_tok:
            break
        start += stride
    return best_payload


def write_outputs(slice_name, section_attr_sum, section_token_count,
                  word_attr_accum, failures, n_success, n_fail):
    section_csv = f'{RESULTS_ATTRIBUTION}/ewang163_attribution_by_section_bert_{slice_name}.csv'
    words_csv   = f'{RESULTS_ATTRIBUTION}/ewang163_top_attributed_words_bert_{slice_name}.csv'
    fail_log    = f'{RESULTS_ATTRIBUTION}/ewang163_attribution_failures_bert_{slice_name}.log'

    total = sum(section_attr_sum.values())
    rows = []
    for name in SECTION_ORDER + ['unknown']:
        if name not in section_attr_sum:
            continue
        a = section_attr_sum[name]; t = section_token_count[name]
        rows.append({
            'section': name,
            'total_attribution': round(a, 4),
            'pct_of_total': round(a / total * 100, 2) if total > 0 else 0,
            'n_tokens': t,
            'mean_attribution_per_token': round(a / t, 6) if t > 0 else 0,
        })
    pd.DataFrame(rows).sort_values('total_attribution', ascending=False)\
        .to_csv(section_csv, index=False)

    word_rows = []
    for word, attrs in word_attr_accum.items():
        if len(attrs) >= 3:
            word_rows.append({
                'word': word,
                'mean_abs_attribution': round(float(np.mean(attrs)), 6),
                'std_attribution': round(float(np.std(attrs)), 6),
                'n_occurrences': len(attrs),
                'total_attribution': round(float(np.sum(attrs)), 4),
            })
    word_df = pd.DataFrame(word_rows)
    if len(word_df) > 0:
        word_df = word_df.sort_values('mean_abs_attribution', ascending=False).head(TOP_K)
    word_df.to_csv(words_csv, index=False)

    with open(fail_log, 'w') as f:
        for line in failures:
            f.write(line + '\n')
    print(f'  → {section_csv}')
    print(f'  → {words_csv}')
    print(f'  → {fail_log}  ({n_success} ok, {n_fail} fail)')


def run_one_slice(slice_name, model, tokenizer, ig, embed_layer, device,
                  sample_df, raw_notes):
    """One pass over the 50-patient sample for either 'trunc' or 'chunkpool'."""
    print(f'\n  === Slice = {slice_name} ===')
    section_attr_sum = defaultdict(float)
    section_token_count = defaultdict(int)
    word_attr_accum = defaultdict(list)
    failures = []
    n_success = 0; n_fail = 0
    t0 = time.time()

    for pat_idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        hadm = row['hadm_id']
        note_text = row['note_text']

        if pat_idx % 10 == 0:
            print(f'    [{pat_idx}/{len(sample_df)}] '
                  f'{time.time() - t0:.0f}s, {n_success} ok / {n_fail} fail',
                  flush=True)

        raw = raw_notes.get(hadm)
        if raw:
            _, section_ranges = build_section_labeled_text(raw)
        else:
            section_ranges = []

        try:
            if slice_name == 'trunc':
                enc = tokenizer(note_text, max_length=MAX_LEN, padding='max_length',
                                truncation=True, return_tensors='pt',
                                return_offsets_mapping=True)
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                offsets = enc['offset_mapping'][0].tolist()
            else:  # chunkpool top window
                payload = chunkpool_top_window(model, tokenizer, note_text, device)
                input_ids, attention_mask, offsets = payload

            with torch.no_grad():
                input_embeds = embed_layer(input_ids)
            baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
            with torch.no_grad():
                baseline_embeds = embed_layer(baseline_ids)
            input_embeds = input_embeds.clone().detach().requires_grad_(True)

            attributions = ig.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                n_steps=N_IG_STEPS,
                internal_batch_size=IG_BATCH_SIZE,
            )
        except Exception as e:
            failures.append(f'hadm_id={hadm}: {e}')
            n_fail += 1
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue

        n_success += 1
        attr_scores = attributions.squeeze(0).sum(dim=-1).abs().detach().cpu().numpy()
        seq_len = int(attention_mask[0].sum().item())

        # Walk tokens: aggregate to section + accumulate whole-word attributions
        cur_word_s = cur_word_e = None
        cur_word_attr = 0.0
        for t_idx in range(1, seq_len - 1):  # skip CLS / SEP
            attr_val = float(attr_scores[t_idx])
            char_s, char_e = offsets[t_idx]

            section = map_token_to_section(char_s, char_e, section_ranges)
            section_attr_sum[section] += attr_val
            section_token_count[section] += 1

            if char_s == 0 and char_e == 0:
                continue

            prev = note_text[char_s - 1] if char_s > 0 else ' '
            extend = (
                cur_word_s is not None
                and (prev.isalpha() or prev in "-'")
                and (char_s == cur_word_e
                     or (note_text[cur_word_e:char_s].strip() == ''
                         and all(c.isalpha() or c in "-'"
                                 for c in note_text[cur_word_e:char_s])))
            )
            if extend:
                cur_word_e = char_e
                cur_word_attr += attr_val
            else:
                if cur_word_s is not None and cur_word_e is not None:
                    w = note_text[cur_word_s:cur_word_e].strip().lower()
                    if len(w) >= 2 and all(c.isalpha() or c in "-'" for c in w):
                        word_attr_accum[w].append(cur_word_attr)
                cur_word_s = char_s; cur_word_e = char_e
                cur_word_attr = attr_val
        # flush the final word
        if cur_word_s is not None and cur_word_e is not None:
            w = note_text[cur_word_s:cur_word_e].strip().lower()
            if len(w) >= 2 and all(c.isalpha() or c in "-'" for c in w):
                word_attr_accum[w].append(cur_word_attr)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f'  Slice {slice_name}: {n_success} ok, {n_fail} fail '
          f'in {time.time() - t0:.0f}s')
    write_outputs(slice_name, section_attr_sum, section_token_count,
                  word_attr_accum, failures, n_success, n_fail)


def main():
    print('=' * 70)
    print('PTSD NLP — BioClinicalBERT Integrated Gradients (trunc + chunk-pool)')
    print('=' * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    os.makedirs(RESULTS_ATTRIBUTION, exist_ok=True)

    print(f'\n[1/4] Loading BERT from {BERT_DIR}')
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR, num_labels=2)
    model.to(device); model.eval()
    embed_layer = model.bert.embeddings.word_embeddings
    wrapper = BertEmbedForward(model); wrapper.eval()
    ig = IntegratedGradients(wrapper)

    print('\n[2/4] Selecting 50 high-confidence true positive patients ...')
    test_df = pd.read_parquet(TEST_PARQUET)
    pos = test_df[test_df['ptsd_label'] == 1].copy()

    # Score positives in truncated mode (fast, used to define the top decile)
    probs = []
    with torch.no_grad():
        for i in range(0, len(pos), 16):
            batch = pos['note_text'].iloc[i:i + 16].tolist()
            enc = tokenizer(batch, max_length=MAX_LEN, padding='max_length',
                            truncation=True, return_tensors='pt').to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(**enc).logits
            probs.extend(F.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    pos['pred_prob'] = probs
    cutoff = float(np.percentile(probs, 90))
    sample = pos[pos['pred_prob'] >= cutoff]
    if len(sample) > N_SAMPLES:
        sample = sample.sample(n=N_SAMPLES, random_state=42)
    print(f'  cutoff={cutoff:.4f} | n_sample={len(sample)}')

    print('\n[3/4] Loading raw discharge notes for section mapping ...')
    sample_hadms = set(sample['hadm_id'].tolist())
    raw_notes = {}
    with open(DISCHARGE_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                h = int(row['hadm_id'])
            except (KeyError, ValueError):
                continue
            if h in sample_hadms:
                raw_notes[h] = row['text']
                if len(raw_notes) == len(sample_hadms):
                    break
    print(f'  loaded {len(raw_notes)} raw notes')

    print('\n[4/4] Running IG slices')
    run_one_slice('trunc', model, tokenizer, ig, embed_layer, device,
                  sample, raw_notes)
    run_one_slice('chunkpool', model, tokenizer, ig, embed_layer, device,
                  sample, raw_notes)

    print('\nDone.')


if __name__ == '__main__':
    main()
