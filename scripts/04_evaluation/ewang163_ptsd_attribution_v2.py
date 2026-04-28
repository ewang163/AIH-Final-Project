"""
ewang163_ptsd_attribution_v2.py
===============================
Integrated Gradients attribution analysis — v2.

Fixes v1 failure (44/50 patients failed) by using IntegratedGradients on
the token embedding output directly, bypassing Captum's layer hooks that
conflict with Longformer's hybrid local/global attention mask.

Approach:
- Wrap model to accept (embeddings, attention_mask) → positive-class logit
- Compute IG on the embedding tensor with a pad-token-embedding baseline
- internal_batch_size=1, n_steps=50

Outputs:
    ewang163_attribution_by_section_v2.csv
    ewang163_top_attributed_tokens_v2.csv
    ewang163_attribution_failures_v2.log

Submit via SLURM:
    sbatch ewang163_ptsd_attribution_v2.sh
"""

import argparse
import csv
import os
import re
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

csv.field_size_limit(sys.maxsize)

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC              = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_ATTRIBUTION = f'{STUDENT_DIR}/results/attribution'

TEST_PARQUET  = f'{DATA_SPLITS}/ewang163_split_test.parquet'
DISCHARGE_F   = f'{MIMIC}/note/2.2/discharge.csv'

# Allow override via env var or CLI arg (set later in main()).
# Default: pi_p=0.25 winner via the symlink.
BEST_DIR      = os.environ.get('PTSD_MODEL_DIR',
                               f'{MODEL_DIR}/ewang163_longformer_best')
SUFFIX        = os.environ.get('PTSD_MODEL_SUFFIX', '')

SECTION_CSV   = f'{RESULTS_ATTRIBUTION}/ewang163_attribution_by_section_v2{SUFFIX}.csv'
TOKENS_CSV    = f'{RESULTS_ATTRIBUTION}/ewang163_top_attributed_tokens_v2{SUFFIX}.csv'
WORDS_CSV     = f'{RESULTS_ATTRIBUTION}/ewang163_top_attributed_words_v2{SUFFIX}.csv'
FAILURES_LOG  = f'{RESULTS_ATTRIBUTION}/ewang163_attribution_failures_v2{SUFFIX}.log'

# Fix 10: IG at full context length (4096), matching training input length.
# Previous MAX_LEN_IG=1024 only attributed the first quarter of each note,
# missing Brief Hospital Course content that appears later in long notes.
# n_steps reduced from 50 to 20 and internal_batch_size=1 for memory safety.
MAX_LEN_INFER = 4096
MAX_LEN_IG    = 4096
N_SAMPLES     = 50
N_IG_STEPS    = 20
IG_BATCH_SIZE = 1
TOP_K_TOKENS  = 50

# ── Section parsing (same as v1 / notes_extract.py) ─────────────────────
SECTION_HEADER_RE = re.compile(
    r'^([A-Z][A-Za-z /&\-]+):[ ]*$',
    re.MULTILINE,
)

INCLUDE_SECTIONS = {
    'history of present illness',
    'social history',
    'past medical history',
    'brief hospital course',
}

SECTION_ORDER = [
    'history of present illness',
    'social history',
    'past medical history',
    'brief hospital course',
]


def parse_section_spans(text):
    """Parse a discharge note and return {section_name: body_text}."""
    if not text:
        return {}
    headers = []
    for m in SECTION_HEADER_RE.finditer(text):
        headers.append((m.start(), m.end(), m.group(1).strip().lower()))

    spans = {}
    for i, (start, end, name) in enumerate(headers):
        if name not in INCLUDE_SECTIONS:
            continue
        body_start = end
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()
        if body and name not in spans:
            spans[name] = body
    return spans


def build_section_labeled_text(raw_text):
    """
    Re-extract sections from raw discharge text and return:
    - concatenated text (same order as training: HPI, social, PMH, BHC)
    - list of (section_name, char_start, char_end) in the concatenated text
    """
    sections = parse_section_spans(raw_text)
    parts = []
    section_ranges = []
    offset = 0
    for name in SECTION_ORDER:
        if name not in sections:
            continue
        body = sections[name]
        start = offset
        parts.append(body)
        end = offset + len(body)
        section_ranges.append((name, start, end))
        offset = end + 2  # for '\n\n' separator
    text = '\n\n'.join(parts)
    return text, section_ranges


def map_token_to_section(token_char_start, token_char_end, section_ranges):
    """Map a token's character span to a section name."""
    for name, sec_start, sec_end in section_ranges:
        if token_char_start >= sec_start and token_char_end <= sec_end:
            return name
    return 'unknown'


# ── Model wrapper for Captum ─────────────────────────────────────────────

class EmbeddingForwardWrapper(torch.nn.Module):
    """Wraps the Longformer so that the forward pass takes
    (input_embeds, attention_mask) and returns the positive-class logit.
    Captum's IntegratedGradients differentiates w.r.t. input_embeds."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_embeds, attention_mask):
        # Longformer auto-creates global_attention_mask via
        # torch.zeros_like(input_ids), but input_ids is None when using
        # inputs_embeds. We must provide it explicitly.
        global_attention_mask = torch.zeros(
            input_embeds.shape[0], input_embeds.shape[1],
            dtype=torch.long, device=input_embeds.device,
        )
        global_attention_mask[:, 0] = 1  # CLS token gets global attention
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        return outputs.logits[:, 1]  # positive-class logit (scalar per sample)


def main():
    print('=' * 65)
    print('PTSD NLP — Integrated Gradients Attribution v2')
    print('=' * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Load model ────────────────────────────────────────────────────────
    print('\n[1/5] Loading model and tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_DIR, num_labels=2
    )
    model.to(device)
    model.eval()

    # Embedding layer reference (for computing embeddings from input_ids)
    embed_layer = model.longformer.embeddings.word_embeddings

    # Wrapper: takes embeddings → positive logit
    wrapper = EmbeddingForwardWrapper(model)
    wrapper.eval()

    # IntegratedGradients on the embedding input (first positional arg)
    ig = IntegratedGradients(wrapper)
    print('  IntegratedGradients initialized on embedding inputs')

    # ── Load test data and select high-confidence TPs ─────────────────────
    print('\n[2/5] Selecting high-confidence true positive patients ...')
    test_df = pd.read_parquet(TEST_PARQUET)
    pos_df = test_df[test_df['ptsd_label'] == 1].copy()

    # Run quick inference to get predicted probabilities
    print('  Running inference on positive test patients ...')
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(pos_df), 4):
            batch_texts = pos_df['note_text'].iloc[i:i+4].tolist()
            enc = tokenizer(batch_texts, max_length=MAX_LEN_INFER, padding='max_length',
                            truncation=True, return_tensors='pt').to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(**enc)
            probs = F.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            del enc, outputs, probs
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pos_df = pos_df.copy()
    pos_df['pred_prob'] = all_probs

    # Top decile
    threshold = np.percentile(all_probs, 90)
    top_decile = pos_df[pos_df['pred_prob'] >= threshold]
    print(f'  Positive test patients: {len(pos_df)}')
    print(f'  Top decile threshold: {threshold:.4f}')
    print(f'  Patients in top decile: {len(top_decile)}')

    # Sample N_SAMPLES from top decile
    if len(top_decile) > N_SAMPLES:
        sample_df = top_decile.sample(n=N_SAMPLES, random_state=42)
    else:
        sample_df = top_decile
    print(f'  Selected for attribution: {len(sample_df)}')

    # ── Load raw discharge notes for section mapping ──────────────────────
    print('\n[3/5] Loading raw discharge notes for section boundaries ...')
    sample_hadms = set(sample_df['hadm_id'].tolist())
    raw_notes = {}

    with open(DISCHARGE_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hadm_id = int(row['hadm_id'])
            if hadm_id in sample_hadms:
                raw_notes[hadm_id] = row['text']
                if len(raw_notes) == len(sample_hadms):
                    break

    print(f'  Loaded {len(raw_notes)} raw notes')

    # ── Compute IG attributions ───────────────────────────────────────────
    print('\n[4/5] Computing Integrated Gradients (this may take a while) ...')

    # Accumulators
    section_attr_sum = defaultdict(float)    # section → sum of |attr|
    section_token_count = defaultdict(int)   # section → total tokens
    token_attr_accum = defaultdict(list)     # clean_token → list of |attr|  (subword level)
    word_attr_accum = defaultdict(list)      # whole_word → list of summed |attr| (Fix 5)

    n_success = 0
    n_fail = 0
    failures = []
    t0 = time.time()

    for pat_idx, (idx, row) in enumerate(sample_df.iterrows(), 1):
        hadm_id = row['hadm_id']
        note_text = row['note_text']

        if pat_idx % 10 == 0:
            elapsed = time.time() - t0
            print(f'  [{pat_idx}/{len(sample_df)}] '
                  f'{elapsed:.0f}s elapsed, {n_success} ok, {n_fail} fail',
                  flush=True)

        # Build section ranges from raw note
        raw = raw_notes.get(hadm_id, None)
        if raw:
            _, section_ranges = build_section_labeled_text(raw)
        else:
            section_ranges = []

        # Tokenize at reduced length for IG
        enc = tokenizer(
            note_text,
            max_length=MAX_LEN_IG,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
        )
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        offsets = enc['offset_mapping'][0].tolist()  # list of (start, end)

        # Compute input embeddings (what IG will differentiate w.r.t.)
        with torch.no_grad():
            input_embeds = embed_layer(input_ids)  # (1, seq_len, embed_dim)

        # Baseline: pad token embeddings with zero attention mask
        baseline_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        with torch.no_grad():
            baseline_embeds = embed_layer(baseline_input_ids)  # (1, seq_len, embed_dim)
        baseline_attention_mask = torch.zeros_like(attention_mask)

        # Require grad for IG
        input_embeds = input_embeds.clone().detach().requires_grad_(True)

        try:
            attributions = ig.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                n_steps=N_IG_STEPS,
                internal_batch_size=IG_BATCH_SIZE,
            )
        except Exception as e:
            err_msg = f'hadm_id={hadm_id}: {e}'
            print(f'    WARNING: IG failed — {err_msg}')
            failures.append(err_msg)
            n_fail += 1
            del input_ids, attention_mask, input_embeds, baseline_embeds
            del baseline_input_ids, baseline_attention_mask, enc
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue

        n_success += 1

        # attributions shape: (1, seq_len, embed_dim) — sum over embed dim
        attr_scores = attributions.squeeze(0).sum(dim=-1).abs().detach().cpu().numpy()

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        # Map each token to section and accumulate
        seq_len = int(attention_mask[0].sum().item())

        # Fix 5: word-level aggregation using offset_mapping + source text
        # Group contiguous subword tokens that map to the same word in note_text.
        # A "word" is a maximal span of alphabetic characters; we collect all
        # token attributions whose offsets fall within each such span.
        current_word_start = None
        current_word_end = None
        current_word_attr = 0.0

        def _flush_current_word():
            nonlocal current_word_start, current_word_end, current_word_attr
            if current_word_start is not None and current_word_end is not None:
                word = note_text[current_word_start:current_word_end].strip().lower()
                # Keep only alphabetic words of length >= 2
                if len(word) >= 2 and all(c.isalpha() or c in "-'" for c in word):
                    word_attr_accum[word].append(current_word_attr)
            current_word_start = None
            current_word_end = None
            current_word_attr = 0.0

        for t_idx in range(1, seq_len - 1):  # skip CLS and SEP
            token = tokens[t_idx]
            attr_val = float(attr_scores[t_idx])
            char_start, char_end = offsets[t_idx]

            # Section mapping (unchanged)
            section = map_token_to_section(char_start, char_end, section_ranges)
            section_attr_sum[section] += attr_val
            section_token_count[section] += 1

            # Subword accumulation (unchanged — kept for comparison)
            clean = token.replace('Ġ', '').strip().lower()
            if len(clean) >= 2 and clean.isalpha():
                token_attr_accum[clean].append(attr_val)

            # Word-level aggregation via offsets (Fix 5)
            # Special tokens have char_start == char_end == 0
            if char_start == 0 and char_end == 0:
                continue

            # Inspect the character immediately before this token:
            # if it's alphabetic and we have a current word, extend it.
            # Otherwise, flush and start a new word.
            prev_char = note_text[char_start - 1] if char_start > 0 else ' '
            token_is_continuation = (
                current_word_start is not None
                and (prev_char.isalpha() or prev_char in "-'")
                and (char_start == current_word_end
                     or (note_text[current_word_end:char_start].strip() == ''
                         and all(c.isalpha() or c in "-'"
                                 for c in note_text[current_word_end:char_start])))
            )

            if token_is_continuation:
                current_word_end = char_end
                current_word_attr += attr_val
            else:
                _flush_current_word()
                current_word_start = char_start
                current_word_end = char_end
                current_word_attr = attr_val

        _flush_current_word()

        # Free GPU memory between patients
        del input_ids, attention_mask, input_embeds, baseline_embeds
        del baseline_input_ids, baseline_attention_mask, enc, attributions, attr_scores
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f'\n  Completed: {n_success} succeeded, {n_fail} failed '
          f'out of {len(sample_df)} patients in {elapsed:.0f}s')

    # ── Write failure log ─────────────────────────────────────────────────
    with open(FAILURES_LOG, 'w') as f:
        for line in failures:
            f.write(line + '\n')
    print(f'  Failures log: {FAILURES_LOG}')

    # ── Aggregate and save results ────────────────────────────────────────
    print('\n[5/5] Aggregating results ...')

    if n_success == 0:
        print('  ERROR: No patients succeeded. Cannot produce results.')
        return

    # Section-level attribution
    section_rows = []
    total_attr = sum(section_attr_sum.values())
    for name in SECTION_ORDER + ['unknown']:
        if name not in section_attr_sum:
            continue
        attr_sum = section_attr_sum[name]
        tok_count = section_token_count[name]
        section_rows.append({
            'section': name,
            'total_attribution': round(attr_sum, 4),
            'pct_of_total': round(attr_sum / total_attr * 100, 2) if total_attr > 0 else 0,
            'n_tokens': tok_count,
            'mean_attribution_per_token': round(attr_sum / tok_count, 6) if tok_count > 0 else 0,
        })

    section_df = pd.DataFrame(section_rows)
    section_df = section_df.sort_values('total_attribution', ascending=False)
    section_df.to_csv(SECTION_CSV, index=False)

    print(f'\n  Section-level attribution:')
    print(f'  {"Section":<30} {"Total":>10} {"% Total":>8} {"Tokens":>8} {"Mean/Token":>12}')
    print(f'  {"-"*30} {"-"*10} {"-"*8} {"-"*8} {"-"*12}')
    for _, r in section_df.iterrows():
        print(f'  {r["section"]:<30} {r["total_attribution"]:>10.4f} '
              f'{r["pct_of_total"]:>7.2f}% {r["n_tokens"]:>8} '
              f'{r["mean_attribution_per_token"]:>12.6f}')
    print(f'  → {SECTION_CSV}')

    # Top tokens by mean absolute attribution (min 3 occurrences)
    token_rows = []
    for token, attrs in token_attr_accum.items():
        if len(attrs) >= 3:
            token_rows.append({
                'token': token,
                'mean_abs_attribution': round(float(np.mean(attrs)), 6),
                'std_attribution': round(float(np.std(attrs)), 6),
                'n_occurrences': len(attrs),
                'total_attribution': round(float(np.sum(attrs)), 4),
            })

    token_df = pd.DataFrame(token_rows)
    token_df = token_df.sort_values('mean_abs_attribution', ascending=False)
    top_tokens = token_df.head(TOP_K_TOKENS)
    top_tokens.to_csv(TOKENS_CSV, index=False)

    print(f'\n  Top 20 tokens by mean |attribution| (min 3 occurrences):')
    print(f'  {"Token":<20} {"Mean |Attr|":>12} {"Std":>10} {"N":>6}')
    print(f'  {"-"*20} {"-"*12} {"-"*10} {"-"*6}')
    for _, r in top_tokens.head(20).iterrows():
        print(f'  {r["token"]:<20} {r["mean_abs_attribution"]:>12.6f} '
              f'{r["std_attribution"]:>10.6f} {r["n_occurrences"]:>6}')
    print(f'  → {TOKENS_CSV}')

    # ── Fix 5: Word-level attribution (whole-word aggregation) ────────────
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
        word_df = word_df.sort_values('mean_abs_attribution', ascending=False)
        top_words = word_df.head(TOP_K_TOKENS)
        top_words.to_csv(WORDS_CSV, index=False)

        print(f'\n  Top 20 WORDS by mean |attribution| (min 3 occurrences) — Fix 5:')
        print(f'  {"Word":<25} {"Mean |Attr|":>12} {"Std":>10} {"N":>6}')
        print(f'  {"-"*25} {"-"*12} {"-"*10} {"-"*6}')
        for _, r in top_words.head(20).iterrows():
            print(f'  {r["word"]:<25} {r["mean_abs_attribution"]:>12.6f} '
                  f'{r["std_attribution"]:>10.6f} {r["n_occurrences"]:>6}')
        print(f'  → {WORDS_CSV}')

    # ── Interpretation ────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('INTERPRETATION')
    print('=' * 65)
    if len(section_df) > 0:
        top_section = section_df.iloc[0]['section']
        top_pct = section_df.iloc[0]['pct_of_total']
        print(f'  Highest-attribution section: {top_section} ({top_pct:.1f}% of total)')
    if len(word_df) > 0:
        top_3_words = ', '.join(word_df['word'].head(3).tolist())
        print(f'  Top 3 attributed WORDS (Fix 5): {top_3_words}')
    if len(top_tokens) > 0:
        top_3 = ', '.join(top_tokens['token'].head(3).tolist())
        print(f'  Top 3 attributed subword tokens: {top_3}')
    print(f'\n  Expected: social history and HPI sections should drive')
    print(f'  predictions if the model learns genuine clinical signal.')
    print(f'  High PMH attribution may indicate residual label leakage.')

    print('\nDone.')


if __name__ == '__main__':
    main()
