"""
ewang163_ptsd_attribution.py
============================
Integrated Gradients attribution analysis for the Longformer model.

Uses Captum's LayerIntegratedGradients on the embedding layer —
NOT attention weights (Jain & Wallace 2019 show attention != explanation).
IG satisfies completeness and sensitivity axioms (Sundararajan et al. 2017).

1. Sample 50 true-positive test patients with high predicted probability
2. Compute token-level IG attributions per patient
3. Aggregate by note section (HPI, social history, PMH, brief hospital course)
4. Rank top 50 tokens/phrases by mean absolute attribution

Outputs:
    ewang163_attribution_by_section.csv
    ewang163_top_attributed_tokens.csv

Submit via SLURM:
    sbatch ewang163_ptsd_attribution.sh
"""

import csv
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

csv.field_size_limit(sys.maxsize)

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC              = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_ATTRIBUTION = f'{STUDENT_DIR}/results/attribution'

TEST_PARQUET  = f'{DATA_SPLITS}/ewang163_split_test.parquet'
BEST_DIR      = f'{MODEL_DIR}/ewang163_longformer_best'
DISCHARGE_F   = f'{MIMIC}/note/2.2/discharge.csv'

SECTION_CSV   = f'{RESULTS_ATTRIBUTION}/ewang163_attribution_by_section.csv'
TOKENS_CSV    = f'{RESULTS_ATTRIBUTION}/ewang163_top_attributed_tokens.csv'

MAX_LEN_INFER = 4096  # for initial probability ranking
MAX_LEN_IG    = 1024  # truncated for IG (memory scales with seq_len * n_steps)
N_SAMPLES     = 50
N_IG_STEPS    = 20    # interpolation steps (reduced from 50 to fit in 24GB GPU)
IG_BATCH_SIZE = 5     # internal_batch_size for Captum (process steps in batches)
TOP_K_TOKENS  = 50

# ── Section parsing (same regex as notes_extract.py) ──────────────────────
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
    """
    Parse a discharge note and return {section_name: (char_start, char_end)}
    for included sections. char_start/end refer to the body text positions.
    """
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
def make_forward_func(model):
    """Create a forward function for Captum that takes input_ids + attention_mask
    and returns the positive-class logit. LayerIntegratedGradients hooks into the
    embedding layer internally — the forward function must use input_ids, NOT
    inputs_embeds."""
    def forward_func(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, 1]
    return forward_func


def main():
    print('=' * 65)
    print('PTSD NLP — Integrated Gradients Attribution Analysis')
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

    # Get embedding layer reference for LayerIntegratedGradients
    embed_layer = model.longformer.embeddings.word_embeddings

    forward_fn = make_forward_func(model)
    lig = LayerIntegratedGradients(forward_fn, embed_layer)
    print('  LayerIntegratedGradients initialized on word_embeddings layer')

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
    torch.cuda.empty_cache() if device.type == 'cuda' else None

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
    token_attr_accum = defaultdict(list)     # clean_token → list of |attr|

    n_processed = 0
    t0 = time.time()

    for idx, row in sample_df.iterrows():
        hadm_id = row['hadm_id']
        note_text = row['note_text']
        n_processed += 1

        if n_processed % 10 == 0:
            elapsed = time.time() - t0
            print(f'  [{n_processed}/{len(sample_df)}] '
                  f'{elapsed:.0f}s elapsed', flush=True)

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
        offsets = enc['offset_mapping'][0].tolist()  # list of (start, end) char offsets

        # Baseline: PAD token IDs (LIG handles embedding internally)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

        try:
            attributions = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                n_steps=N_IG_STEPS,
                internal_batch_size=IG_BATCH_SIZE,
            )
        except Exception as e:
            print(f'    WARNING: IG failed for hadm_id={hadm_id}: {e}')
            del input_ids, attention_mask, baseline_ids, enc
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            continue

        # attributions shape: (1, seq_len, embed_dim) — sum over embed dim
        attr_scores = attributions.squeeze(0).sum(dim=-1).abs().cpu().numpy()

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        # Map each token to section and accumulate
        seq_len = int(attention_mask[0].sum().item())
        for t_idx in range(1, seq_len - 1):  # skip CLS and SEP
            token = tokens[t_idx]
            attr_val = float(attr_scores[t_idx])
            char_start, char_end = offsets[t_idx]

            # Section mapping
            section = map_token_to_section(char_start, char_end, section_ranges)
            section_attr_sum[section] += attr_val
            section_token_count[section] += 1

            # Token accumulation (clean up Ġ prefix for readability)
            clean = token.replace('Ġ', '').strip().lower()
            if len(clean) >= 2 and clean.isalpha():
                token_attr_accum[clean].append(attr_val)

        # Free GPU memory between patients
        del input_ids, attention_mask, baseline_ids, enc, attributions, attr_scores
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    elapsed = time.time() - t0
    print(f'  Completed {n_processed} patients in {elapsed:.0f}s')

    # ── Aggregate and save results ────────────────────────────────────────
    print('\n[5/5] Aggregating results ...')

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

    print(f'\n  Top {TOP_K_TOKENS} tokens by mean |attribution| (min 3 occurrences):')
    print(f'  {"Token":<20} {"Mean |Attr|":>12} {"Std":>10} {"N":>6}')
    print(f'  {"-"*20} {"-"*12} {"-"*10} {"-"*6}')
    for _, r in top_tokens.head(25).iterrows():
        print(f'  {r["token"]:<20} {r["mean_abs_attribution"]:>12.6f} '
              f'{r["std_attribution"]:>10.6f} {r["n_occurrences"]:>6}')
    print(f'  → {TOKENS_CSV}')

    # ── Interpretation ────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('INTERPRETATION')
    print('=' * 65)
    if len(section_df) > 0:
        top_section = section_df.iloc[0]['section']
        top_pct = section_df.iloc[0]['pct_of_total']
        print(f'  Highest-attribution section: {top_section} ({top_pct:.1f}% of total)')
    if len(top_tokens) > 0:
        top_3 = ', '.join(top_tokens['token'].head(3).tolist())
        print(f'  Top 3 attributed tokens: {top_3}')
    print(f'\n  Expected: social history and HPI sections should drive')
    print(f'  predictions if the model learns genuine clinical signal.')
    print(f'  High PMH attribution may indicate residual label leakage.')

    print('\nDone.')


if __name__ == '__main__':
    main()
