"""
ewang163_ptsd_ablations.py
==========================
Runs two ablation experiments on the held-out test set using the best
Longformer checkpoint, to verify the model is not relying on leaked
label information.

Ablation 1 — PTSD string masking:
    Apply PTSD-related regex masking to ALL test notes, then run inference.

Ablation 2 — PMH section removal:
    Re-extract test notes from discharge.csv WITHOUT the 'past medical history'
    section, then run inference.

Baseline: unmasked test notes (as stored in the split parquet).

For each condition, reports: AUPRC, AUROC, sensitivity, specificity,
precision, F1 at the threshold calibrated for recall >= 0.85.

Outputs:
    ewang163_ablation_results.csv

Submit via SLURM:
    sbatch ewang163_ptsd_ablations.sh
"""

import csv
import sys
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import average_precision_score, roc_auc_score

csv.field_size_limit(sys.maxsize)

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC = '/oscar/data/shared/ursa/mimic-iv'
OUT   = '/oscar/data/class/biol1595_2595/students/ewang163'

DISCHARGE_F   = f'{MIMIC}/note/2.2/discharge.csv'
TEST_PARQUET  = f'{OUT}/ewang163_split_test.parquet'
ADM_PARQUET   = f'{OUT}/ewang163_ptsd_adm_extract.parquet'
BEST_DIR      = f'{OUT}/ewang163_longformer_best'
RESULTS_CSV   = f'{OUT}/ewang163_ablation_results.csv'

MAX_LEN    = 4096
BATCH_SIZE = 4

# ── Ablation 1: PTSD masking patterns (from corpus_build.py) ─────────────
MASK_PATTERNS = [
    r'post-traumatic',
    r'post\s+traumatic',
    r'posttraumatic',
    r'trauma-related\s+stress',
    r'ptsd',
    r'f43\.1',
    r'309\.81',
]
MASK_RE = re.compile('|'.join(MASK_PATTERNS), re.IGNORECASE)
MASK_TOKEN = '[PTSD_MASKED]'


def apply_masking(text):
    """Replace all PTSD-related strings with [PTSD_MASKED]."""
    return MASK_RE.sub(MASK_TOKEN, text)


# ── Ablation 2: Section parsing (from notes_extract.py) ──────────────────
SECTION_HEADER_RE = re.compile(
    r'^([A-Z][A-Za-z /&\-]+):[ ]*$',
    re.MULTILINE,
)

# Sections to keep for Ablation 2 — same as extraction but WITHOUT PMH
INCLUDE_SECTIONS_NO_PMH = {
    'history of present illness',
    'social history',
    'brief hospital course',
}

# Full section set for baseline comparison
INCLUDE_SECTIONS_ALL = {
    'history of present illness',
    'social history',
    'past medical history',
    'brief hospital course',
}

SECTION_ORDER_NO_PMH = [
    'history of present illness',
    'social history',
    'brief hospital course',
]


def parse_sections(text, include_set):
    """Parse a discharge note and return only sections in include_set."""
    if not text:
        return {}
    headers = []
    for m in SECTION_HEADER_RE.finditer(text):
        headers.append((m.start(), m.end(), m.group(1).strip().lower()))

    result = {}
    for i, (start, end, name) in enumerate(headers):
        if name not in include_set:
            continue
        body_start = end
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()
        if body and name not in result:
            result[name] = body
    return result


def concatenate_sections(sections_dict, order):
    """Join section texts in canonical order."""
    parts = []
    for name in order:
        if name in sections_dict:
            parts.append(sections_dict[name])
    return '\n\n'.join(parts)


# ── Dataset ───────────────────────────────────────────────────────────────
class NoteDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, texts, labels, tokenizer, device):
    """Run inference and return predicted probabilities and true labels."""
    ds = NoteDataset(texts, labels, tokenizer, MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_probs = []
    all_labels = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch['label'].numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(probs, labels, condition_name):
    """Compute AUPRC, AUROC, and threshold-calibrated metrics at recall>=0.85."""
    auprc = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs)

    # Find threshold for recall >= 0.85
    thresholds = np.linspace(1.0, 0.0, 1001)
    best_thresh = 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if recall >= 0.85:
            best_thresh = t
            break

    preds = (probs >= best_thresh).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1          = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return {
        'condition':   condition_name,
        'AUPRC':       round(auprc, 4),
        'AUROC':       round(auroc, 4),
        'threshold':   round(best_thresh, 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'precision':   round(precision, 4),
        'F1':          round(f1, 4),
        'n_samples':   len(labels),
        'n_pos':       int(labels.sum()),
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('PTSD NLP — Ablation Experiments')
    print('=' * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Load model and tokenizer ──────────────────────────────────────────
    print('\n[1/5] Loading best Longformer checkpoint ...')
    tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_DIR, num_labels=2
    )
    model.to(device)
    print('  Model loaded.')

    # ── Load test data ────────────────────────────────────────────────────
    print('\n[2/5] Loading test split ...')
    test_df = pd.read_parquet(TEST_PARQUET)
    labels = test_df['ptsd_label'].values.astype(np.int64)
    baseline_texts = test_df['note_text'].tolist()
    print(f'  {len(test_df):,} test notes '
          f'(pos={int(labels.sum()):,}, unl={int((labels == 0).sum()):,})')

    results = []

    # ── Baseline: unmasked test notes ─────────────────────────────────────
    print('\n[3/5] Baseline inference (unmasked) ...')
    t0 = time.time()
    probs_base, labels_base = run_inference(
        model, baseline_texts, labels, tokenizer, device
    )
    t_base = time.time() - t0
    metrics_base = compute_metrics(probs_base, labels_base, 'Baseline (unmasked)')
    results.append(metrics_base)
    print(f'  AUPRC={metrics_base["AUPRC"]:.4f}  '
          f'AUROC={metrics_base["AUROC"]:.4f}  '
          f'({t_base:.0f}s)')

    # ── Ablation 1: PTSD string masking ───────────────────────────────────
    print('\n[4/5] Ablation 1 — PTSD string masking ...')
    masked_texts = [apply_masking(t) for t in baseline_texts]
    n_changed = sum(1 for orig, masked in zip(baseline_texts, masked_texts)
                    if orig != masked)
    print(f'  Notes with substitutions: {n_changed:,} / {len(masked_texts):,}')

    t0 = time.time()
    probs_abl1, labels_abl1 = run_inference(
        model, masked_texts, labels, tokenizer, device
    )
    t_abl1 = time.time() - t0
    metrics_abl1 = compute_metrics(probs_abl1, labels_abl1,
                                   'Ablation 1: PTSD masking')
    results.append(metrics_abl1)
    print(f'  AUPRC={metrics_abl1["AUPRC"]:.4f}  '
          f'AUROC={metrics_abl1["AUROC"]:.4f}  '
          f'({t_abl1:.0f}s)')
    print(f'  AUPRC delta: {metrics_abl1["AUPRC"] - metrics_base["AUPRC"]:+.4f}')

    # ── Ablation 2: PMH section removal ───────────────────────────────────
    # Re-extract test notes from discharge.csv without PMH section
    print('\n[5/5] Ablation 2 — PMH section removal ...')
    print('  Re-extracting test notes without PMH from discharge.csv ...')

    # Get hadm_ids for test patients
    test_hadm_ids = set(test_df['hadm_id'].tolist())
    # Map hadm_id → row index in test_df for ordering
    hadm_to_idx = {row['hadm_id']: i for i, (_, row) in
                   enumerate(test_df.iterrows())}

    # Stream discharge.csv, re-parse sections without PMH
    no_pmh_texts = [''] * len(test_df)
    matched = 0

    with open(DISCHARGE_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hadm_id = int(row['hadm_id'])
            if hadm_id not in test_hadm_ids:
                continue

            sections = parse_sections(row['text'], INCLUDE_SECTIONS_NO_PMH)
            text = concatenate_sections(sections, SECTION_ORDER_NO_PMH)
            idx = hadm_to_idx[hadm_id]
            no_pmh_texts[idx] = text if text.strip() else baseline_texts[idx]
            matched += 1

            if matched == len(test_hadm_ids):
                break  # found all, stop streaming

    # Any hadm_ids not found in discharge.csv → keep baseline text
    n_fallback = sum(1 for t in no_pmh_texts if t == '')
    if n_fallback > 0:
        print(f'  WARNING: {n_fallback} notes not found in discharge.csv, '
              f'using baseline text')
        for i, t in enumerate(no_pmh_texts):
            if t == '':
                no_pmh_texts[i] = baseline_texts[i]

    print(f'  Re-extracted {matched:,} / {len(test_df):,} test notes without PMH')

    t0 = time.time()
    probs_abl2, labels_abl2 = run_inference(
        model, no_pmh_texts, labels, tokenizer, device
    )
    t_abl2 = time.time() - t0
    metrics_abl2 = compute_metrics(probs_abl2, labels_abl2,
                                   'Ablation 2: PMH removed')
    results.append(metrics_abl2)
    print(f'  AUPRC={metrics_abl2["AUPRC"]:.4f}  '
          f'AUROC={metrics_abl2["AUROC"]:.4f}  '
          f'({t_abl2:.0f}s)')
    print(f'  AUPRC delta: {metrics_abl2["AUPRC"] - metrics_base["AUPRC"]:+.4f}')

    # ── Save results ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f'\nResults saved → {RESULTS_CSV}')

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('ABLATION RESULTS SUMMARY')
    print('=' * 65)
    print(f'\n  {"Condition":<30} {"AUPRC":>7} {"AUROC":>7} '
          f'{"Sens":>6} {"Spec":>6} {"Prec":>6} {"F1":>6}')
    print(f'  {"-"*30} {"-"*7} {"-"*7} {"-"*6} {"-"*6} {"-"*6} {"-"*6}')
    for r in results:
        print(f'  {r["condition"]:<30} {r["AUPRC"]:>7.4f} {r["AUROC"]:>7.4f} '
              f'{r["sensitivity"]:>6.4f} {r["specificity"]:>6.4f} '
              f'{r["precision"]:>6.4f} {r["F1"]:>6.4f}')

    print(f'\n  Interpretation:')
    d1 = metrics_abl1['AUPRC'] - metrics_base['AUPRC']
    d2 = metrics_abl2['AUPRC'] - metrics_base['AUPRC']
    print(f'    Ablation 1 (PTSD masking) AUPRC delta: {d1:+.4f}')
    print(f'    Ablation 2 (PMH removal)  AUPRC delta: {d2:+.4f}')
    if abs(d1) < 0.02:
        print(f'    → Ablation 1: minimal drop — model does NOT rely on '
              f'PTSD keyword leakage.')
    else:
        print(f'    → Ablation 1: notable drop — model may be partially '
              f'relying on PTSD keyword leakage.')
    if abs(d2) < 0.02:
        print(f'    → Ablation 2: minimal drop — model does NOT rely heavily '
              f'on PMH section.')
    else:
        print(f'    → Ablation 2: notable drop — model relies meaningfully '
              f'on PMH section content.')

    print('\nDone.')


if __name__ == '__main__':
    main()
