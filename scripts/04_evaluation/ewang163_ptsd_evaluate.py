"""
ewang163_ptsd_evaluate.py
=========================
LEGACY all-baselines evaluator. The canonical per-model evaluation flow now
uses three dedicated scripts:

  - ewang163_ptsd_pulsnar_reeval.py   Clinical Longformer (PULSNAR) — primary
  - ewang163_ptsd_bert_full_eval.py   BioClinicalBERT (truncated + chunk-pool)
  - ewang163_ptsd_cross_model.py      cross-model comparisons

This script is kept because it still produces the val-derived thresholds for
the structured logistic and keyword baselines (read by cross_model.py from
results/metrics/ewang163_evaluation_results.json::val_thresholds). Do not rely
on its Longformer / BERT numbers — the dedicated scripts above are the ones
referenced in writeups.

Models still scored here (val + test):
  1. Clinical Longformer (the symlinked best — currently PULSNAR)
  2. BioClinicalBERT (truncated + chunk-pool variants)
  3. Structured features only (logistic regression baseline)
  4. Keyword/phrase-lookup baseline (zero-training)

All thresholds are derived from the VALIDATION set and frozen before any
test-set metrics are computed. This eliminates selection-on-test bias for
sensitivity/specificity/F1/PPV.

Reports per model:
  AUPRC, AUROC, precision/recall/F1/specificity at val-derived threshold
  Calibration curve (10 decile bins)
  McNemar's test p-value vs. Longformer

Clinical utility metrics (Longformer):
  Prevalence recalibration (PPV + NNS at 1%, 2%, 5%)
  Number needed to evaluate (NNE)
  Alert rate at operating threshold
  Positive/negative likelihood ratios (LR+, LR-)
  Clinical workup reduction (vs. treat-all)
  Diagnostic odds ratio (DOR)

Subgroup analysis (Longformer):
  AUPRC by sex, age group, race/ethnicity, emergency vs. non-emergency

Outputs:
    ewang163_evaluation_results.json
    ewang163_evaluation_summary.csv

Submit via SLURM:
    sbatch ewang163_ptsd_evaluate.sh
"""

import csv
import json
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import average_precision_score, roc_auc_score

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC              = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
DATA_NOTES         = f'{STUDENT_DIR}/data/notes'
DATA_COHORT        = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_METRICS    = f'{STUDENT_DIR}/results/metrics'
RESULTS_PREDICTIONS = f'{STUDENT_DIR}/results/predictions'

VAL_PARQUET       = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET      = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET       = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
LONGFORMER_DIR    = f'{MODEL_DIR}/ewang163_longformer_best'
BIOCLINBERT_DIR   = f'{MODEL_DIR}/ewang163_bioclinbert_best'   # may not exist
STRUCT_LR_PKL     = f'{MODEL_DIR}/ewang163_structured_logreg.pkl'
STRUCT_FEAT_JSON  = f'{MODEL_DIR}/ewang163_structured_features.json'

DIAGNOSES_F       = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'
PRESCRIPTIONS_F   = f'{MIMIC}/hosp/3.1/prescriptions.csv'

RESULTS_JSON = f'{RESULTS_METRICS}/ewang163_evaluation_results.json'
SUMMARY_CSV  = f'{RESULTS_METRICS}/ewang163_evaluation_summary.csv'

MAX_LEN    = 4096
BATCH_SIZE = 4

# ── ICD comorbidity prefixes (from CLAUDE.md) ────────────────────────────
COMORBIDITY_PFXS = {
    'dx_MDD':      ['F32', 'F33', '296'],
    'dx_anxiety':  ['F41', '300'],
    'dx_SUD':      ['F10','F11','F12','F13','F14','F15','F16','F17','F18','F19',
                    '303','304','305'],
    'dx_TBI':      ['S06','800','801','802','803','804','850','851','852','853','854'],
    'dx_pain':     ['G89', '338'],
    'dx_suicidal': ['R458', 'V6284', 'E95'],
}

DRUG_PATTERNS = {
    'rx_ssri_snri': [
        'sertraline', 'fluoxetine', 'paroxetine', 'escitalopram', 'citalopram',
        'fluvoxamine', 'venlafaxine', 'duloxetine', 'desvenlafaxine',
        'levomilnacipran', 'milnacipran',
    ],
    'rx_prazosin': ['prazosin'],
    'rx_SGA': [
        'quetiapine', 'olanzapine', 'risperidone', 'aripiprazole', 'ziprasidone',
        'clozapine', 'lurasidone', 'asenapine', 'paliperidone', 'iloperidone',
        'brexpiprazole', 'cariprazine',
    ],
}


# ── Race mapping (same as structured training script) ─────────────────────
def map_race(race_str):
    if pd.isna(race_str):
        return 'Other/Unknown'
    r = str(race_str).upper()
    if 'WHITE' in r:
        return 'White'
    elif 'BLACK' in r or 'AFRICAN' in r:
        return 'Black'
    elif 'HISPANIC' in r or 'LATINO' in r:
        return 'Hispanic'
    elif 'ASIAN' in r:
        return 'Asian'
    else:
        return 'Other/Unknown'


def age_decade(age):
    if pd.isna(age):
        return 'Other'
    a = int(age)
    if 20 <= a <= 29: return '20s'
    if 30 <= a <= 39: return '30s'
    if 40 <= a <= 49: return '40s'
    if 50 <= a <= 59: return '50s'
    return 'Other'


# ── Dataset for transformer inference ─────────────────────────────────────
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


@torch.no_grad()
def run_transformer_inference(model, tokenizer, texts, labels, device, max_len):
    """Run inference with a transformer model. Returns (probs, labels)."""
    ds = NoteDataset(texts, labels, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch['label'].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def run_chunk_pool_inference(model, tokenizer, texts, labels, device,
                             chunk_len=512, stride=256, pool='max'):
    """Fix 8: chunk-and-pool inference for BERT models on long texts.

    Splits each note into overlapping windows, runs inference on each,
    and aggregates via max-pool (default) or mean-pool.
    """
    model.eval()
    all_probs = np.zeros(len(texts))

    for i, text in enumerate(texts):
        tokens = tokenizer(text, return_tensors='pt', truncation=False,
                           add_special_tokens=False)
        input_ids_full = tokens['input_ids'].squeeze(0)
        n_tokens = len(input_ids_full)

        if n_tokens <= chunk_len - 2:
            enc = tokenizer(text, max_length=chunk_len, padding='max_length',
                            truncation=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prob = F.softmax(outputs.logits.float(), dim=-1)[0, 1].item()
            all_probs[i] = prob
            continue

        cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 101
        sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 102
        usable = chunk_len - 2

        chunk_probs = []
        start = 0
        while start < n_tokens:
            end = min(start + usable, n_tokens)
            chunk_ids = torch.cat([
                torch.tensor([cls_id]),
                input_ids_full[start:end],
                torch.tensor([sep_id]),
            ])
            attn_mask = torch.ones(len(chunk_ids), dtype=torch.long)

            pad_len = chunk_len - len(chunk_ids)
            if pad_len > 0:
                chunk_ids = torch.cat([chunk_ids, torch.zeros(pad_len, dtype=torch.long)])
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])

            chunk_ids = chunk_ids.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(input_ids=chunk_ids, attention_mask=attn_mask)
            prob = F.softmax(outputs.logits.float(), dim=-1)[0, 1].item()
            chunk_probs.append(prob)

            if end >= n_tokens:
                break
            start += stride

        if pool == 'max':
            all_probs[i] = max(chunk_probs)
        else:
            all_probs[i] = sum(chunk_probs) / len(chunk_probs)

        if (i + 1) % 100 == 0:
            print(f'    Chunk-pool inference: {i+1}/{len(texts)}', flush=True)

    return all_probs, np.array(labels)


# ── PU-corrected metrics (Ramola et al. 2019, Fix 6) ─────────────────────
def ramola_corrected_metrics(auprc_raw, auroc_raw, sens_raw, spec_raw,
                             prec_raw, pi_p):
    """Compute Ramola et al. (2019) PU-corrected performance estimates.

    In a PU setting, treating unlabeled as negative systematically
    *underestimates* model performance because correct detections of
    hidden positives are counted as false positives.

    Ramola et al. (Pac Symp Biocomput 2019, PMID 30864316) provide
    correction formulas parameterized by the class prior pi_p.

    Returns a dict of corrected metrics (all are lower bounds on true
    performance when pi_p is itself a lower bound).
    """
    if pi_p <= 0 or pi_p >= 1:
        return {}

    alpha = pi_p

    corrected_auroc = (auroc_raw - 0.5 * alpha) / (1 - alpha) if (1 - alpha) > 0 else auroc_raw

    corrected_prec = prec_raw / (prec_raw + (1 - prec_raw) * (1 - alpha)) if prec_raw > 0 else 0

    corrected_auprc = auprc_raw / alpha if alpha > 0 else auprc_raw
    corrected_auprc = min(corrected_auprc, 1.0)

    corrected_sens = sens_raw

    fp_rate_raw = 1 - spec_raw
    corrected_fp_rate = max(0, fp_rate_raw - alpha * sens_raw) / (1 - alpha) if (1 - alpha) > 0 else fp_rate_raw
    corrected_spec = 1 - corrected_fp_rate

    return {
        'AUPRC_PU_corrected': round(min(corrected_auprc, 1.0), 4),
        'AUROC_PU_corrected': round(min(max(corrected_auroc, 0.5), 1.0), 4),
        'sensitivity_PU_corrected': round(corrected_sens, 4),
        'specificity_PU_corrected': round(min(max(corrected_spec, 0), 1.0), 4),
        'precision_PU_corrected': round(min(corrected_prec, 1.0), 4),
        'pi_p_used': round(alpha, 4),
        'note': 'Ramola et al. 2019 correction; raw metrics are PU lower bounds',
    }


# ── Metrics ───────────────────────────────────────────────────────────────
def threshold_at_recall(probs, labels, target_recall=0.85):
    """Find lowest threshold that achieves recall >= target_recall."""
    for t in np.linspace(1.0, 0.0, 1001):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if recall >= target_recall:
            return t
    return 0.0


def compute_metrics(probs, labels, name, val_threshold=None):
    """Full metric suite for one model.

    Fix 4: if val_threshold is provided, use it instead of computing from
    the test set.  This eliminates selection-on-test bias.
    """
    auprc = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs)

    if val_threshold is not None:
        thresh = val_threshold
    else:
        thresh = threshold_at_recall(probs, labels, 0.85)

    preds = (probs >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # Clinical utility metrics
    n_total = len(labels)
    alert_rate = (tp + fp) / n_total if n_total > 0 else 0
    lr_pos = (sens / (1 - spec)) if spec < 1 else float('inf')
    lr_neg = ((1 - sens) / spec) if spec > 0 else float('inf')
    dor = (lr_pos / lr_neg) if lr_neg > 0 and lr_neg != float('inf') else float('inf')
    workup_reduction = 1.0 - alert_rate

    # Calibration curve — 10 decile bins
    cal_bins = []
    bin_edges = np.linspace(0, 1, 11)
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < 9 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            cal_bins.append({'bin': f'{lo:.1f}-{hi:.1f}', 'n': 0,
                             'mean_predicted': None, 'observed_rate': None})
        else:
            cal_bins.append({
                'bin': f'{lo:.1f}-{hi:.1f}',
                'n': int(mask.sum()),
                'mean_predicted': round(float(probs[mask].mean()), 4),
                'observed_rate':  round(float(labels[mask].mean()), 4),
            })

    return {
        'model': name,
        'AUPRC': round(auprc, 4),
        'AUROC': round(auroc, 4),
        'threshold_recall_85': round(thresh, 4),
        'threshold_source': 'validation' if val_threshold is not None else 'test',
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'precision': round(prec, 4),
        'F1': round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_samples': len(labels),
        'n_pos': int(labels.sum()),
        'alert_rate': round(alert_rate, 4),
        'LR_positive': round(lr_pos, 4) if lr_pos != float('inf') else 'inf',
        'LR_negative': round(lr_neg, 4) if lr_neg != float('inf') else 'inf',
        'diagnostic_odds_ratio': round(dor, 2) if dor != float('inf') else 'inf',
        'workup_reduction_vs_treat_all': round(workup_reduction, 4),
        'calibration': cal_bins,
    }


def mcnemar_test(probs_a, probs_b, labels, thresh_a, thresh_b):
    """McNemar's test comparing two models' binary predictions."""
    preds_a = (probs_a >= thresh_a).astype(int)
    preds_b = (probs_b >= thresh_b).astype(int)
    correct_a = (preds_a == labels).astype(int)
    correct_b = (preds_b == labels).astype(int)
    # b = A correct & B wrong; c = A wrong & B correct
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    # McNemar's chi-squared with continuity correction
    if b + c == 0:
        return 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return round(p_value, 6), b, c


def recalibrate_ppv(sens, spec, prev):
    """PPV at a given deployment prevalence."""
    return (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))


# ── Build structured features for test set ────────────────────────────────
def build_structured_features(test_df, adm):
    """Rebuild the same feature matrix used during structured model training."""
    with open(STRUCT_FEAT_JSON) as f:
        feat_info = json.load(f)
    feature_cols = feat_info['feature_columns']

    test_sids = set(test_df['subject_id'].unique())
    test_hadms = set(test_df['hadm_id'].unique())

    # Index admission rows
    idx_rows = adm[adm['hadm_id'].isin(test_hadms)].copy()
    idx_rows = idx_rows.sort_values('admittime').drop_duplicates('subject_id', keep='first')

    # Demographics
    idx_rows['sex_female'] = (idx_rows['gender'] == 'F').astype(int)
    idx_rows['race_cat'] = idx_rows['race'].apply(map_race)
    idx_rows['emergency_admission'] = (
        idx_rows['admission_type'].str.upper().str.contains('EMER', na=False).astype(int)
    )
    idx_rows['medicaid_selfpay'] = (
        idx_rows['insurance'].str.upper().isin(['MEDICAID', 'SELF PAY']).astype(int)
    )
    idx_rows['los_days'] = (
        (idx_rows['dischtime'] - idx_rows['admittime']).dt.total_seconds() / 86400.0
    )

    # Race dummies
    race_dummies = pd.get_dummies(idx_rows['race_cat'], prefix='race')
    idx_rows = pd.concat([idx_rows, race_dummies], axis=1)
    idx_rows = idx_rows.drop(columns=['race_cat'])

    # Prior admissions
    adm_cohort = adm[adm['subject_id'].isin(test_sids)].copy()
    index_adm = idx_rows[['subject_id', 'admittime']].rename(columns={'admittime': 'idx_admittime'})
    prior_adm = adm_cohort.merge(index_adm, on='subject_id', how='inner')
    prior_adm = prior_adm[prior_adm['admittime'] < prior_adm['idx_admittime']].copy()
    prior_hadms = set(prior_adm['hadm_id'].unique())

    prior_counts = prior_adm.groupby('subject_id').size().reset_index(name='n_prior_admissions')
    idx_rows = idx_rows.merge(prior_counts, on='subject_id', how='left')
    idx_rows['n_prior_admissions'] = idx_rows['n_prior_admissions'].fillna(0).astype(int)

    # Comorbidity flags from prior diagnoses
    prior_hadm_to_sid = dict(zip(
        prior_adm['hadm_id'].astype(str), prior_adm['subject_id'].astype(str)
    ))
    dx_flags = {feat: set() for feat in COMORBIDITY_PFXS}
    with open(DIAGNOSES_F) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            hadm_str = parts[1]
            if hadm_str not in prior_hadm_to_sid:
                continue
            sid = prior_hadm_to_sid[hadm_str]
            icd = parts[3].strip().replace('.', '').upper()
            for feat, pfxs in COMORBIDITY_PFXS.items():
                if any(icd.startswith(p) for p in pfxs):
                    dx_flags[feat].add(sid)

    for feat in COMORBIDITY_PFXS:
        idx_rows[feat] = idx_rows['subject_id'].apply(
            lambda s, f=feat: 1 if str(s) in dx_flags[f] else 0
        )

    # Medication flags from prior prescriptions
    rx_flags = {feat: set() for feat in DRUG_PATTERNS}
    with open(PRESCRIPTIONS_F) as f:
        header_line = next(f)
        cols = [c.strip() for c in header_line.split(',')]
        hadm_idx = cols.index('hadm_id')
        drug_idx = cols.index('drug')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(hadm_idx, drug_idx):
                continue
            hadm_str = parts[hadm_idx].strip()
            if hadm_str not in prior_hadm_to_sid:
                continue
            sid = prior_hadm_to_sid[hadm_str]
            drug_lower = parts[drug_idx].strip().lower()
            for feat, patterns in DRUG_PATTERNS.items():
                if any(p in drug_lower for p in patterns):
                    rx_flags[feat].add(sid)

    for feat in DRUG_PATTERNS:
        idx_rows[feat] = idx_rows['subject_id'].apply(
            lambda s, f=feat: 1 if str(s) in rx_flags[f] else 0
        )

    # Ensure all feature columns exist (add missing as 0)
    for col in feature_cols:
        if col not in idx_rows.columns:
            idx_rows[col] = 0

    # Build feature matrix aligned with test_df order
    idx_rows_keyed = idx_rows.set_index('subject_id')
    # Map test_df rows to their subject_id feature row
    X = np.zeros((len(test_df), len(feature_cols)), dtype=np.float32)
    for i, sid in enumerate(test_df['subject_id'].values):
        if sid in idx_rows_keyed.index:
            row = idx_rows_keyed.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for j, col in enumerate(feature_cols):
                val = row.get(col, 0)
                X[i, j] = float(val) if not pd.isna(val) else 0.0

    return X, feature_cols


# ── Keyword scoring (for keyword baseline) ──────────────────────────────
def score_notes_keyword(texts):
    """Score notes using the keyword baseline's phrase lexicon.

    Imports the lexicon and scoring from the keyword training script to
    ensure consistency.  Returns raw_scores array.
    """
    from scripts.common.ewang163_bench_utils import BenchmarkLogger  # noqa: already imported
    try:
        from scripts.training.ewang163_ptsd_train_keyword import (
            score_notes as _kw_score, COMPILED_LEXICON)
        raw, norm, crit = _kw_score(texts)
        return raw, norm
    except ImportError:
        pass

    # Inline fallback if import path doesn't resolve
    import re as _re
    _PHRASE_LEXICON_INLINE = [
        (r'\bptsd\b', 3.0), (r'\bpost[- ]?traumatic\s+stress\b', 3.0),
        (r'\bposttraumatic\s+stress\b', 3.0), (r'\btrauma\b', 2.0),
        (r'\btraumatic\b', 2.0), (r'\bflashback[s]?\b', 3.0),
        (r'\bnightmare[s]?\b', 2.0), (r'\bhypervigilance\b', 3.0),
        (r'\bhypervigilant\b', 3.0), (r'\bexaggerated\s+startle\b', 3.0),
        (r'\bassault(?:ed)?\b', 2.0), (r'\brape[d]?\b', 3.0),
        (r'\bdomestic\s+violence\b', 2.5), (r'\babuse[d]?\b', 1.5),
        (r'\bcombat\s+(veteran|exposure|related)', 3.0),
        (r'\bsexual\s+assault\b', 3.0), (r'\bre-?experienc(?:e|ing)\b', 3.0),
        (r'\bavoidance\b', 1.5), (r'\binsomnia\b', 1.0),
        (r'\bprazosin\b', 2.0), (r'\bemdr\b', 3.0),
    ]
    compiled = [(_re.compile(p, _re.IGNORECASE), w) for p, w in _PHRASE_LEXICON_INLINE]
    raw = np.zeros(len(texts))
    for i, text in enumerate(texts):
        for regex, w in compiled:
            raw[i] += w * len(regex.findall(text))
    norm = np.array([raw[i] / max(len(texts[i].split()), 1) for i in range(len(texts))])
    return raw, norm


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('PTSD NLP — Full Model Evaluation on Test Set')
    print('  (Fix 4: thresholds derived from validation set)')
    print('=' * 65, flush=True)

    bench = BenchmarkLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Load val + test data ─────────────────────────────────────────────
    print('\n[1/9] Loading validation and test data ...')
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    val_labels = val_df['ptsd_label'].values.astype(np.int64)
    val_texts = val_df['note_text'].tolist()
    labels = test_df['ptsd_label'].values.astype(np.int64)
    texts = test_df['note_text'].tolist()
    print(f'  Val:  {len(val_df):,} notes '
          f'(pos={int(val_labels.sum()):,}, unl={int((val_labels==0).sum()):,})')
    print(f'  Test: {len(test_df):,} notes '
          f'(pos={int(labels.sum()):,}, unl={int((labels==0).sum()):,})')

    adm = pd.read_parquet(ADM_PARQUET)

    all_results = {}
    model_probs = {}
    val_thresholds = {}

    # ── Phase A: Compute val-derived thresholds for all models ───────────
    print('\n[2/9] Computing val-derived thresholds (Fix 4) ...')

    # --- Longformer on val ---
    print('  Loading Longformer for val inference ...')
    with bench.track('evaluate', stage='lf_val_inference', device='gpu' if device.type == 'cuda' else 'cpu',
                     n_samples=len(val_df)):
        lf_tokenizer = AutoTokenizer.from_pretrained(LONGFORMER_DIR)
        lf_model = AutoModelForSequenceClassification.from_pretrained(
            LONGFORMER_DIR, num_labels=2
        )
        lf_model.to(device)
        val_probs_lf, _ = run_transformer_inference(
            lf_model, lf_tokenizer, val_texts, val_labels, device, MAX_LEN
        )
    val_thresholds['longformer'] = threshold_at_recall(val_probs_lf, val_labels, 0.85)
    print(f'    Longformer val threshold: {val_thresholds["longformer"]:.4f}')

    # --- BioClinicalBERT on val ---
    if os.path.isdir(BIOCLINBERT_DIR):
        with bench.track('evaluate', stage='bert_val_inference',
                         device='gpu' if device.type == 'cuda' else 'cpu',
                         n_samples=len(val_df)):
            bert_tokenizer = AutoTokenizer.from_pretrained(BIOCLINBERT_DIR)
            bert_model = AutoModelForSequenceClassification.from_pretrained(
                BIOCLINBERT_DIR, num_labels=2
            )
            bert_model.to(device)
            val_probs_bert, _ = run_transformer_inference(
                bert_model, bert_tokenizer, val_texts, val_labels, device, 512
            )
            del bert_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        val_thresholds['bioclinbert'] = threshold_at_recall(val_probs_bert, val_labels, 0.85)
        print(f'    BioClinicalBERT val threshold: {val_thresholds["bioclinbert"]:.4f}')

    # --- Structured on val ---
    X_struct_val, feat_cols = build_structured_features(val_df, adm)
    with open(STRUCT_LR_PKL, 'rb') as f:
        struct_lr = pickle.load(f)
    val_probs_struct = struct_lr.predict_proba(X_struct_val)[:, 1]
    val_thresholds['structured'] = threshold_at_recall(val_probs_struct, val_labels, 0.85)
    print(f'    Structured val threshold: {val_thresholds["structured"]:.4f}')

    # --- Keyword on val ---
    print('  Scoring keyword baseline on val ...')
    val_kw_raw, val_kw_norm = score_notes_keyword(val_texts)
    val_thresholds['keyword'] = threshold_at_recall(val_kw_raw, val_labels, 0.85)
    print(f'    Keyword val threshold: {val_thresholds["keyword"]:.4f}')

    all_results['val_thresholds'] = {k: round(float(v), 4) for k, v in val_thresholds.items()}
    print('\n  All val-derived thresholds stored.')

    # ── Phase B: Test-set evaluation with val thresholds ─────────────────

    # ── Model 1: Clinical Longformer ─────────────────────────────────────
    print('\n[3/9] Clinical Longformer test inference ...')
    with bench.track('evaluate', stage='lf_test_inference',
                     device='gpu' if device.type == 'cuda' else 'cpu',
                     n_samples=len(test_df)):
        probs_lf, _ = run_transformer_inference(
            lf_model, lf_tokenizer, texts, labels, device, MAX_LEN
        )
    del lf_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    metrics_lf = compute_metrics(probs_lf, labels, 'Clinical Longformer (PU)',
                                 val_threshold=val_thresholds['longformer'])
    all_results['longformer'] = metrics_lf
    model_probs['longformer'] = probs_lf
    print(f'  AUPRC={metrics_lf["AUPRC"]:.4f}  AUROC={metrics_lf["AUROC"]:.4f}')

    # ── Model 2: BioClinicalBERT ─────────────────────────────────────────
    print('\n[4/9] BioClinicalBERT ...')
    if os.path.isdir(BIOCLINBERT_DIR):
        with bench.track('evaluate', stage='bert_test_inference',
                         device='gpu' if device.type == 'cuda' else 'cpu',
                         n_samples=len(test_df)):
            bert_tokenizer = AutoTokenizer.from_pretrained(BIOCLINBERT_DIR)
            bert_model = AutoModelForSequenceClassification.from_pretrained(
                BIOCLINBERT_DIR, num_labels=2
            )
            bert_model.to(device)
            probs_bert, _ = run_transformer_inference(
                bert_model, bert_tokenizer, texts, labels, device, 512
            )
            del bert_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        metrics_bert = compute_metrics(probs_bert, labels, 'BioClinicalBERT (PU, truncated)',
                                       val_threshold=val_thresholds['bioclinbert'])
        all_results['bioclinbert'] = metrics_bert
        model_probs['bioclinbert'] = probs_bert
        print(f'  Truncated: AUPRC={metrics_bert["AUPRC"]:.4f}  AUROC={metrics_bert["AUROC"]:.4f}')

        # Fix 8: chunk-and-pool inference for fair comparison
        print('  Running chunk-and-pool inference (Fix 8) ...')
        with bench.track('evaluate', stage='bert_chunk_pool_test',
                         device='gpu' if device.type == 'cuda' else 'cpu',
                         n_samples=len(test_df),
                         notes='chunk_len=512, stride=256, pool=max'):
            bert_model_cp = AutoModelForSequenceClassification.from_pretrained(
                BIOCLINBERT_DIR, num_labels=2)
            bert_model_cp.to(device)
            probs_bert_cp, _ = run_chunk_pool_inference(
                bert_model_cp, bert_tokenizer, texts, labels, device,
                chunk_len=512, stride=256, pool='max'
            )
            del bert_model_cp
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Val threshold for chunk-pool (reuse val probs from truncated for now)
        val_thresholds['bioclinbert_chunkpool'] = val_thresholds['bioclinbert']
        metrics_bert_cp = compute_metrics(
            probs_bert_cp, labels, 'BioClinicalBERT (PU, chunk-pool)',
            val_threshold=val_thresholds['bioclinbert_chunkpool']
        )
        all_results['bioclinbert_chunkpool'] = metrics_bert_cp
        model_probs['bioclinbert_chunkpool'] = probs_bert_cp
        print(f'  Chunk-pool: AUPRC={metrics_bert_cp["AUPRC"]:.4f}  '
              f'AUROC={metrics_bert_cp["AUROC"]:.4f}')
    else:
        print('  Checkpoint not found — skipping.')
        all_results['bioclinbert'] = None
        all_results['bioclinbert_chunkpool'] = None

    # ── Model 3: Structured Features ─────────────────────────────────────
    print('\n[6/9] Structured Features + Logistic Regression ...')
    with bench.track('evaluate', stage='struct_test', device='cpu',
                     n_samples=len(test_df)):
        X_struct, feat_cols = build_structured_features(test_df, adm)
        probs_struct = struct_lr.predict_proba(X_struct)[:, 1]

    metrics_struct = compute_metrics(probs_struct, labels, 'Structured + LogReg',
                                     val_threshold=val_thresholds['structured'])
    all_results['structured'] = metrics_struct
    model_probs['structured'] = probs_struct
    print(f'  AUPRC={metrics_struct["AUPRC"]:.4f}  AUROC={metrics_struct["AUROC"]:.4f}')

    # ── Model 5: Keyword Baseline ────────────────────────────────────────
    print('\n[7/9] Keyword/Phrase-Lookup Baseline ...')
    with bench.track('evaluate', stage='keyword_test', device='cpu',
                     n_samples=len(test_df)):
        test_kw_raw, test_kw_norm = score_notes_keyword(texts)

    metrics_keyword = compute_metrics(test_kw_raw, labels, 'Keyword (DSM-5/PCL-5)',
                                      val_threshold=val_thresholds['keyword'])
    all_results['keyword'] = metrics_keyword
    model_probs['keyword'] = test_kw_raw
    print(f'  AUPRC={metrics_keyword["AUPRC"]:.4f}  AUROC={metrics_keyword["AUROC"]:.4f}')

    # ── McNemar's tests vs. Longformer ────────────────────────────────────
    print('\n[8/9] McNemar\'s tests vs. Longformer ...')
    mcnemar_results = {}
    lf_thresh = val_thresholds['longformer']

    for name, key in [('Structured + LogReg', 'structured'),
                      ('Keyword (DSM-5/PCL-5)', 'keyword')]:
        if all_results.get(key) is None:
            continue
        comp_thresh = val_thresholds.get(key, all_results[key]['threshold_recall_85'])
        p_val, b, c = mcnemar_test(
            probs_lf, model_probs[key], labels, lf_thresh, comp_thresh
        )
        mcnemar_results[key] = {'p_value': p_val, 'b_lf_correct_comp_wrong': b,
                                'c_lf_wrong_comp_correct': c}
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f'  Longformer vs {name}: p={p_val:.6f} ({sig})  '
              f'b={b}, c={c}')

    if all_results['bioclinbert'] is not None:
        comp_thresh = val_thresholds['bioclinbert']
        p_val, b, c = mcnemar_test(
            probs_lf, model_probs['bioclinbert'], labels, lf_thresh, comp_thresh
        )
        mcnemar_results['bioclinbert'] = {
            'p_value': p_val, 'b_lf_correct_comp_wrong': b,
            'c_lf_wrong_comp_correct': c
        }
        print(f'  Longformer vs BioClinicalBERT: p={p_val:.6f}  b={b}, c={c}')

    all_results['mcnemar_vs_longformer'] = mcnemar_results

    # ── Prevalence recalibration + clinical utility (Longformer) ─────────
    print('\n  Prevalence recalibration + clinical utility (Longformer):')
    sens_lf = metrics_lf['sensitivity']
    spec_lf = metrics_lf['specificity']
    prevalences = [0.01, 0.02, 0.05, 0.10, 0.20]
    recal_results = []
    print(f'    {"Prev":>6} {"PPV":>8} {"NPV":>8} {"NNS":>8} {"NNE":>8} {"LR+":>8} {"LR-":>8}')
    for prev in prevalences:
        ppv = recalibrate_ppv(sens_lf, spec_lf, prev)
        npv_num = spec_lf * (1 - prev)
        npv_den = spec_lf * (1 - prev) + (1 - sens_lf) * prev
        npv = npv_num / npv_den if npv_den > 0 else 0
        nns = 1.0 / ppv if ppv > 0 else float('inf')
        nne = 1.0 / (sens_lf - (1 - spec_lf)) if sens_lf > (1 - spec_lf) else float('inf')
        lr_pos = sens_lf / (1 - spec_lf) if spec_lf < 1 else float('inf')
        lr_neg = (1 - sens_lf) / spec_lf if spec_lf > 0 else float('inf')
        recal_results.append({
            'prevalence': prev,
            'PPV': round(ppv, 4),
            'NPV': round(npv, 4),
            'NNS': round(nns, 1) if nns != float('inf') else 'inf',
            'NNE': round(nne, 1) if nne != float('inf') else 'inf',
            'LR_positive': round(lr_pos, 2) if lr_pos != float('inf') else 'inf',
            'LR_negative': round(lr_neg, 4) if lr_neg != float('inf') else 'inf',
        })
        print(f'    {prev:>5.0%} {ppv:>8.4f} {npv:>8.4f} '
              f'{nns:>8.1f} {nne:>8.1f} {lr_pos:>8.2f} {lr_neg:>8.4f}')
    all_results['prevalence_recalibration'] = recal_results

    # ── Clinical utility summary (study-cohort level) ────────────────────
    clinical_utility = {
        'threshold_source': 'validation (Fix 4)',
        'operating_threshold': round(float(val_thresholds['longformer']), 4),
        'alert_rate_study_cohort': metrics_lf['alert_rate'],
        'LR_positive': metrics_lf['LR_positive'],
        'LR_negative': metrics_lf['LR_negative'],
        'diagnostic_odds_ratio': metrics_lf['diagnostic_odds_ratio'],
        'workup_reduction_vs_treat_all': metrics_lf['workup_reduction_vs_treat_all'],
        'sensitivity_at_threshold': metrics_lf['sensitivity'],
        'specificity_at_threshold': metrics_lf['specificity'],
    }
    all_results['clinical_utility'] = clinical_utility
    print(f'\n  Clinical utility (study cohort):')
    print(f'    Alert rate:        {clinical_utility["alert_rate_study_cohort"]:.1%}')
    print(f'    LR+:               {clinical_utility["LR_positive"]}')
    print(f'    LR-:               {clinical_utility["LR_negative"]}')
    print(f'    DOR:               {clinical_utility["diagnostic_odds_ratio"]}')
    print(f'    Workup reduction:  {clinical_utility["workup_reduction_vs_treat_all"]:.1%} '
          f'(vs. treat-all)')

    # ── PU-corrected metrics (Fix 6: Ramola et al. 2019) ────────────────
    print('\n  PU-corrected metrics (Fix 6, Ramola et al. 2019):')
    pi_p_estimate = labels.sum() / len(labels)
    pu_corrections = {}
    for key in ['longformer', 'bioclinbert', 'bioclinbert_chunkpool',
                'structured', 'keyword']:
        m = all_results.get(key)
        if m is None:
            continue
        corrected = ramola_corrected_metrics(
            m['AUPRC'], m['AUROC'], m['sensitivity'], m['specificity'],
            m['precision'], pi_p_estimate
        )
        pu_corrections[key] = corrected
        m['PU_corrected'] = corrected
        m['metrics_note'] = 'Raw metrics are PU lower bounds (Fix 6)'

    all_results['pu_correction_info'] = {
        'pi_p_estimate_used': round(float(pi_p_estimate), 4),
        'pi_p_source': 'empirical labeled fraction in test set',
        'reference': 'Ramola et al. 2019, Pac Symp Biocomput, PMID 30864316',
    }

    print(f'    pi_p estimate: {pi_p_estimate:.4f}')
    if 'longformer' in pu_corrections:
        c = pu_corrections['longformer']
        print(f'    Longformer corrected AUPRC: {c.get("AUPRC_PU_corrected", "N/A")}')
        print(f'    Longformer corrected AUROC: {c.get("AUROC_PU_corrected", "N/A")}')

    # ── Subgroup analysis (Longformer AUPRC) ──────────────────────────────
    print('\n[9/9] Subgroup analysis (Longformer) ...')

    test_hadms = set(test_df['hadm_id'].tolist())
    test_adm = adm[adm['hadm_id'].isin(test_hadms)].copy()
    test_adm = test_adm.sort_values('admittime').drop_duplicates('subject_id', keep='first')
    test_adm['race_cat'] = test_adm['race'].apply(map_race)
    test_adm['age_group'] = test_adm['age_at_admission'].apply(age_decade)
    test_adm['is_emergency'] = (
        test_adm['admission_type'].str.upper().str.contains('EMER', na=False)
    )

    demo_map = test_adm.set_index('subject_id')[
        ['gender', 'race_cat', 'age_group', 'is_emergency']
    ].to_dict('index')

    test_df['gender'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('gender', 'Unknown'))
    test_df['race_cat'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('race_cat', 'Other/Unknown'))
    test_df['age_group'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('age_group', 'Other'))
    test_df['is_emergency'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('is_emergency', False))

    subgroup_results = {}

    def subgroup_auprc(col, label):
        results = {}
        for val in sorted(test_df[col].unique()):
            mask = (test_df[col] == val).values
            n = int(mask.sum())
            n_pos_sub = int(labels[mask].sum())
            if n_pos_sub == 0 or n_pos_sub == n:
                results[str(val)] = {'n': n, 'n_pos': n_pos_sub, 'AUPRC': None}
                continue
            auprc = average_precision_score(labels[mask], probs_lf[mask])
            results[str(val)] = {'n': n, 'n_pos': n_pos_sub, 'AUPRC': round(auprc, 4)}
        return results

    print('  Sex:')
    sex_res = subgroup_auprc('gender', 'Sex')
    subgroup_results['sex'] = sex_res
    for k, v in sex_res.items():
        print(f'    {k}: n={v["n"]}, pos={v["n_pos"]}, AUPRC={v["AUPRC"]}')

    print('  Age group:')
    age_res = subgroup_auprc('age_group', 'Age')
    subgroup_results['age_group'] = age_res
    for k, v in age_res.items():
        print(f'    {k}: n={v["n"]}, pos={v["n_pos"]}, AUPRC={v["AUPRC"]}')

    print('  Race/ethnicity:')
    race_res = subgroup_auprc('race_cat', 'Race')
    subgroup_results['race'] = race_res
    for k, v in race_res.items():
        print(f'    {k}: n={v["n"]}, pos={v["n_pos"]}, AUPRC={v["AUPRC"]}')

    print('  Emergency admission:')
    emer_res = subgroup_auprc('is_emergency', 'Emergency')
    subgroup_results['emergency'] = emer_res
    for k, v in emer_res.items():
        label = 'Emergency' if k == 'True' else 'Non-emergency'
        print(f'    {label}: n={v["n"]}, pos={v["n_pos"]}, AUPRC={v["AUPRC"]}')

    all_results['subgroup_analysis'] = subgroup_results

    # ── Save results ──────────────────────────────────────────────────────
    with open(RESULTS_JSON, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nFull results → {RESULTS_JSON}')

    summary_rows = []
    for key in ['longformer', 'bioclinbert', 'bioclinbert_chunkpool',
                'structured', 'keyword']:
        m = all_results.get(key)
        if m is None:
            continue
        row = {
            'model': m['model'],
            'AUPRC': m['AUPRC'],
            'AUROC': m['AUROC'],
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
            'precision': m['precision'],
            'F1': m['F1'],
            'threshold': m['threshold_recall_85'],
            'threshold_source': m.get('threshold_source', 'validation'),
            'alert_rate': m.get('alert_rate', ''),
            'LR_positive': m.get('LR_positive', ''),
            'LR_negative': m.get('LR_negative', ''),
            'DOR': m.get('diagnostic_odds_ratio', ''),
            'workup_reduction': m.get('workup_reduction_vs_treat_all', ''),
        }
        if key in mcnemar_results:
            row['mcnemar_p_vs_longformer'] = mcnemar_results[key]['p_value']
        else:
            row['mcnemar_p_vs_longformer'] = None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f'Summary CSV  → {SUMMARY_CSV}')

    # ── Speed comparison table ────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('SPEED vs. ACCURACY COMPARISON')
    print('=' * 65)
    print(f'\n  {"Model":<30} {"AUPRC":>7} {"Training":>12} {"Inference":>10}')
    print(f'  {"-"*30} {"-"*7} {"-"*12} {"-"*10}')
    print(f'  {"Keyword (DSM-5/PCL-5)":<30} {metrics_keyword["AUPRC"]:>7.4f} {"0s":>12} {"~seconds":>10}')
    print(f'  {"Structured + LogReg":<30} {metrics_struct["AUPRC"]:>7.4f} {"~minutes":>12} {"~seconds":>10}')
    if all_results['bioclinbert'] is not None:
        print(f'  {"BioClinicalBERT (512)":<30} {all_results["bioclinbert"]["AUPRC"]:>7.4f} {"~12 min":>12} {"~minutes":>10}')
    print(f'  {"Clinical Longformer (4096)":<30} {metrics_lf["AUPRC"]:>7.4f} {"~7.5 hr":>12} {"~minutes":>10}')

    # ── Final summary ─────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('TEST SET EVALUATION SUMMARY (thresholds from validation)')
    print('=' * 65)
    print(f'\n  {"Model":<30} {"AUPRC":>7} {"AUROC":>7} '
          f'{"Sens":>6} {"Spec":>6} {"Prec":>6} {"F1":>6} {"McNemar p":>10}')
    print(f'  {"-"*30} {"-"*7} {"-"*7} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*10}')
    for row in summary_rows:
        mc_p = row['mcnemar_p_vs_longformer']
        mc_str = f'{mc_p:.6f}' if mc_p is not None else '—'
        print(f'  {row["model"]:<30} {row["AUPRC"]:>7.4f} {row["AUROC"]:>7.4f} '
              f'{row["sensitivity"]:>6.4f} {row["specificity"]:>6.4f} '
              f'{row["precision"]:>6.4f} {row["F1"]:>6.4f} {mc_str:>10}')

    print(f'\n  Prevalence recalibration (Longformer, sens={sens_lf:.4f}, spec={spec_lf:.4f}):')
    for r in recal_results:
        nns_str = f'{r["NNS"]}' if isinstance(r['NNS'], str) else f'{r["NNS"]:.1f}'
        print(f'    prev={r["prevalence"]:.0%}: PPV={r["PPV"]:.4f}, NPV={r["NPV"]:.4f}, NNS={nns_str}')

    print('\nDone.')


if __name__ == '__main__':
    main()
