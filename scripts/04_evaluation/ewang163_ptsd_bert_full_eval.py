"""
ewang163_ptsd_bert_full_eval.py
================================
Full evaluation suite for BioClinicalBERT — both inference modes (truncated 512
and chunk-and-pool 512x256) — to mirror what was already done for the primary
PULSNAR Clinical Longformer.

For each mode (truncated / chunk-pool) the script computes and writes:

  Predictions:
    results/predictions/ewang163_bioclinbert_{trunc,chunkpool}_val_predictions.csv
    results/predictions/ewang163_bioclinbert_{trunc,chunkpool}_test_predictions.csv

  Calibration (raw + Platt + Elkan-Noto, ECE, plot):
    results/metrics/ewang163_calibration_results_bert_{trunc,chunkpool}.csv
    results/figures/ewang163_calibration_curve_bert_{trunc,chunkpool}.png

  Decision curve analysis at 2% / 5% deployment prevalence:
    results/metrics/ewang163_dca_results_bert_{trunc,chunkpool}.csv
    results/figures/ewang163_dca_2pct_bert_{trunc,chunkpool}.png
    results/figures/ewang163_dca_5pct_bert_{trunc,chunkpool}.png

  Fairness (calibration-in-large, equal-opportunity diff, bootstrap CI AUPRC):
    results/metrics/ewang163_fairness_results_bert_{trunc,chunkpool}.csv

  Subgroup AUPRC + clinical utility at deployment prevalences:
    results/metrics/ewang163_subgroup_auprc_bert_{trunc,chunkpool}.csv

  Proxy validation (Mann-Whitney AUC vs random unlabeled):
    results/metrics/ewang163_proxy_validation_bert_{trunc,chunkpool}.csv

  Error analysis (FP/FN demographics + distinctive trauma terms):
    results/error_analysis/ewang163_bert_{trunc,chunkpool}_error_summary.csv

  Ablations (Ablation 1: PTSD masking; Ablation 2: PMH section removal):
    results/metrics/ewang163_ablation_results_bert_{trunc,chunkpool}.csv

The Ablation 2 PMH-removed test corpus is built once and cached at
data/notes/ewang163_test_no_pmh.parquet so repeated runs reuse it.

Submit via SLURM:
    sbatch scripts/04_evaluation/ewang163_ptsd_bert_full_eval.sh
"""

import csv
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC           = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_NOTES      = f'{STUDENT_DIR}/data/notes'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
DATA_COHORT     = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR       = f'{STUDENT_DIR}/models'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'
RESULTS_PRED    = f'{STUDENT_DIR}/results/predictions'
RESULTS_FIG     = f'{STUDENT_DIR}/results/figures'
RESULTS_ERR     = f'{STUDENT_DIR}/results/error_analysis'

VAL_PARQUET     = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET    = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET     = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
CORPUS_PARQUET  = f'{DATA_NOTES}/ewang163_ptsd_corpus.parquet'
PROXY_PARQUET   = f'{DATA_NOTES}/ewang163_proxy_notes.parquet'
SPLITS_JSON     = f'{DATA_SPLITS}/ewang163_split_subject_ids.json'
DISCHARGE_F     = f'{MIMIC}/note/2.2/discharge.csv'

BERT_DIR        = f'{MODEL_DIR}/ewang163_bioclinbert_best'
NO_PMH_PARQUET  = f'{DATA_NOTES}/ewang163_test_no_pmh.parquet'

CHUNK_LEN       = 512
TRUNC_LEN       = 512
STRIDE          = 256
BATCH_SIZE      = 16
N_BINS          = 10
N_BOOTSTRAP     = 1000
RANDOM_SEED     = 42
N_UNLAB_SAMPLE  = 500

PREVALENCES     = (0.01, 0.02, 0.05, 0.10, 0.20)
DEPLOY_PREVS    = {'2pct': 0.02, '5pct': 0.05}

# Section parsing for Ablation 2
SECTION_HEADER_RE = re.compile(r'^([A-Z][A-Za-z /&\-]+):[ ]*$', re.MULTILINE)
INCLUDE_SECTIONS_NO_PMH = {
    'history of present illness', 'social history', 'brief hospital course',
}
SECTION_ORDER_NO_PMH = [
    'history of present illness', 'social history', 'brief hospital course',
]

# Ablation 1 — PTSD-string masking
MASK_PATTERNS = [
    r'post-traumatic', r'post\s+traumatic', r'posttraumatic',
    r'trauma-related\s+stress', r'ptsd', r'f43\.1', r'309\.81',
]
MASK_RE = re.compile('|'.join(MASK_PATTERNS), re.IGNORECASE)
MASK_TOKEN = '[PTSD_MASKED]'

# Trauma vocabulary used in error-analysis "distinctive terms"
TRAUMA_TERMS = [
    'ptsd', 'trauma', 'flashback', 'nightmare', 'hypervigil', 'assault',
    'abuse', 'rape', 'mva', 'combat', 'veteran', 'military', 'gunshot',
    'stab', 'mst', 'mvc',
]


# ── Inference helpers ─────────────────────────────────────────────────────
class TruncDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


@torch.no_grad()
def run_truncated_inference(model, tokenizer, texts, device):
    ds = TruncDataset(texts, tokenizer, TRUNC_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    out = []
    for batch in loader:
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(input_ids=ids, attention_mask=am).logits
        probs = F.softmax(logits.float(), dim=-1)[:, 1]
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def run_chunkpool_inference(model, tokenizer, texts, device,
                            chunk_len=CHUNK_LEN, stride=STRIDE):
    """Overlapping windowed inference, max-pool positive-class probs."""
    model.eval()
    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102
    usable = chunk_len - 2

    out = np.zeros(len(texts))
    for i, text in enumerate(texts):
        toks = tokenizer(text, return_tensors='pt', truncation=False,
                         add_special_tokens=False)
        ids_full = toks['input_ids'].squeeze(0)
        n_tok = len(ids_full)

        if n_tok <= usable:
            enc = tokenizer(text, max_length=chunk_len, padding='max_length',
                            truncation=True, return_tensors='pt')
            ids = enc['input_ids'].to(device)
            am = enc['attention_mask'].to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(input_ids=ids, attention_mask=am).logits
            out[i] = float(F.softmax(logits.float(), dim=-1)[0, 1])
            continue

        chunk_probs = []
        start = 0
        while start < n_tok:
            end = min(start + usable, n_tok)
            chunk = torch.cat([
                torch.tensor([cls_id]),
                ids_full[start:end],
                torch.tensor([sep_id]),
            ])
            am = torch.ones(len(chunk), dtype=torch.long)
            pad = chunk_len - len(chunk)
            if pad > 0:
                chunk = torch.cat([chunk, torch.zeros(pad, dtype=torch.long)])
                am = torch.cat([am, torch.zeros(pad, dtype=torch.long)])
            chunk = chunk.unsqueeze(0).to(device)
            am = am.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(input_ids=chunk, attention_mask=am).logits
            chunk_probs.append(float(F.softmax(logits.float(), dim=-1)[0, 1]))
            if end >= n_tok:
                break
            start += stride
        out[i] = max(chunk_probs)
        if (i + 1) % 200 == 0:
            print(f'    chunk-pool {i + 1}/{len(texts)}', flush=True)
    return out


def threshold_at_recall(probs, labels, target=0.85):
    for t in np.linspace(1.0, 0.0, 1001):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if recall >= target:
            return float(t)
    return 0.0


def compute_metrics(probs, labels, thresh):
    preds = (probs >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return {
        'AUPRC': round(average_precision_score(labels, probs), 4),
        'AUROC': round(roc_auc_score(labels, probs), 4),
        'threshold': round(thresh, 4),
        'sensitivity': round(sens, 4), 'specificity': round(spec, 4),
        'precision': round(prec, 4), 'F1': round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_total': len(labels), 'n_pos': int(labels.sum()),
    }


def clinical_utility(sens, spec, alert_rate, prevs=PREVALENCES):
    eps = 1e-9
    lr_pos = sens / (1 - spec + eps)
    lr_neg = (1 - sens) / (spec + eps)
    dor = (sens / (1 - sens + eps)) / ((1 - spec + eps) / (spec + eps))
    rows = []
    for p in prevs:
        ppv = (sens * p) / (sens * p + (1 - spec) * (1 - p) + eps)
        npv = (spec * (1 - p)) / ((1 - sens) * p + spec * (1 - p) + eps)
        nns = 1 / ppv if ppv > 0 else float('inf')
        rows.append({'prevalence': p, 'PPV': round(ppv, 4),
                     'NPV': round(npv, 4), 'NNS': round(nns, 2)})
    return {
        'LR_positive': round(lr_pos, 4),
        'LR_negative': round(lr_neg, 4),
        'DOR': round(dor, 2),
        'alert_rate': round(alert_rate, 4),
        'workup_reduction': round(1 - alert_rate, 4),
        'by_prevalence': rows,
    }


# ── Calibration ───────────────────────────────────────────────────────────
def equal_freq_bins(probs, labels, n_bins):
    order = np.argsort(probs)
    ps = probs[order]
    ls = labels[order]
    sz = len(probs) // n_bins
    bins = []
    for i in range(n_bins):
        s = i * sz
        e = (i + 1) * sz if i < n_bins - 1 else len(probs)
        n = e - s
        mp = float(ps[s:e].mean())
        of = float(ls[s:e].mean())
        z = 1.96
        denom = 1 + z**2 / n
        centre = (of + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((of * (1 - of) + z**2 / (4 * n)) / n) / denom
        bins.append({
            'bin': i + 1, 'n': n,
            'mean_predicted': round(mp, 4),
            'observed_fraction': round(of, 4),
            'ci_lower': round(max(0, centre - margin), 4),
            'ci_upper': round(min(1, centre + margin), 4),
        })
    return bins


def compute_ece(bins):
    total = sum(b['n'] for b in bins)
    return sum((b['n'] / total) * abs(b['mean_predicted'] - b['observed_fraction'])
               for b in bins)


# ── Fairness ──────────────────────────────────────────────────────────────
def map_race(s):
    if pd.isna(s): return 'Other/Unknown'
    r = str(s).upper()
    if 'WHITE' in r: return 'White'
    if 'BLACK' in r or 'AFRICAN' in r: return 'Black'
    if 'HISPANIC' in r or 'LATINO' in r: return 'Hispanic'
    if 'ASIAN' in r: return 'Asian'
    return 'Other/Unknown'


def age_decade(a):
    if pd.isna(a): return 'Other'
    a = int(a)
    if 20 <= a <= 29: return '20s'
    if 30 <= a <= 39: return '30s'
    if 40 <= a <= 49: return '40s'
    if 50 <= a <= 59: return '50s'
    return 'Other'


def bootstrap_auprc(labels, probs, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    aps = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y = labels[idx]; p = probs[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aps.append(average_precision_score(y, p))
    if not aps:
        return None, None, None
    aps = sorted(aps)
    pt = average_precision_score(labels, probs)
    return round(pt, 4), round(aps[int(0.025 * len(aps))], 4), round(aps[int(0.975 * len(aps))], 4)


def wilson_ci(p, n, z=1.96):
    if n == 0: return (0, 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (round(max(centre - spread, 0), 4), round(min(centre + spread, 1), 4))


def attach_demographics(df, adm):
    """Join admissions to test_df to attach race / age_group / emergency."""
    hadms = set(df['hadm_id'].tolist())
    adm = adm[adm['hadm_id'].isin(hadms)].copy()
    adm = adm.sort_values('admittime').drop_duplicates('subject_id', keep='first')
    adm['race_cat'] = adm['race'].apply(map_race)
    adm['age_group'] = adm['age_at_admission'].apply(age_decade)
    adm['is_emergency'] = adm['admission_type'].str.upper().str.contains('EMER', na=False)
    demo_map = adm.set_index('subject_id')[
        ['gender', 'race_cat', 'age_group', 'is_emergency', 'age_at_admission']
    ].to_dict('index')
    df = df.copy()
    df['gender'] = df['subject_id'].map(lambda s: demo_map.get(s, {}).get('gender', 'Unknown'))
    df['race_cat'] = df['subject_id'].map(lambda s: demo_map.get(s, {}).get('race_cat', 'Other/Unknown'))
    df['age_group'] = df['subject_id'].map(lambda s: demo_map.get(s, {}).get('age_group', 'Other'))
    df['is_emergency'] = df['subject_id'].map(lambda s: demo_map.get(s, {}).get('is_emergency', False))
    df['race_binary'] = df['race_cat'].apply(lambda r: 'White' if r == 'White' else 'Non-White')
    return df


def fairness_table(test_df, probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    rows = []
    groupings = [
        ('sex', 'gender'), ('age_group', 'age_group'),
        ('race', 'race_cat'), ('race_binary', 'race_binary'),
        ('emergency', 'is_emergency'),
    ]
    for gname, col in groupings:
        for val in sorted(test_df[col].unique()):
            mask = (test_df[col] == val).values
            n = int(mask.sum())
            n_pos = int(labels[mask].sum())
            if n_pos == 0 or n_pos == n:
                rows.append({
                    'group': gname, 'value': str(val), 'n': n, 'n_pos': n_pos,
                    'calibration_in_large': None, 'cal_ci_lo': None, 'cal_ci_hi': None,
                    'recall_at_threshold': None,
                    'AUPRC': None, 'AUPRC_ci_lo': None, 'AUPRC_ci_hi': None,
                    'AUPRC_ci_width': None, 'AUPRC_reliable': False,
                })
                continue
            mp = float(probs[mask].mean())
            mo = float(labels[mask].mean())
            cal = round(mp - mo, 4)
            cal_ci = wilson_ci(mo, n)
            tp = ((preds[mask] == 1) & (labels[mask] == 1)).sum()
            fn = ((preds[mask] == 0) & (labels[mask] == 1)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            ap, lo, hi = bootstrap_auprc(labels[mask], probs[mask])
            ci_w = round(hi - lo, 4) if lo is not None else None
            rows.append({
                'group': gname, 'value': str(val), 'n': n, 'n_pos': n_pos,
                'calibration_in_large': cal,
                'cal_ci_lo': cal_ci[0], 'cal_ci_hi': cal_ci[1],
                'recall_at_threshold': round(float(recall), 4),
                'AUPRC': ap, 'AUPRC_ci_lo': lo, 'AUPRC_ci_hi': hi,
                'AUPRC_ci_width': ci_w,
                'AUPRC_reliable': ci_w is not None and ci_w < 0.15,
            })
    return pd.DataFrame(rows)


def subgroup_table(test_df, probs, labels):
    """Per-subgroup AUPRC + AUROC + per-prevalence NNS."""
    rows = []
    groupings = [
        ('sex', 'gender'), ('age_group', 'age_group'),
        ('race', 'race_cat'), ('race_binary', 'race_binary'),
        ('emergency', 'is_emergency'),
    ]
    for gname, col in groupings:
        for val in sorted(test_df[col].unique()):
            mask = (test_df[col] == val).values
            n = int(mask.sum())
            n_pos = int(labels[mask].sum())
            if n_pos == 0 or n_pos == n:
                rows.append({'group': gname, 'value': str(val),
                             'n': n, 'n_pos': n_pos,
                             'AUPRC': None, 'AUROC': None})
                continue
            ap = average_precision_score(labels[mask], probs[mask])
            ar = roc_auc_score(labels[mask], probs[mask])
            rows.append({'group': gname, 'value': str(val),
                         'n': n, 'n_pos': n_pos,
                         'AUPRC': round(ap, 4), 'AUROC': round(ar, 4)})
    return pd.DataFrame(rows)


# ── Decision curves ───────────────────────────────────────────────────────
def decision_curve(test_probs_cal, test_labels, p_study, p_deploy):
    """Bayes-shifts calibrated probs to deployment prevalence and returns
    NB(model) and NB(treat-all) over the threshold sweep."""
    cal_deploy = (test_probs_cal * p_deploy / p_study) / (
        test_probs_cal * p_deploy / p_study +
        (1 - test_probs_cal) * (1 - p_deploy) / (1 - p_study)
    )
    thresholds = np.round(np.arange(0.01, 0.405, 0.005), 4)
    rows = []
    N = len(test_labels)
    prev_test = test_labels.mean()
    for t in thresholds:
        preds = (cal_deploy >= t).astype(int)
        tp = ((preds == 1) & (test_labels == 1)).sum()
        fp = ((preds == 1) & (test_labels == 0)).sum()
        nb_model = tp / N - fp / N * (t / (1 - t))
        nb_all = prev_test - (1 - prev_test) * (t / (1 - t))
        rows.append({'threshold': t,
                     'net_benefit_model': round(nb_model, 6),
                     'net_benefit_treatall': round(nb_all, 6)})
    return pd.DataFrame(rows)


def plot_dca(dca_df, p_deploy, model_label, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    ts = dca_df['threshold'].values
    nb_m = dca_df['net_benefit_model'].values
    nb_a = dca_df['net_benefit_treatall'].values
    ax.plot(ts, nb_m, 'b-', linewidth=2, label=model_label)
    ax.plot(ts, nb_a, 'r--', linewidth=1.5, label='Treat all')
    ax.axhline(0, color='k', linestyle=':', linewidth=1, label='Treat none')
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    pct = int(p_deploy * 100)
    ax.set_title(f'Decision Curve Analysis — {pct}% deployment prevalence')
    ax.legend(loc='upper right')
    ax.set_xlim(0.01, 0.40)
    y_min = min(min(nb_m), min(nb_a), 0) - 0.02
    y_max = max(max(nb_m), max(nb_a), 0) * 1.15 + 0.02
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ── Ablation 2 helpers ────────────────────────────────────────────────────
def parse_sections(text, include_set):
    if not text:
        return {}
    headers = []
    for m in SECTION_HEADER_RE.finditer(text):
        headers.append((m.start(), m.end(), m.group(1).strip().lower()))
    res = {}
    for i, (s, e, name) in enumerate(headers):
        if name not in include_set:
            continue
        body_s = e
        body_e = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[body_s:body_e].strip()
        if body and name not in res:
            res[name] = body
    return res


def build_no_pmh_test_corpus(test_df):
    """Stream discharge.csv and rewrite test notes without the PMH section."""
    test_hadms = set(test_df['hadm_id'].tolist())
    hadm_to_idx = {h: i for i, h in enumerate(test_df['hadm_id'].tolist())}
    no_pmh = test_df['note_text'].tolist()  # fallback
    matched = 0
    with open(DISCHARGE_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                h = int(row['hadm_id'])
            except (KeyError, ValueError):
                continue
            if h not in test_hadms:
                continue
            sects = parse_sections(row['text'], INCLUDE_SECTIONS_NO_PMH)
            joined = '\n\n'.join(sects[n] for n in SECTION_ORDER_NO_PMH if n in sects)
            if joined.strip():
                no_pmh[hadm_to_idx[h]] = joined
            matched += 1
            if matched == len(test_hadms):
                break
    out = test_df[['subject_id', 'hadm_id', 'ptsd_label']].copy()
    out['note_text'] = no_pmh
    out.to_parquet(NO_PMH_PARQUET, index=False)
    print(f'  No-PMH test corpus written → {NO_PMH_PARQUET}')
    return out


def apply_masking(text):
    return MASK_RE.sub(MASK_TOKEN, text)


# ── Per-mode pipeline ─────────────────────────────────────────────────────
def evaluate_one_mode(model, tokenizer, mode, val_df, test_df, val_probs,
                      test_probs, val_labels, test_labels, adm, proxy_df,
                      unlab_sample_df, no_pmh_test_df, device, bench):
    """All downstream analyses for one inference mode (truncated or chunkpool).
    Returns a dict summarising the headline metrics."""
    suffix = f'bert_{mode}'

    print(f'\n  ▶ Mode = {mode}: predictions cached, running analyses')

    # Step A: write predictions
    pd.DataFrame({
        'subject_id': val_df['subject_id'].values,
        'hadm_id': val_df['hadm_id'].values,
        'ptsd_label': val_labels,
        'predicted_prob': val_probs,
    }).to_csv(f'{RESULTS_PRED}/ewang163_bioclinbert_{mode}_val_predictions.csv',
              index=False)
    pd.DataFrame({
        'subject_id': test_df['subject_id'].values,
        'hadm_id': test_df['hadm_id'].values,
        'ptsd_label': test_labels,
        'predicted_prob': test_probs,
    }).to_csv(f'{RESULTS_PRED}/ewang163_bioclinbert_{mode}_test_predictions.csv',
              index=False)

    # Step B: threshold + base metrics + utility
    val_thresh = threshold_at_recall(val_probs, val_labels, 0.85)
    val_m = compute_metrics(val_probs, val_labels, val_thresh)
    test_m = compute_metrics(test_probs, test_labels, val_thresh)
    alert_rate = float((test_probs >= val_thresh).mean())
    util = clinical_utility(test_m['sensitivity'], test_m['specificity'],
                            alert_rate)
    print(f'    Val threshold={val_thresh:.4f}  '
          f'Test AUPRC={test_m["AUPRC"]} AUROC={test_m["AUROC"]} '
          f'F1={test_m["F1"]}')

    # Step C: calibration
    platt = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
    platt.fit(val_probs.reshape(-1, 1), val_labels)
    test_cal = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    val_pos = val_labels == 1
    c_est = float(val_probs[val_pos].mean()) if val_pos.any() else 1.0
    test_en = np.clip(test_cal / max(c_est, 1e-6), 0, 1)

    bins_raw = equal_freq_bins(test_probs, test_labels, N_BINS)
    bins_cal = equal_freq_bins(test_cal, test_labels, N_BINS)
    bins_en  = equal_freq_bins(test_en,  test_labels, N_BINS)
    ece_raw = compute_ece(bins_raw)
    ece_cal = compute_ece(bins_cal)
    ece_en  = compute_ece(bins_en)
    print(f'    ECE: raw={ece_raw:.4f} platt={ece_cal:.4f} en={ece_en:.4f} '
          f'(c={c_est:.4f})')

    cal_rows = []
    for btype, bs in [('raw', bins_raw), ('platt_scaled', bins_cal),
                      ('elkan_noto', bins_en)]:
        for b in bs:
            cal_rows.append({'type': btype, **b})
    for label, ev in [('raw', ece_raw), ('platt_scaled', ece_cal),
                      ('elkan_noto', ece_en)]:
        cal_rows.append({'type': 'ece_summary', 'bin': label,
                         'n': len(test_labels),
                         'mean_predicted': round(ev, 4),
                         'observed_fraction': None,
                         'ci_lower': None, 'ci_upper': None})
    cal_rows.append({'type': 'elkan_noto_c', 'bin': 'c_estimate',
                     'n': int(val_pos.sum()),
                     'mean_predicted': round(c_est, 4),
                     'observed_fraction': None,
                     'ci_lower': None, 'ci_upper': None})
    pd.DataFrame(cal_rows).to_csv(
        f'{RESULTS_METRICS}/ewang163_calibration_results_{suffix}.csv',
        index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, bins, ttl, ev in [
        (axes[0], bins_raw, f'Raw BERT ({mode})', ece_raw),
        (axes[1], bins_cal, 'Platt-scaled', ece_cal),
        (axes[2], bins_en,  f'Elkan-Noto (c={c_est:.3f})', ece_en),
    ]:
        mp = [b['mean_predicted'] for b in bins]
        of = [b['observed_fraction'] for b in bins]
        lo = [o - b['ci_lower'] for o, b in zip(of, bins)]
        hi = [b['ci_upper'] - o for o, b in zip(of, bins)]
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
        ax.errorbar(mp, of, yerr=[lo, hi], fmt='o-', color='steelblue',
                    capsize=4, linewidth=1.5, markersize=6, label='Observed')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction positive')
        ax.set_title(f'{ttl}\nECE = {ev:.4f}')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{RESULTS_FIG}/ewang163_calibration_curve_{suffix}.png', dpi=150)
    plt.close(fig)

    # Step D: decision curves
    p_study = float(test_labels.mean())
    dca_2 = decision_curve(test_cal, test_labels, p_study, 0.02)
    dca_5 = decision_curve(test_cal, test_labels, p_study, 0.05)
    dca_combined = dca_2.merge(dca_5, on='threshold',
                                suffixes=('_2pct', '_5pct'))
    dca_combined.to_csv(f'{RESULTS_METRICS}/ewang163_dca_results_{suffix}.csv',
                        index=False)
    plot_dca(dca_2, 0.02, f'BERT ({mode})',
             f'{RESULTS_FIG}/ewang163_dca_2pct_{suffix}.png')
    plot_dca(dca_5, 0.05, f'BERT ({mode})',
             f'{RESULTS_FIG}/ewang163_dca_5pct_{suffix}.png')
    nb2_max = float(dca_2['net_benefit_model'].max())
    nb5_max = float(dca_5['net_benefit_model'].max())

    # Step E: fairness + subgroup
    test_with_demo = attach_demographics(test_df, adm)
    fair_df = fairness_table(test_with_demo, test_probs, test_labels, val_thresh)
    fair_df.to_csv(f'{RESULTS_METRICS}/ewang163_fairness_results_{suffix}.csv',
                   index=False)
    sub_df = subgroup_table(test_with_demo, test_probs, test_labels)
    sub_df.to_csv(f'{RESULTS_METRICS}/ewang163_subgroup_auprc_{suffix}.csv',
                  index=False)

    # Step F: proxy validation (use already-cached proxy probs from caller)
    # done outside this function

    # Step G: error analysis (FP/FN demographics + trauma terms)
    err_summary = error_analysis(test_with_demo, test_probs, test_labels,
                                 val_thresh, mode)
    err_summary.to_csv(f'{RESULTS_ERR}/ewang163_bert_{mode}_error_summary.csv',
                       index=False)

    # Step H: ablations
    ablations_rows = []
    test_texts = test_df['note_text'].tolist()
    # Baseline = current test_probs
    abl_baseline = compute_metrics(test_probs, test_labels, val_thresh)
    abl_baseline['condition'] = 'Baseline (unmasked)'
    ablations_rows.append(abl_baseline)

    print(f'    Ablation 1 (PTSD masking)...')
    masked_texts = [apply_masking(t) for t in test_texts]
    if mode == 'trunc':
        probs_a1 = run_truncated_inference(model, tokenizer, masked_texts, device)
    else:
        probs_a1 = run_chunkpool_inference(model, tokenizer, masked_texts, device)
    abl1 = compute_metrics(probs_a1, test_labels, val_thresh)
    abl1['condition'] = 'Ablation 1: PTSD masking'
    ablations_rows.append(abl1)

    print(f'    Ablation 2 (PMH removed)...')
    no_pmh_texts = no_pmh_test_df['note_text'].tolist()
    if mode == 'trunc':
        probs_a2 = run_truncated_inference(model, tokenizer, no_pmh_texts, device)
    else:
        probs_a2 = run_chunkpool_inference(model, tokenizer, no_pmh_texts, device)
    # Recompute AUPRC at val_thresh — but threshold may now miss recall=0.85.
    # Report at val_thresh AND at re-tuned threshold (matches Longformer ablation
    # report which also re-tunes).
    abl2 = compute_metrics(probs_a2, test_labels, val_thresh)
    abl2['condition'] = 'Ablation 2: PMH removed (val threshold)'
    abl2_t = threshold_at_recall(probs_a2, test_labels, 0.85)
    abl2_re = compute_metrics(probs_a2, test_labels, abl2_t)
    abl2_re['condition'] = 'Ablation 2: PMH removed (recall>=0.85 retuned)'
    ablations_rows.append(abl2)
    ablations_rows.append(abl2_re)

    pd.DataFrame(ablations_rows).to_csv(
        f'{RESULTS_METRICS}/ewang163_ablation_results_{suffix}.csv',
        index=False)
    print(f'    Ablation deltas: A1={abl1["AUPRC"] - abl_baseline["AUPRC"]:+.4f} '
          f'A2={abl2_re["AUPRC"] - abl_baseline["AUPRC"]:+.4f}')

    return {
        'mode': mode,
        'val_threshold': val_thresh,
        'val_metrics': val_m,
        'test_metrics': test_m,
        'utility': util,
        'calibration': {'ece_raw': round(ece_raw, 4),
                        'ece_platt': round(ece_cal, 4),
                        'ece_elkan_noto': round(ece_en, 4),
                        'elkan_noto_c': round(c_est, 4)},
        'dca': {'max_nb_2pct': round(nb2_max, 4),
                'max_nb_5pct': round(nb5_max, 4)},
        'ablation': {'baseline_AUPRC': abl_baseline['AUPRC'],
                     'A1_PTSD_masking_AUPRC': abl1['AUPRC'],
                     'A1_delta': round(abl1['AUPRC'] - abl_baseline['AUPRC'], 4),
                     'A2_no_PMH_AUPRC_retuned': abl2_re['AUPRC'],
                     'A2_delta': round(abl2_re['AUPRC'] - abl_baseline['AUPRC'], 4)},
    }


def error_analysis(test_with_demo, probs, labels, threshold, mode):
    """Aggregate stats for FP / FN / overall."""
    df = test_with_demo.copy()
    df['predicted_prob'] = probs
    df['pred'] = (probs >= threshold).astype(int)
    df['note_len'] = df['note_text'].str.len()

    fp = df[(df['pred'] == 1) & (df['ptsd_label'] == 0)]
    fn = df[(df['pred'] == 0) & (df['ptsd_label'] == 1)]

    rows = []
    for name, sub in [('false_positive', fp), ('false_negative', fn),
                      ('overall', df)]:
        if len(sub) == 0:
            rows.append({'group': name, 'mode': mode, 'n': 0})
            continue
        # Trauma-term rate: any trauma term in note text
        trauma_count = sum(any(t in str(text).lower() for t in TRAUMA_TERMS)
                           for text in sub['note_text'])
        rows.append({
            'group': name, 'mode': mode, 'n': int(len(sub)),
            'mean_predicted_prob': round(float(sub['predicted_prob'].mean()), 4),
            'mean_note_len': round(float(sub['note_len'].mean()), 1),
            'pct_female': round(float((sub['gender'] == 'F').mean()) * 100, 1),
            'pct_age_20s': round(float((sub['age_group'] == '20s').mean()) * 100, 1),
            'pct_age_other': round(float((sub['age_group'] == 'Other').mean()) * 100, 1),
            'pct_emergency': round(float(sub['is_emergency'].mean()) * 100, 1),
            'pct_white': round(float((sub['race_cat'] == 'White').mean()) * 100, 1),
            'pct_with_trauma_term': round(trauma_count / len(sub) * 100, 1),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('PTSD NLP — BioClinicalBERT FULL Evaluation Suite (truncated + chunk-pool)')
    print('=' * 70, flush=True)

    bench = BenchmarkLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    for d in (RESULTS_PRED, RESULTS_METRICS, RESULTS_FIG, RESULTS_ERR):
        os.makedirs(d, exist_ok=True)

    # Step 1 — load data once
    print('\n[1/8] Loading splits, admissions, proxy ...')
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    adm = pd.read_parquet(ADM_PARQUET)
    proxy_df = pd.read_parquet(PROXY_PARQUET)

    val_labels = val_df['ptsd_label'].values.astype(np.int64)
    test_labels = test_df['ptsd_label'].values.astype(np.int64)

    # Build / cache no-PMH test corpus
    if os.path.exists(NO_PMH_PARQUET):
        print('  loading cached no-PMH test corpus')
        no_pmh_test_df = pd.read_parquet(NO_PMH_PARQUET)
    else:
        print('  building no-PMH test corpus from discharge.csv (stream)...')
        no_pmh_test_df = build_no_pmh_test_corpus(test_df)

    # Build unlabeled sample (500 patients from training pool, fixed seed)
    corpus = pd.read_parquet(CORPUS_PARQUET)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    train_sids = set(splits['train'])
    unlab = corpus[(corpus['group'] == 'unlabeled') &
                   (corpus['subject_id'].isin(train_sids))].copy()
    rng = np.random.RandomState(RANDOM_SEED)
    sample_sids = rng.choice(unlab['subject_id'].unique(),
                             size=min(N_UNLAB_SAMPLE, unlab['subject_id'].nunique()),
                             replace=False)
    unlab_sample_df = unlab[unlab['subject_id'].isin(set(sample_sids))]\
        .drop_duplicates('subject_id', keep='first').reset_index(drop=True)

    # Step 2 — load BERT model + tokenizer
    print(f'\n[2/8] Loading BioClinicalBERT from {BERT_DIR}')
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR, num_labels=2)
    model.to(device)

    summary = {}

    # ── Truncated mode ──────────────────────────────────────────────────────
    print('\n[3/8] Inference: truncated 512')
    with bench.track('bert_full_eval', stage='trunc_val', device='gpu',
                     n_samples=len(val_df)):
        val_probs_t = run_truncated_inference(model, tokenizer,
                                              val_df['note_text'].tolist(), device)
    with bench.track('bert_full_eval', stage='trunc_test', device='gpu',
                     n_samples=len(test_df)):
        test_probs_t = run_truncated_inference(model, tokenizer,
                                               test_df['note_text'].tolist(), device)

    print('\n[4/8] Truncated proxy + unlab inference')
    with bench.track('bert_full_eval', stage='trunc_proxy', device='gpu',
                     n_samples=len(proxy_df)):
        proxy_probs_t = run_truncated_inference(model, tokenizer,
                                                proxy_df['note_text'].tolist(), device)
    with bench.track('bert_full_eval', stage='trunc_unlab', device='gpu',
                     n_samples=len(unlab_sample_df)):
        unlab_probs_t = run_truncated_inference(model, tokenizer,
                                                unlab_sample_df['note_text'].tolist(),
                                                device)

    summary['trunc'] = evaluate_one_mode(
        model, tokenizer, 'trunc',
        val_df, test_df, val_probs_t, test_probs_t,
        val_labels, test_labels, adm, proxy_df, unlab_sample_df,
        no_pmh_test_df, device, bench,
    )
    save_proxy_validation('trunc', proxy_probs_t, unlab_probs_t,
                          summary['trunc']['val_threshold'])

    # ── Chunk-pool mode ─────────────────────────────────────────────────────
    print('\n[5/8] Inference: chunk-pool (512 stride 256)')
    with bench.track('bert_full_eval', stage='chunkpool_val', device='gpu',
                     n_samples=len(val_df)):
        val_probs_c = run_chunkpool_inference(model, tokenizer,
                                              val_df['note_text'].tolist(), device)
    with bench.track('bert_full_eval', stage='chunkpool_test', device='gpu',
                     n_samples=len(test_df)):
        test_probs_c = run_chunkpool_inference(model, tokenizer,
                                               test_df['note_text'].tolist(), device)

    print('\n[6/8] Chunk-pool proxy + unlab inference')
    with bench.track('bert_full_eval', stage='chunkpool_proxy', device='gpu',
                     n_samples=len(proxy_df)):
        proxy_probs_c = run_chunkpool_inference(model, tokenizer,
                                                proxy_df['note_text'].tolist(), device)
    with bench.track('bert_full_eval', stage='chunkpool_unlab', device='gpu',
                     n_samples=len(unlab_sample_df)):
        unlab_probs_c = run_chunkpool_inference(model, tokenizer,
                                                unlab_sample_df['note_text'].tolist(),
                                                device)

    summary['chunkpool'] = evaluate_one_mode(
        model, tokenizer, 'chunkpool',
        val_df, test_df, val_probs_c, test_probs_c,
        val_labels, test_labels, adm, proxy_df, unlab_sample_df,
        no_pmh_test_df, device, bench,
    )
    save_proxy_validation('chunkpool', proxy_probs_c, unlab_probs_c,
                          summary['chunkpool']['val_threshold'])

    # ── Step 7: write top-line JSON summary ─────────────────────────────────
    print('\n[7/8] Writing summary JSON ...')
    with open(f'{RESULTS_METRICS}/ewang163_bioclinbert_full_eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  → {RESULTS_METRICS}/ewang163_bioclinbert_full_eval_summary.json')

    print('\n[8/8] Done. Highlights:')
    for mode, s in summary.items():
        print(f'  {mode:>9}  AUPRC={s["test_metrics"]["AUPRC"]}  '
              f'AUROC={s["test_metrics"]["AUROC"]}  '
              f'F1={s["test_metrics"]["F1"]}  '
              f'ECE={s["calibration"]["ece_raw"]}  '
              f'A1={s["ablation"]["A1_delta"]:+.4f}  '
              f'A2={s["ablation"]["A2_delta"]:+.4f}')


def save_proxy_validation(mode, proxy_probs, unlab_probs, threshold):
    U_stat, mw_p = mannwhitneyu(proxy_probs, unlab_probs, alternative='greater')
    auc = U_stat / (len(proxy_probs) * len(unlab_probs))
    proxy_above = int((proxy_probs >= threshold).sum())
    unlab_above = int((unlab_probs >= threshold).sum())
    rows = [
        {'group': 'Proxy', 'n': len(proxy_probs),
         'median_score': round(float(np.median(proxy_probs)), 4),
         'mean_score': round(float(np.mean(proxy_probs)), 4),
         'q25': round(float(np.percentile(proxy_probs, 25)), 4),
         'q75': round(float(np.percentile(proxy_probs, 75)), 4),
         'frac_above_threshold': round(proxy_above / len(proxy_probs), 4),
         'n_above_threshold': proxy_above},
        {'group': 'Unlabeled sample', 'n': len(unlab_probs),
         'median_score': round(float(np.median(unlab_probs)), 4),
         'mean_score': round(float(np.mean(unlab_probs)), 4),
         'q25': round(float(np.percentile(unlab_probs, 25)), 4),
         'q75': round(float(np.percentile(unlab_probs, 75)), 4),
         'frac_above_threshold': round(unlab_above / len(unlab_probs), 4),
         'n_above_threshold': unlab_above},
        {'group':
            f'Mann-Whitney U={U_stat:.0f}, p={mw_p:.2e}, '
            f'AUC={auc:.4f}, threshold={threshold:.4f}',
         'n': None, 'median_score': None, 'mean_score': None,
         'q25': None, 'q75': None,
         'frac_above_threshold': None, 'n_above_threshold': None},
    ]
    pd.DataFrame(rows).to_csv(
        f'{RESULTS_METRICS}/ewang163_proxy_validation_bert_{mode}.csv',
        index=False)
    print(f'    Proxy AUC ({mode})={auc:.4f}, p={mw_p:.2e}')


if __name__ == '__main__':
    main()
