"""
ewang163_ptsd_evaluate.py
=========================
Evaluates all models on the held-out test set:
  1. Clinical Longformer (PU-trained)
  2. BioClinicalBERT (skipped if no checkpoint found)
  3. TF-IDF + logistic regression
  4. Structured features only

Reports per model:
  AUPRC, AUROC, precision/recall/F1/specificity at recall>=0.85
  Calibration curve (10 decile bins)
  McNemar's test p-value vs. Longformer

Prevalence recalibration (Longformer):
  PPV and NNS at 1%, 2%, 5% prevalence

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

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC              = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
DATA_NOTES         = f'{STUDENT_DIR}/data/notes'
DATA_COHORT        = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_METRICS    = f'{STUDENT_DIR}/results/metrics'
RESULTS_PREDICTIONS = f'{STUDENT_DIR}/results/predictions'

TEST_PARQUET      = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET       = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
LONGFORMER_DIR    = f'{MODEL_DIR}/ewang163_longformer_best'
BIOCLINBERT_DIR   = f'{MODEL_DIR}/ewang163_bioclinbert_best'   # may not exist
TFIDF_VEC_PKL     = f'{MODEL_DIR}/ewang163_tfidf_vectorizer.pkl'
TFIDF_LR_PKL      = f'{MODEL_DIR}/ewang163_tfidf_logreg.pkl'
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


def compute_metrics(probs, labels, name):
    """Full metric suite for one model."""
    auprc = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs)

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
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'precision': round(prec, 4),
        'F1': round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_samples': len(labels),
        'n_pos': int(labels.sum()),
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


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('PTSD NLP — Full Model Evaluation on Test Set')
    print('=' * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Load test data ────────────────────────────────────────────────────
    print('\n[1/7] Loading test data ...')
    test_df = pd.read_parquet(TEST_PARQUET)
    labels = test_df['ptsd_label'].values.astype(np.int64)
    texts = test_df['note_text'].tolist()
    print(f'  {len(test_df):,} test notes '
          f'(pos={int(labels.sum()):,}, unl={int((labels==0).sum()):,})')

    # Load admissions for demographics
    adm = pd.read_parquet(ADM_PARQUET)

    all_results = {}
    model_probs = {}

    # ── Model 1: Clinical Longformer ──────────────────────────────────────
    print('\n[2/7] Clinical Longformer inference ...')
    t0 = time.time()
    lf_tokenizer = AutoTokenizer.from_pretrained(LONGFORMER_DIR)
    lf_model = AutoModelForSequenceClassification.from_pretrained(
        LONGFORMER_DIR, num_labels=2
    )
    lf_model.to(device)
    probs_lf, _ = run_transformer_inference(
        lf_model, lf_tokenizer, texts, labels, device, MAX_LEN
    )
    del lf_model
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    t_lf = time.time() - t0

    metrics_lf = compute_metrics(probs_lf, labels, 'Clinical Longformer (PU)')
    all_results['longformer'] = metrics_lf
    model_probs['longformer'] = probs_lf
    print(f'  AUPRC={metrics_lf["AUPRC"]:.4f}  AUROC={metrics_lf["AUROC"]:.4f}  ({t_lf:.0f}s)')

    # ── Model 2: BioClinicalBERT (if available) ───────────────────────────
    print('\n[3/7] BioClinicalBERT ...')
    if os.path.isdir(BIOCLINBERT_DIR):
        t0 = time.time()
        bert_tokenizer = AutoTokenizer.from_pretrained(BIOCLINBERT_DIR)
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            BIOCLINBERT_DIR, num_labels=2
        )
        bert_model.to(device)
        probs_bert, _ = run_transformer_inference(
            bert_model, bert_tokenizer, texts, labels, device, 512
        )
        del bert_model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        t_bert = time.time() - t0

        metrics_bert = compute_metrics(probs_bert, labels, 'BioClinicalBERT (PU)')
        all_results['bioclinbert'] = metrics_bert
        model_probs['bioclinbert'] = probs_bert
        print(f'  AUPRC={metrics_bert["AUPRC"]:.4f}  AUROC={metrics_bert["AUROC"]:.4f}  ({t_bert:.0f}s)')
    else:
        print('  Checkpoint not found — skipping.')
        all_results['bioclinbert'] = None

    # ── Model 3: TF-IDF + Logistic Regression ────────────────────────────
    print('\n[4/7] TF-IDF + Logistic Regression ...')
    t0 = time.time()
    with open(TFIDF_VEC_PKL, 'rb') as f:
        tfidf_vec = pickle.load(f)
    with open(TFIDF_LR_PKL, 'rb') as f:
        tfidf_lr = pickle.load(f)
    X_tfidf = tfidf_vec.transform(texts)
    probs_tfidf = tfidf_lr.predict_proba(X_tfidf)[:, 1]
    t_tfidf = time.time() - t0

    metrics_tfidf = compute_metrics(probs_tfidf, labels, 'TF-IDF + LogReg')
    all_results['tfidf'] = metrics_tfidf
    model_probs['tfidf'] = probs_tfidf
    print(f'  AUPRC={metrics_tfidf["AUPRC"]:.4f}  AUROC={metrics_tfidf["AUROC"]:.4f}  ({t_tfidf:.0f}s)')

    # ── Model 4: Structured Features + Logistic Regression ────────────────
    print('\n[5/7] Structured Features + Logistic Regression ...')
    t0 = time.time()
    X_struct, feat_cols = build_structured_features(test_df, adm)
    with open(STRUCT_LR_PKL, 'rb') as f:
        struct_lr = pickle.load(f)
    probs_struct = struct_lr.predict_proba(X_struct)[:, 1]
    t_struct = time.time() - t0

    metrics_struct = compute_metrics(probs_struct, labels, 'Structured + LogReg')
    all_results['structured'] = metrics_struct
    model_probs['structured'] = probs_struct
    print(f'  AUPRC={metrics_struct["AUPRC"]:.4f}  AUROC={metrics_struct["AUROC"]:.4f}  ({t_struct:.0f}s)')

    # ── McNemar's tests vs. Longformer ────────────────────────────────────
    print('\n[6/7] McNemar\'s tests vs. Longformer ...')
    mcnemar_results = {}
    lf_thresh = metrics_lf['threshold_recall_85']

    for name, key in [('TF-IDF + LogReg', 'tfidf'),
                      ('Structured + LogReg', 'structured')]:
        if all_results[key] is None:
            continue
        comp_thresh = all_results[key]['threshold_recall_85']
        p_val, b, c = mcnemar_test(
            probs_lf, model_probs[key], labels, lf_thresh, comp_thresh
        )
        mcnemar_results[key] = {'p_value': p_val, 'b_lf_correct_comp_wrong': b,
                                'c_lf_wrong_comp_correct': c}
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f'  Longformer vs {name}: p={p_val:.6f} ({sig})  '
              f'b={b}, c={c}')

    if all_results['bioclinbert'] is not None:
        comp_thresh = all_results['bioclinbert']['threshold_recall_85']
        p_val, b, c = mcnemar_test(
            probs_lf, model_probs['bioclinbert'], labels, lf_thresh, comp_thresh
        )
        mcnemar_results['bioclinbert'] = {
            'p_value': p_val, 'b_lf_correct_comp_wrong': b,
            'c_lf_wrong_comp_correct': c
        }
        print(f'  Longformer vs BioClinicalBERT: p={p_val:.6f}  b={b}, c={c}')

    all_results['mcnemar_vs_longformer'] = mcnemar_results

    # ── Prevalence recalibration (Longformer) ─────────────────────────────
    print('\n  Prevalence recalibration (Longformer):')
    sens_lf = metrics_lf['sensitivity']
    spec_lf = metrics_lf['specificity']
    prevalences = [0.01, 0.02, 0.05]
    recal_results = []
    print(f'    {"Prevalence":>12} {"PPV":>8} {"NNS":>8}')
    for prev in prevalences:
        ppv = recalibrate_ppv(sens_lf, spec_lf, prev)
        nns = 1.0 / ppv if ppv > 0 else float('inf')
        recal_results.append({
            'prevalence': prev,
            'PPV': round(ppv, 4),
            'NNS': round(nns, 1),
        })
        print(f'    {prev:>11.0%} {ppv:>8.4f} {nns:>8.1f}')
    all_results['prevalence_recalibration'] = recal_results

    # ── Subgroup analysis (Longformer AUPRC) ──────────────────────────────
    print('\n[7/7] Subgroup analysis (Longformer) ...')

    # Build demographic info for test patients
    test_hadms = set(test_df['hadm_id'].tolist())
    test_adm = adm[adm['hadm_id'].isin(test_hadms)].copy()
    test_adm = test_adm.sort_values('admittime').drop_duplicates('subject_id', keep='first')
    test_adm['race_cat'] = test_adm['race'].apply(map_race)
    test_adm['age_group'] = test_adm['age_at_admission'].apply(age_decade)
    test_adm['is_emergency'] = (
        test_adm['admission_type'].str.upper().str.contains('EMER', na=False)
    )

    # Map demographics to test_df rows (by subject_id)
    demo_map = test_adm.set_index('subject_id')[
        ['gender', 'race_cat', 'age_group', 'is_emergency']
    ].to_dict('index')

    test_df['gender'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('gender', 'Unknown'))
    test_df['race_cat'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('race_cat', 'Other/Unknown'))
    test_df['age_group'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('age_group', 'Other'))
    test_df['is_emergency'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('is_emergency', False))

    subgroup_results = {}

    def subgroup_auprc(col, label):
        """Compute AUPRC for each value of a categorical column."""
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

    # Sex
    print('  Sex:')
    sex_res = subgroup_auprc('gender', 'Sex')
    subgroup_results['sex'] = sex_res
    for k, v in sex_res.items():
        print(f'    {k}: n={v["n"]}, pos={v["n_pos"]}, AUPRC={v["AUPRC"]}')

    # Age group
    print('  Age group:')
    age_res = subgroup_auprc('age_group', 'Age')
    subgroup_results['age_group'] = age_res
    for k, v in age_res.items():
        print(f'    {k}: n={v["n"]}, pos={v["n_pos"]}, AUPRC={v["AUPRC"]}')

    # Race/ethnicity
    print('  Race/ethnicity:')
    race_res = subgroup_auprc('race_cat', 'Race')
    subgroup_results['race'] = race_res
    for k, v in race_res.items():
        print(f'    {k}: n={v["n"]}, pos={v["n_pos"]}, AUPRC={v["AUPRC"]}')

    # Emergency vs non-emergency
    print('  Emergency admission:')
    emer_res = subgroup_auprc('is_emergency', 'Emergency')
    subgroup_results['emergency'] = emer_res
    for k, v in emer_res.items():
        label = 'Emergency' if k == 'True' else 'Non-emergency'
        print(f'    {label}: n={v["n"]}, pos={v["n_pos"]}, AUPRC={v["AUPRC"]}')

    all_results['subgroup_analysis'] = subgroup_results

    # ── Save results ──────────────────────────────────────────────────────
    # JSON (full results)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nFull results → {RESULTS_JSON}')

    # Summary CSV (one row per model)
    summary_rows = []
    for key in ['longformer', 'bioclinbert', 'tfidf', 'structured']:
        m = all_results[key]
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
        }
        # Add McNemar p-value if available
        if key in mcnemar_results:
            row['mcnemar_p_vs_longformer'] = mcnemar_results[key]['p_value']
        else:
            row['mcnemar_p_vs_longformer'] = None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f'Summary CSV  → {SUMMARY_CSV}')

    # ── Final summary ─────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('TEST SET EVALUATION SUMMARY')
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
        print(f'    prev={r["prevalence"]:.0%}: PPV={r["PPV"]:.4f}, NNS={r["NNS"]:.1f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
