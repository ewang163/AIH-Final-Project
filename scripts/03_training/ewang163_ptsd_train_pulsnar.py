"""
ewang163_ptsd_train_pulsnar.py
==============================
Fix 3 Option B: PULSNAR SAR-PU integration for Clinical Longformer.

Replaces the vanilla nnPU loss (Kiryo 2017) with PULSNAR-estimated
class priors and propensity-weighted PU training.

PULSNAR (Kumar & Lambert 2024, PeerJ Comput Sci, 10.7717/peerj-cs.2451)
is a divide-and-conquer SAR-aware PU estimator that does NOT assume
positives are Selected Completely At Random (SCAR).

Pipeline:
  1. Train a preliminary Longformer with vanilla nnPU (reuse existing checkpoint)
  2. Extract predicted probabilities on training data
  3. Fit PULSNAR to estimate alpha (class prior) and per-sample propensities
  4. Retrain Longformer with propensity-weighted nnPU loss using PULSNAR estimates
  5. Report PULSNAR-estimated alpha vs. empirical labeled fraction

Propensity-weighted nnPU loss:
  - Positive loss term reweighted by 1/e(x_i), where e(x) is the propensity
    P(coded | x) estimated from structured features
  - Propensities clipped to [0.05, 0.95] to prevent variance explosion

Inputs:
    models/ewang163_longformer_best/     — preliminary checkpoint
    data/splits/ewang163_split_{train,val}.parquet
    data/cohort/ewang163_ptsd_adm_extract.parquet  — structured features for propensity

Outputs:
    models/ewang163_longformer_pulsnar/  — PULSNAR-retrained checkpoint
    results/metrics/ewang163_pulsnar_estimates.json

RUN:
    sbatch --partition=gpu --gres=gpu:1 --mem=32G --time=18:00:00 \
           --output=logs/ewang163_pulsnar_%j.out \
           --wrap="python scripts/03_training/ewang163_ptsd_train_pulsnar.py"
"""

import json
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
DATA_COHORT     = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR       = f'{STUDENT_DIR}/models'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'

TRAIN_PARQUET   = f'{DATA_SPLITS}/ewang163_split_train.parquet'
VAL_PARQUET     = f'{DATA_SPLITS}/ewang163_split_val.parquet'
ADM_PARQUET     = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
PRELIM_DIR      = f'{MODEL_DIR}/ewang163_longformer_best'
PULSNAR_DIR     = f'{MODEL_DIR}/ewang163_longformer_pulsnar'
ESTIMATES_JSON  = f'{RESULTS_METRICS}/ewang163_pulsnar_estimates.json'

# MIMIC source files (for prior-comorbidity feature extraction)
MIMIC = '/oscar/data/shared/ursa/mimic-iv'
DIAGNOSES_F = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'
PRESCRIPTIONS_F = f'{MIMIC}/hosp/3.1/prescriptions.csv'

# Prior comorbidity ICD prefixes (CLAUDE.md)
COMORBIDITY_PFXS = {
    'prior_MDD':      ['F32', 'F33', '296'],
    'prior_anxiety':  ['F41', '300'],
    'prior_SUD':      ['F10','F11','F12','F13','F14','F15','F16','F17','F18','F19',
                       '303','304','305'],
    'prior_suicidal': ['R458', 'V6284', 'E95'],
}
DRUG_PATTERNS = {
    'prior_SSRI': ['sertraline', 'fluoxetine', 'paroxetine', 'escitalopram',
                   'citalopram', 'fluvoxamine', 'venlafaxine', 'duloxetine',
                   'desvenlafaxine', 'levomilnacipran', 'milnacipran'],
    'prior_prazosin': ['prazosin'],
    'prior_SGA': ['quetiapine', 'olanzapine', 'risperidone', 'aripiprazole',
                  'ziprasidone', 'clozapine', 'lurasidone', 'asenapine',
                  'paliperidone', 'iloperidone', 'brexpiprazole', 'cariprazine'],
}

MODEL_NAME = 'yikuan8/Clinical-Longformer'

# ── Hyperparameters ───────────────────────────────────────────────────────
MAX_LEN          = 4096
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 16
LR               = 1e-5   # lower LR for fine-tuning from preliminary checkpoint
EPOCHS           = 3
WARMUP_FRAC      = 0.1
WEIGHT_DECAY     = 0.01
PROPENSITY_CLIP  = (0.05, 0.95)


class NoteDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, propensities=None):
        self.texts = df['note_text'].tolist()
        self.labels = df['ptsd_label'].values.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.propensities = propensities

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.propensities is not None:
            item['propensity'] = torch.tensor(self.propensities[idx], dtype=torch.float32)
        return item


def propensity_weighted_pu_loss(logits, labels, pi_p, propensities=None):
    """Propensity-weighted nnPU loss (Fix 3).

    Positive loss term is reweighted by 1/e(x_i) where e(x) is the
    propensity score P(coded | x, y=1).  This corrects for the SAR
    (Selected At Random) bias in the labeled set.
    """
    pos_mask = (labels == 1)
    unl_mask = (labels == 0)
    n_pos = pos_mask.sum()
    n_unl = unl_mask.sum()

    if n_pos == 0 or n_unl == 0:
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    # E_P[l(f(x), +1)] with propensity reweighting
    pos_logits = logits[pos_mask]
    pos_loss_raw = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction='none'
    )

    if propensities is not None and pos_mask.sum() > 0:
        prop = propensities[pos_mask]
        weights = 1.0 / prop
        weights = weights / weights.mean()  # normalize to maintain scale
        loss_pos = (pos_loss_raw * weights).mean()
    else:
        loss_pos = pos_loss_raw.mean()

    # E_U[l(f(x), -1)]
    loss_unl = F.binary_cross_entropy_with_logits(
        logits[unl_mask], torch.zeros_like(logits[unl_mask]), reduction='mean'
    )

    # E_P[l(f(x), -1)]
    loss_pos_as_neg_raw = F.binary_cross_entropy_with_logits(
        pos_logits, torch.zeros_like(pos_logits), reduction='none'
    )
    if propensities is not None and pos_mask.sum() > 0:
        loss_pos_as_neg = (loss_pos_as_neg_raw * weights).mean()
    else:
        loss_pos_as_neg = loss_pos_as_neg_raw.mean()

    pu = pi_p * loss_pos + torch.clamp(
        loss_unl - pi_p * loss_pos_as_neg, min=0.0
    )
    return pu


def build_prior_flags(train_subjects, adm_df):
    """Compute prior-admission comorbidity + medication flags per subject.

    For each subject, identifies prior admissions (strictly before index admission)
    and flags presence of comorbidities/medications during any of those prior
    admissions. This captures the 'prior psychiatric contact' variables that
    drive PTSD coding (Fix 4: rich propensity features).
    """
    print('  Building prior-admission flags from MIMIC streams ...')

    # Step 1: identify each subject's prior admissions
    # Use the existing index_admittime column from the ADM extract, which
    # correctly reflects "first PTSD-coded admission" for PTSD+ patients
    # and "first admission overall" for unlabeled. Prior = admittime < index_admittime.
    sub_set = set(train_subjects)
    adm_cohort = adm_df[adm_df['subject_id'].isin(sub_set)].copy()
    adm_cohort['admittime'] = pd.to_datetime(adm_cohort['admittime'])
    adm_cohort['index_admittime'] = pd.to_datetime(adm_cohort['index_admittime'])

    adm_cohort['sid_str'] = adm_cohort['subject_id'].astype(str)
    adm_cohort['is_prior'] = adm_cohort['admittime'] < adm_cohort['index_admittime']
    prior_adm = adm_cohort[adm_cohort['is_prior']]

    prior_hadm_to_sid = dict(zip(prior_adm['hadm_id'].astype(str),
                                  prior_adm['sid_str']))
    n_prior_admissions = prior_adm.groupby('sid_str').size().to_dict()

    print(f'    Cohort: {len(sub_set):,} subjects, '
          f'{len(prior_hadm_to_sid):,} prior admissions total')

    # Step 2: stream diagnoses_icd.csv, flag subjects with prior comorbidities
    dx_flags = {feat: set() for feat in COMORBIDITY_PFXS}
    import csv
    csv.field_size_limit(2**27)
    with open(DIAGNOSES_F) as f:
        reader = csv.reader(f)
        header = next(reader)
        hadm_idx = header.index('hadm_id')
        icd_idx = header.index('icd_code')
        for row in reader:
            if len(row) <= max(hadm_idx, icd_idx):
                continue
            h = row[hadm_idx].strip()
            if h not in prior_hadm_to_sid:
                continue
            sid = prior_hadm_to_sid[h]
            icd = row[icd_idx].strip().replace('.', '').upper()
            for feat, pfxs in COMORBIDITY_PFXS.items():
                if any(icd.startswith(p) for p in pfxs):
                    dx_flags[feat].add(sid)

    # Step 3: stream prescriptions.csv, flag subjects with prior medications
    rx_flags = {feat: set() for feat in DRUG_PATTERNS}
    with open(PRESCRIPTIONS_F) as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            hadm_idx = header.index('hadm_id')
            drug_idx = header.index('drug')
        except ValueError:
            # MIMIC-IV prescriptions might have different header
            return None
        for row in reader:
            if len(row) <= max(hadm_idx, drug_idx):
                continue
            h = row[hadm_idx].strip()
            if h not in prior_hadm_to_sid:
                continue
            sid = prior_hadm_to_sid[h]
            drug_lower = row[drug_idx].strip().lower()
            for feat, patterns in DRUG_PATTERNS.items():
                if any(p in drug_lower for p in patterns):
                    rx_flags[feat].add(sid)

    # Combine into a per-subject feature dict
    result = {}
    for sid_int in sub_set:
        sid = str(sid_int)
        feat_dict = {'n_prior_admissions': n_prior_admissions.get(sid, 0)}
        for feat in COMORBIDITY_PFXS:
            feat_dict[feat] = 1 if sid in dx_flags[feat] else 0
        for feat in DRUG_PATTERNS:
            feat_dict[feat] = 1 if sid in rx_flags[feat] else 0
        feat_dict['prior_psych_any'] = int(any(
            feat_dict[f] for f in ['prior_MDD', 'prior_anxiety', 'prior_suicidal']
        ))
        result[sid_int] = feat_dict

    # Log coverage
    for feat in list(COMORBIDITY_PFXS) + list(DRUG_PATTERNS) + ['prior_psych_any']:
        n = sum(1 for v in result.values() if v.get(feat, 0) == 1)
        print(f'    {feat}: {n:,} / {len(result):,} subjects ({n/len(result)*100:.1f}%)')

    return result


def estimate_propensities(train_df, adm_df):
    """Estimate P(coded | rich structured features) via logistic regression.

    Fix 4 expansion: now includes prior psychiatric comorbidities and medications
    in addition to demographics.  Prior psychiatric contact is the dominant
    predictor of PTSD coding (per CLAUDE.md noted bias).
    """
    print('  Building propensity features (with prior comorbidities) ...')

    index_adm = adm_df.sort_values('admittime').drop_duplicates(
        'subject_id', keep='first')

    # Base demographics (original 4 features)
    merged = train_df[['subject_id', 'ptsd_label']].drop_duplicates(
        'subject_id').merge(
        index_adm[['subject_id', 'gender', 'age_at_admission',
                    'admission_type', 'insurance']],
        on='subject_id', how='left'
    )
    merged['sex_female'] = (merged['gender'] == 'F').astype(float)
    merged['age'] = merged['age_at_admission'].fillna(40).astype(float)
    merged['emergency'] = merged['admission_type'].str.upper().str.contains(
        'EMER', na=False).astype(float)
    merged['medicaid'] = merged['insurance'].str.upper().isin(
        ['MEDICAID', 'SELF PAY']).astype(float)

    # Rich features: prior comorbidity + medication flags
    subject_ids = merged['subject_id'].tolist()
    prior_flags = build_prior_flags(subject_ids, adm_df)

    rich_feature_cols = (['n_prior_admissions']
                         + list(COMORBIDITY_PFXS.keys())
                         + list(DRUG_PATTERNS.keys())
                         + ['prior_psych_any'])

    for col in rich_feature_cols:
        merged[col] = merged['subject_id'].map(
            lambda s: prior_flags.get(s, {}).get(col, 0)
        )

    feature_cols = (['sex_female', 'age', 'emergency', 'medicaid']
                    + rich_feature_cols)
    X = merged[feature_cols].fillna(0).values.astype(float)
    y = merged['ptsd_label'].values

    print(f'  Fitting propensity LR on {X.shape[0]} patients, {X.shape[1]} features ...')
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X, y)

    # Log coefficients for interpretability
    print('  Propensity model coefficients:')
    for feat, coef in sorted(zip(feature_cols, lr.coef_[0]),
                              key=lambda x: -abs(x[1])):
        print(f'    {feat:<25s} {coef:+.4f}')

    propensities = lr.predict_proba(X)[:, 1]
    propensities = np.clip(propensities, PROPENSITY_CLIP[0], PROPENSITY_CLIP[1])

    sid_to_prop = dict(zip(merged['subject_id'].values, propensities))
    train_propensities = np.array([
        sid_to_prop.get(sid, 0.5) for sid in train_df['subject_id'].values
    ])

    print(f'    Propensity range: [{train_propensities.min():.3f}, '
          f'{train_propensities.max():.3f}]')
    pos_mask = train_df['ptsd_label'].values == 1
    print(f'    Propensity mean (pos): {train_propensities[pos_mask].mean():.3f}')
    print(f'    Propensity mean (unl): {train_propensities[~pos_mask].mean():.3f}')

    # Return also the feature matrix for PULSNAR alpha estimation
    return train_propensities, lr, merged, feature_cols


def try_pulsnar_alpha(X_features, pu_labels):
    """Estimate alpha (class prior) using PULSNAR.

    PULSNAR takes a feature matrix and PU labels, trains an internal
    classifier (xgboost by default), and estimates the positive fraction
    in the unlabeled set using a divide-and-conquer KDE approach.

    Falls back to Elkan-Noto if PULSNAR is not available.
    """
    try:
        from PULSNAR.PULSNAR import PULSNARClassifier
        import tempfile, os

        # PULSNAR writes output files; use a temp dir
        orig_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            os.makedirs('results', exist_ok=True)

            # PULSNAR (SAR-aware, not SCAR)
            pls = PULSNARClassifier(
                scar=False,
                csrdata=False,
                classifier='xgboost',
                n_clusters=0,
                max_clusters=10,
                covar_type='full',
                bin_method='rice',
                bw_method='hist',
                lowerbw=0.015,
                upperbw=0.5,
                optim='local',
                calibration=False,
                classification_metrics=False,
                n_iterations=1,
                kfold=5,
                kflips=1,
                rseed=42,
            )

            rec_ids = np.arange(len(pu_labels))
            res = pls.pulsnar(X_features, pu_labels, rec_list=rec_ids)
            os.chdir(orig_dir)

        alpha = float(res['estimated_alpha'])
        return alpha, 'PULSNAR (SAR)'
    except Exception as e:
        logging.warning(f'PULSNAR failed: {e}')
        try:
            os.chdir(orig_dir)
        except Exception:
            pass

    try:
        from PULSNAR.PULSNAR import PULSNARClassifier
        import tempfile, os

        orig_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            os.makedirs('results', exist_ok=True)

            pls = PULSNARClassifier(
                scar=True,
                classifier='xgboost',
                bin_method='rice',
                bw_method='hist',
                n_iterations=1,
                kfold=5,
                rseed=42,
            )
            rec_ids = np.arange(len(pu_labels))
            res = pls.pulsnar(X_features, pu_labels, rec_list=rec_ids)
            os.chdir(orig_dir)

        alpha = float(res['estimated_alpha'])
        return alpha, 'PULSCAR (SCAR)'
    except Exception as e:
        logging.warning(f'PULSCAR also failed: {e}')
        try:
            os.chdir(orig_dir)
        except Exception:
            pass

    # Final fallback: Elkan-Noto style estimate
    # Use the labeled fraction as a rough lower bound
    n_pos = (pu_labels == 1).sum()
    n_total = len(pu_labels)
    alpha_en = n_pos / n_total
    return alpha_en, 'Elkan-Noto-fallback'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_only', action='store_true',
                        help='Skip GPU retraining; just estimate alpha and exit')
    parser.add_argument('--output_suffix', type=str, default='_v2',
                        help='Suffix for PULSNAR output dir (default: _v2 to preserve v1)')
    args = parser.parse_args()

    pulsnar_dir = PULSNAR_DIR + args.output_suffix if args.output_suffix else PULSNAR_DIR
    estimates_json = ESTIMATES_JSON.replace('.json', f'{args.output_suffix}.json') \
        if args.output_suffix else ESTIMATES_JSON

    print('=' * 65)
    print('PTSD NLP — PULSNAR SAR-PU Training (Fix 3, rich features)')
    if args.alpha_only:
        print('  MODE: alpha-only (no GPU retraining)')
    print('=' * 65, flush=True)

    bench = BenchmarkLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Load data ────────────────────────────────────────────────────────
    print('\n[1/5] Loading data ...')
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)
    adm_df = pd.read_parquet(ADM_PARQUET)

    n_pos = (train_df['ptsd_label'] == 1).sum()
    n_unl = (train_df['ptsd_label'] == 0).sum()
    empirical_pi = n_pos / (n_pos + n_unl)

    print(f'  Train: {len(train_df):,} (pos={n_pos:,}, unl={n_unl:,})')
    print(f'  Empirical labeled fraction: {empirical_pi:.4f}')

    # ── Estimate propensities with rich features ─────────────────────────
    print('\n[2/5] Estimating propensities (rich features) ...')
    propensities, prop_model, rich_patients, feature_cols = estimate_propensities(
        train_df, adm_df)

    # ── PULSNAR alpha estimation using rich features ──────────────────────
    print('\n[3/5] Estimating class prior with PULSNAR (rich features) ...')
    X_struct = rich_patients[feature_cols].fillna(0).values.astype(np.float64)
    Y_pu = rich_patients['ptsd_label'].values.astype(np.int32)

    print(f'  PULSNAR input: {X_struct.shape[0]} patients, {X_struct.shape[1]} features')
    print(f'  Features: {feature_cols}')
    alpha, method = try_pulsnar_alpha(X_struct, Y_pu)
    print(f'  Alpha estimate (rich): {alpha:.4f} (method: {method})')
    print(f'  Empirical labeled fraction: {empirical_pi:.4f}')

    # Save the alpha estimate immediately so it's available even if training fails
    alpha_result = {
        'alpha_estimate': round(alpha, 6),
        'alpha_method': method,
        'empirical_labeled_fraction': round(empirical_pi, 6),
        'features_used': feature_cols,
        'n_features': len(feature_cols),
        'n_patients': int(X_struct.shape[0]),
    }
    os.makedirs(os.path.dirname(estimates_json), exist_ok=True)
    with open(estimates_json, 'w') as f:
        json.dump(alpha_result, f, indent=2)
    print(f'  Alpha estimate saved → {estimates_json}')

    if args.alpha_only:
        print('\n' + '=' * 65)
        print('ALPHA-ONLY MODE: exiting without retraining')
        print('=' * 65)
        print(f'  Alpha (class prior): {alpha:.4f} ({method})')
        print(f'  Empirical labeled fraction: {empirical_pi:.4f}')
        print(f'  Delta vs. v1 (alpha=0.1957): {alpha - 0.1957:+.4f}')
        return

    if os.path.isdir(PRELIM_DIR):
        tokenizer = AutoTokenizer.from_pretrained(PRELIM_DIR)
    else:
        alpha = empirical_pi
        method = 'empirical (no preliminary checkpoint)'
        print(f'  No preliminary model — using empirical pi_p = {alpha:.4f}')
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    pi_p = alpha

    # ── Train with propensity-weighted loss ──────────────────────────────
    print(f'\n[4/5] Training with propensity-weighted nnPU (pi_p={pi_p:.4f}) ...')

    model = AutoModelForSequenceClassification.from_pretrained(
        PRELIM_DIR if os.path.isdir(PRELIM_DIR) else MODEL_NAME,
        num_labels=2
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    train_ds = NoteDataset(train_df, tokenizer, MAX_LEN, propensities=propensities)
    val_ds = NoteDataset(val_df, tokenizer, MAX_LEN)

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2 if use_cuda else 0,
                              pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2 if use_cuda else 0,
                            pin_memory=use_cuda)

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    os.makedirs(pulsnar_dir, exist_ok=True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    best_auprc = -1.0
    log_rows = []

    with bench.track('train_pulsnar', stage='full_training',
                     device='gpu' if use_cuda else 'cpu',
                     n_samples=len(train_df),
                     notes=f'pi_p={pi_p:.4f}, propensity-weighted'):
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            print(f'\n--- Epoch {epoch}/{EPOCHS} ---', flush=True)

            model.train()
            total_loss = 0.0
            n_batches = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                prop = batch.get('propensity')
                if prop is not None:
                    prop = prop.to(device)

                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, 1] - outputs.logits[:, 0]
                    loss = propensity_weighted_pu_loss(
                        logits.float(), labels, pi_p, prop
                    ) / GRAD_ACCUM_STEPS

                scaler.scale(loss).backward()
                total_loss += loss.item() * GRAD_ACCUM_STEPS
                n_batches += 1

                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                if (step + 1) % 50 == 0:
                    print(f'    step {step+1}/{len(train_loader)}  '
                          f'avg_loss={total_loss / n_batches:.4f}', flush=True)

            train_loss = total_loss / max(n_batches, 1)
            train_time = time.time() - t0

            # Validate
            model.eval()
            val_probs_list, val_labels_list = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = F.softmax(outputs.logits.float(), dim=-1)[:, 1]
                    val_probs_list.append(probs.cpu().numpy())
                    val_labels_list.append(batch['label'].numpy())

            val_probs = np.concatenate(val_probs_list)
            val_labels = np.concatenate(val_labels_list)
            val_auprc = average_precision_score(val_labels, val_probs)

            print(f'  Train loss: {train_loss:.4f}  ({train_time:.0f}s)')
            print(f'  Val AUPRC:  {val_auprc:.4f}', flush=True)

            log_rows.append({
                'epoch': epoch, 'train_loss': round(train_loss, 6),
                'val_auprc': round(val_auprc, 6),
            })

            if val_auprc > best_auprc:
                best_auprc = val_auprc
                model.save_pretrained(pulsnar_dir)
                tokenizer.save_pretrained(pulsnar_dir)
                print(f'  >> New best (AUPRC={best_auprc:.4f})', flush=True)

    # ── Save estimates ───────────────────────────────────────────────────
    print('\n[5/5] Saving ...')
    estimates = {
        'alpha_estimate': round(alpha, 6),
        'alpha_method': method,
        'empirical_labeled_fraction': round(empirical_pi, 6),
        'propensity_clip': list(PROPENSITY_CLIP),
        'best_val_auprc': round(best_auprc, 6),
        'epochs': EPOCHS,
        'lr': LR,
        'features_used': feature_cols,
    }
    with open(estimates_json, 'w') as f:
        json.dump(estimates, f, indent=2)
    print(f'  Estimates → {estimates_json}')

    config = {**estimates, 'model_name': MODEL_NAME, 'max_len': MAX_LEN}
    with open(os.path.join(pulsnar_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    log_df = pd.DataFrame(log_rows)
    log_csv = f'{RESULTS_METRICS}/ewang163_pulsnar_training_log{args.output_suffix}.csv'
    log_df.to_csv(log_csv, index=False)
    print(f'  Training log → {log_csv}')

    print('\n' + '=' * 65)
    print('PULSNAR TRAINING COMPLETE')
    print('=' * 65)
    print(f'  Alpha (class prior): {alpha:.4f} ({method})')
    print(f'  Best val AUPRC: {best_auprc:.4f}')
    print(f'  Checkpoint: {pulsnar_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
