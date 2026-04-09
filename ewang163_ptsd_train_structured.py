"""
ewang163_ptsd_train_structured.py
=================================
Structured-features-only logistic regression baseline for PTSD detection.

Features (all from prior admissions, NOT the index admission):
  - age_at_admission, sex_female, race (one-hot 5 categories)
  - n_prior_admissions, los_days, emergency_admission, medicaid_selfpay
  - dx_MDD, dx_anxiety, dx_SUD, dx_TBI, dx_pain, dx_suicidal
  - rx_ssri_snri, rx_prazosin, rx_SGA

Inputs:
    ewang163_ptsd_adm_extract.parquet
    ewang163_split_train.parquet
    ewang163_split_val.parquet
    MIMIC-IV source: diagnoses_icd.csv, prescriptions.csv

Outputs:
    ewang163_structured_logreg.pkl
    ewang163_structured_features.json
    ewang163_structured_val_results.txt

RUN:
    sbatch ewang163_ptsd_train_structured.sh
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC = '/oscar/data/shared/ursa/mimic-iv'
OUT   = '/oscar/data/class/biol1595_2595/students/ewang163'

DIAGNOSES_F     = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'
PRESCRIPTIONS_F = f'{MIMIC}/hosp/3.1/prescriptions.csv'

ADM_PARQUET   = f'{OUT}/ewang163_ptsd_adm_extract.parquet'
TRAIN_PARQUET = f'{OUT}/ewang163_split_train.parquet'
VAL_PARQUET   = f'{OUT}/ewang163_split_val.parquet'

LOGREG_PKL    = f'{OUT}/ewang163_structured_logreg.pkl'
FEATURES_JSON = f'{OUT}/ewang163_structured_features.json'
RESULTS_TXT   = f'{OUT}/ewang163_structured_val_results.txt'

# ── ICD comorbidity prefixes (ICD-10 + ICD-9, from CLAUDE.md) ────────────
COMORBIDITY_PFXS = {
    'dx_MDD':      ['F32', 'F33', '296'],
    'dx_anxiety':  ['F41', '300'],
    'dx_SUD':      ['F10','F11','F12','F13','F14','F15','F16','F17','F18','F19',
                    '303','304','305'],
    'dx_TBI':      ['S06','800','801','802','803','804','850','851','852','853','854'],
    'dx_pain':     ['G89', '338'],
    'dx_suicidal': ['R458', 'V6284', 'E95'],
}

# ── Drug patterns (lowercase substring match) ────────────────────────────
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

# ── Race mapping (same as table1) ────────────────────────────────────────
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

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def precision_at_recall(y_true, y_score, target_recall=0.85):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = recall >= target_recall
    if valid.any():
        return precision[valid].max(), thresholds[valid[:-1]].min() if valid[:-1].any() else 0.5
    return 0.0, 0.5


def main():
    print('=' * 65)
    print('PTSD NLP Project — Structured Features Baseline')
    print('=' * 65)

    # ── Step 1: Load admissions extract + splits ──────────────────────────
    print('\n[1/6] Loading admissions extract and splits ...')
    adm = pd.read_parquet(ADM_PARQUET)
    train_df = pd.read_parquet(TRAIN_PARQUET)[['subject_id', 'hadm_id', 'ptsd_label']].drop_duplicates()
    val_df   = pd.read_parquet(VAL_PARQUET)[['subject_id', 'hadm_id', 'ptsd_label']].drop_duplicates()

    train_sids = set(train_df['subject_id'].unique())
    val_sids   = set(val_df['subject_id'].unique())
    all_sids   = train_sids | val_sids

    print(f'  Admissions extract: {len(adm):,} rows')
    print(f'  Train: {len(train_sids):,} patients, Val: {len(val_sids):,} patients')

    # Filter admissions to cohort patients
    adm = adm[adm['subject_id'].isin(all_sids)].copy()

    # Index admission = the hadm_id(s) present in the corpus splits
    train_index_hadms = set(train_df['hadm_id'].unique())
    val_index_hadms   = set(val_df['hadm_id'].unique())
    index_hadms       = train_index_hadms | val_index_hadms

    # Get index_admittime per patient (earliest if multiple notes)
    index_adm = adm[adm['hadm_id'].isin(index_hadms)][['subject_id', 'hadm_id', 'admittime']].copy()
    index_adm = index_adm.sort_values('admittime').drop_duplicates('subject_id', keep='first')
    index_adm = index_adm.rename(columns={'admittime': 'idx_admittime'})

    # Prior admissions = strictly before index
    prior_adm = adm.merge(index_adm[['subject_id', 'idx_admittime']], on='subject_id', how='inner')
    prior_adm = prior_adm[prior_adm['admittime'] < prior_adm['idx_admittime']].copy()
    prior_hadms = set(prior_adm['hadm_id'].unique())

    print(f'  Patients with prior admissions: {prior_adm["subject_id"].nunique():,}')
    print(f'  Prior admission rows: {len(prior_adm):,}')

    # ── Step 2: Build index-admission demographic features ────────────────
    print('\n[2/6] Building demographic features from index admission ...')

    idx_rows = adm[adm['hadm_id'].isin(index_hadms)].copy()
    idx_rows = idx_rows.sort_values('admittime').drop_duplicates('subject_id', keep='first')

    idx_rows['sex_female'] = (idx_rows['gender'] == 'F').astype(int)
    idx_rows['race_cat'] = idx_rows['race'].apply(map_race)
    idx_rows['emergency_admission'] = idx_rows['admission_type'].str.upper().str.contains('EMER', na=False).astype(int)
    idx_rows['medicaid_selfpay'] = idx_rows['insurance'].str.upper().isin(['MEDICAID', 'SELF PAY']).astype(int)
    idx_rows['los_days'] = (idx_rows['dischtime'] - idx_rows['admittime']).dt.total_seconds() / 86400.0

    # One-hot race
    race_dummies = pd.get_dummies(idx_rows['race_cat'], prefix='race')
    idx_rows = pd.concat([idx_rows, race_dummies], axis=1)
    idx_rows = idx_rows.drop(columns=['race_cat'])

    # Prior admission count
    prior_counts = prior_adm.groupby('subject_id').size().reset_index(name='n_prior_admissions')
    idx_rows = idx_rows.merge(prior_counts, on='subject_id', how='left')
    idx_rows['n_prior_admissions'] = idx_rows['n_prior_admissions'].fillna(0).astype(int)

    demo_cols = ['subject_id', 'age_at_admission', 'sex_female', 'los_days',
                 'emergency_admission', 'medicaid_selfpay', 'n_prior_admissions']
    race_cols = sorted([c for c in idx_rows.columns if c.startswith('race_')])
    demo = idx_rows[demo_cols + race_cols].copy()

    print(f'  Demographic features: {len(demo):,} patients, {len(demo_cols) + len(race_cols) - 1} features')

    # ── Step 3: Build comorbidity features from prior diagnoses ───────────
    print('\n[3/6] Streaming diagnoses for prior-admission comorbidity flags ...')

    prior_hadm_to_sid = dict(zip(prior_adm['hadm_id'].astype(str), prior_adm['subject_id'].astype(str)))

    dx_flags = {feat: set() for feat in COMORBIDITY_PFXS}
    n_dx_rows = 0

    with open(DIAGNOSES_F) as f:
        header = next(f)
        for line in f:
            parts = line.strip().split(',')
            hadm_id_str = parts[1]
            if hadm_id_str not in prior_hadm_to_sid:
                continue
            sid = prior_hadm_to_sid[hadm_id_str]
            icd_code = parts[3].strip().replace('.', '').upper()
            n_dx_rows += 1
            for feat, pfxs in COMORBIDITY_PFXS.items():
                if any(icd_code.startswith(p) for p in pfxs):
                    dx_flags[feat].add(sid)

    print(f'  Scanned {n_dx_rows:,} diagnosis rows from prior admissions')
    for feat, sids in dx_flags.items():
        print(f'    {feat}: {len(sids):,} patients')

    dx_records = []
    for sid in demo['subject_id'].values:
        row = {'subject_id': sid}
        for feat in COMORBIDITY_PFXS:
            row[feat] = 1 if str(sid) in dx_flags[feat] else 0
        dx_records.append(row)
    dx_df = pd.DataFrame(dx_records)

    # ── Step 4: Build medication features from prior prescriptions ────────
    print('\n[4/6] Streaming prescriptions for prior-admission medication flags ...')

    rx_flags = {feat: set() for feat in DRUG_PATTERNS}
    n_rx_rows = 0

    with open(PRESCRIPTIONS_F) as f:
        header_line = next(f)
        cols = [c.strip() for c in header_line.split(',')]
        hadm_idx = cols.index('hadm_id')
        drug_idx = cols.index('drug')

        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(hadm_idx, drug_idx):
                continue
            hadm_id_str = parts[hadm_idx].strip()
            if hadm_id_str not in prior_hadm_to_sid:
                continue
            sid = prior_hadm_to_sid[hadm_id_str]
            drug_lower = parts[drug_idx].strip().lower()
            n_rx_rows += 1
            for feat, patterns in DRUG_PATTERNS.items():
                if any(p in drug_lower for p in patterns):
                    rx_flags[feat].add(sid)

    print(f'  Scanned {n_rx_rows:,} prescription rows from prior admissions')
    for feat, sids in rx_flags.items():
        print(f'    {feat}: {len(sids):,} patients')

    rx_records = []
    for sid in demo['subject_id'].values:
        row = {'subject_id': sid}
        for feat in DRUG_PATTERNS:
            row[feat] = 1 if str(sid) in rx_flags[feat] else 0
        rx_records.append(row)
    rx_df = pd.DataFrame(rx_records)

    # ── Step 5: Merge all features ────────────────────────────────────────
    print('\n[5/6] Merging features and training model ...')

    features = demo.merge(dx_df, on='subject_id').merge(rx_df, on='subject_id')

    train_labels = train_df[['subject_id', 'ptsd_label']].drop_duplicates('subject_id')
    val_labels   = val_df[['subject_id', 'ptsd_label']].drop_duplicates('subject_id')

    train_feat = features.merge(train_labels, on='subject_id', how='inner')
    val_feat   = features.merge(val_labels, on='subject_id', how='inner')

    feature_cols = [c for c in features.columns if c != 'subject_id']
    print(f'  Feature columns ({len(feature_cols)}): {feature_cols}')
    print(f'  Train: {len(train_feat):,} patients, Val: {len(val_feat):,} patients')

    X_train = train_feat[feature_cols].values.astype(np.float32)
    y_train = train_feat['ptsd_label'].values
    X_val   = val_feat[feature_cols].values.astype(np.float32)
    y_val   = val_feat['ptsd_label'].values

    # Impute missing with column medians
    for j in range(X_train.shape[1]):
        col_median = np.nanmedian(X_train[:, j])
        mask_train = np.isnan(X_train[:, j])
        mask_val   = np.isnan(X_val[:, j])
        if mask_train.any():
            X_train[mask_train, j] = col_median
        if mask_val.any():
            X_val[mask_val, j] = col_median

    print(f'  Train pos: {y_train.sum():,}, unl: {(y_train==0).sum():,}')
    print(f'  Val   pos: {y_val.sum():,}, unl: {(y_val==0).sum():,}')

    # Tune C
    print(f'\n  Tuning C on validation AUPRC ... C grid: {C_GRID}')
    best_auprc = -1
    best_C = None
    best_model = None
    results = []

    for C in C_GRID:
        model = LogisticRegression(
            C=C, class_weight='balanced', solver='lbfgs',
            max_iter=1000, random_state=42,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, y_prob)
        auroc = roc_auc_score(y_val, y_prob)
        results.append({'C': C, 'AUPRC': auprc, 'AUROC': auroc})
        marker = ' <-- best' if auprc > best_auprc else ''
        print(f'    C={C:<8}  AUPRC={auprc:.4f}  AUROC={auroc:.4f}{marker}')
        if auprc > best_auprc:
            best_auprc = auprc
            best_C = C
            best_model = model

    # ── Step 6: Final metrics, feature coefficients, save ─────────────────
    print(f'\n[6/6] Best C={best_C} — final metrics and saving ...')

    y_prob_best = best_model.predict_proba(X_val)[:, 1]
    auprc_final = average_precision_score(y_val, y_prob_best)
    auroc_final = roc_auc_score(y_val, y_prob_best)
    prec_r85, thresh_r85 = precision_at_recall(y_val, y_prob_best, 0.85)

    print(f'  Validation AUPRC:            {auprc_final:.4f}')
    print(f'  Validation AUROC:            {auroc_final:.4f}')
    print(f'  Precision @ recall>=0.85:    {prec_r85:.4f}  (threshold={thresh_r85:.4f})')

    # Feature coefficients ranked by magnitude
    coefs = best_model.coef_[0]
    coef_pairs = sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
    print('\n  Feature coefficients (ranked by |coef|):')
    for feat, coef in coef_pairs:
        print(f'    {feat:<25} {coef:+.4f}')

    # Save model
    with open(LOGREG_PKL, 'wb') as f:
        pickle.dump(best_model, f)
    print(f'\n  Saved: {LOGREG_PKL}')

    # Save feature list + coefficients
    feat_info = {
        'feature_columns': feature_cols,
        'best_C': best_C,
        'coefficients': {feat: float(coef) for feat, coef in coef_pairs},
    }
    with open(FEATURES_JSON, 'w') as f:
        json.dump(feat_info, f, indent=2)
    print(f'  Saved: {FEATURES_JSON}')

    # Save results text
    lines = [
        '=' * 65,
        'Structured Features + Logistic Regression — Validation Results',
        '=' * 65,
        '',
        f'LogReg: class_weight=balanced, solver=lbfgs, max_iter=1000',
        f'Best C: {best_C}',
        '',
        'C tuning results:',
    ]
    for r in results:
        lines.append(f'  C={r["C"]:<8}  AUPRC={r["AUPRC"]:.4f}  AUROC={r["AUROC"]:.4f}')
    lines += [
        '',
        f'Final validation metrics (C={best_C}):',
        f'  AUPRC:                    {auprc_final:.4f}',
        f'  AUROC:                    {auroc_final:.4f}',
        f'  Precision @ recall>=0.85: {prec_r85:.4f}  (threshold={thresh_r85:.4f})',
        '',
        'Feature coefficients (ranked by |coef|):',
    ]
    for feat, coef in coef_pairs:
        lines.append(f'  {feat:<25} {coef:+.4f}')
    lines += [
        '',
        f'Train: {len(train_feat):,} patients ({y_train.sum():,} pos, {(y_train==0).sum():,} unl)',
        f'Val:   {len(val_feat):,} patients ({y_val.sum():,} pos, {(y_val==0).sum():,} unl)',
        f'Features: {len(feature_cols)}',
    ]
    with open(RESULTS_TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved: {RESULTS_TXT}')

    print('\nDone.')


if __name__ == '__main__':
    main()
