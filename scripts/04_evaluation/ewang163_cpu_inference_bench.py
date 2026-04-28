"""
ewang163_cpu_inference_bench.py
================================
Re-measures inference runtime for the two CPU-only models (Keyword DSM-5/PCL-5
and Structured + LogReg) on a SLURM batch partition node with realistic CPU
allocation (16 CPUs, 32 GB).

Why this script exists
----------------------
The original `ewang163_unified_inference_bench.py` ran all 5 models inside a
single L40S GPU job that allocated only 1 CPU.  For Longformer and BERT this
is fine — they run on the GPU.  For Keyword and Structured, however, the
1-CPU bottleneck made their reported timings unrepresentative of any realistic
CPU deployment.

This script reuses the same code paths (regex lexicon, structured feature
build) but parallelises the keyword regex scan across CPU cores via
joblib.Parallel.  Structured feature build is dominated by streaming
diagnoses_icd.csv + prescriptions.csv from disk and benefits primarily from
the larger memory pool and faster I/O queue available on a batch node.

Outputs
-------
  - results/metrics/ewang163_cpu_inference_bench.csv (this script's CSV)
  - results/metrics/ewang163_runtime_benchmarks.csv  (BenchmarkLogger appends)
"""

import csv
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC              = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
DATA_COHORT        = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_METRICS    = f'{STUDENT_DIR}/results/metrics'

VAL_PARQUET        = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET       = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET        = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'

KEYWORD_JSON       = f'{MODEL_DIR}/ewang163_keyword_weights.json'
STRUCT_LR_PKL      = f'{MODEL_DIR}/ewang163_structured_logreg.pkl'
STRUCT_FEAT_JSON   = f'{MODEL_DIR}/ewang163_structured_features.json'

DIAGNOSES_F        = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'
PRESCRIPTIONS_F    = f'{MIMIC}/hosp/3.1/prescriptions.csv'

OUT_CSV            = f'{RESULTS_METRICS}/ewang163_cpu_inference_bench.csv'

# ── Comorbidity / drug constants ─────────────────────────────────────────
COMORBIDITY_PFXS = {
    'dx_MDD':      ['F32', 'F33', '296'],
    'dx_anxiety':  ['F41', '300'],
    'dx_SUD':      ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17',
                    'F18', 'F19', '303', '304', '305'],
    'dx_TBI':      ['S06', '800', '801', '802', '803', '804', '850', '851',
                    '852', '853', '854'],
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


def map_race(r):
    if pd.isna(r):
        return 'Other/Unknown'
    s = str(r).upper()
    if 'WHITE' in s:
        return 'White'
    if 'BLACK' in s or 'AFRICAN' in s:
        return 'Black'
    if 'HISPANIC' in s or 'LATINO' in s:
        return 'Hispanic'
    if 'ASIAN' in s:
        return 'Asian'
    return 'Other/Unknown'


# ── Keyword scoring ───────────────────────────────────────────────────────
def load_keyword_lexicon(path):
    """Load the JSON-serialised phrase list and pre-compile each regex."""
    with open(path) as f:
        spec = json.load(f)
    return [(entry['pattern'], float(entry['weight']))
            for entry in spec['phrases']]


def _score_chunk(texts_chunk, raw_specs):
    """Worker function: compile regexes once per worker, score a chunk."""
    compiled = [(re.compile(pat, re.IGNORECASE), w) for pat, w in raw_specs]
    out = np.zeros(len(texts_chunk))
    for i, t in enumerate(texts_chunk):
        s = 0.0
        for regex, w in compiled:
            s += w * len(regex.findall(t))
        out[i] = s
    return out


def score_notes_keyword_parallel(texts, raw_specs, n_jobs):
    """Parallel weighted-phrase scoring across CPU cores via joblib."""
    if n_jobs <= 1 or len(texts) <= n_jobs:
        return _score_chunk(texts, raw_specs)
    # Split into n_jobs roughly equal chunks
    chunks = np.array_split(np.arange(len(texts)), n_jobs)
    chunk_texts = [[texts[i] for i in idx] for idx in chunks]
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_score_chunk)(ct, raw_specs) for ct in chunk_texts
    )
    return np.concatenate(results)


# ── Structured feature build ──────────────────────────────────────────────
def build_structured_features(eval_df, adm, feature_cols):
    """Reproduces the inference-time structured feature build (same logic as
    the unified bench script)."""
    eval_sids = set(eval_df['subject_id'].unique())
    eval_hadms = set(eval_df['hadm_id'].unique())

    idx_rows = adm[adm['hadm_id'].isin(eval_hadms)].copy()
    idx_rows = idx_rows.sort_values('admittime').drop_duplicates(
        'subject_id', keep='first')

    idx_rows['sex_female'] = (idx_rows['gender'] == 'F').astype(int)
    idx_rows['race_cat'] = idx_rows['race'].apply(map_race)
    idx_rows['emergency_admission'] = (
        idx_rows['admission_type'].str.upper().str.contains('EMER', na=False)
        .astype(int)
    )
    idx_rows['medicaid_selfpay'] = (
        idx_rows['insurance'].str.upper().isin(['MEDICAID', 'SELF PAY'])
        .astype(int)
    )
    idx_rows['los_days'] = (
        (idx_rows['dischtime'] - idx_rows['admittime']).dt.total_seconds()
        / 86400.0
    )

    race_dummies = pd.get_dummies(idx_rows['race_cat'], prefix='race')
    idx_rows = pd.concat([idx_rows, race_dummies], axis=1)
    idx_rows = idx_rows.drop(columns=['race_cat'])

    adm_cohort = adm[adm['subject_id'].isin(eval_sids)].copy()
    index_adm = idx_rows[['subject_id', 'admittime']].rename(
        columns={'admittime': 'idx_admittime'})
    prior_adm = adm_cohort.merge(index_adm, on='subject_id', how='inner')
    prior_adm = prior_adm[prior_adm['admittime'] < prior_adm['idx_admittime']].copy()

    prior_counts = prior_adm.groupby('subject_id').size().reset_index(
        name='n_prior_admissions')
    idx_rows = idx_rows.merge(prior_counts, on='subject_id', how='left')
    idx_rows['n_prior_admissions'] = idx_rows['n_prior_admissions'].fillna(0).astype(int)

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

    for col in feature_cols:
        if col not in idx_rows.columns:
            idx_rows[col] = 0

    idx_keyed = idx_rows.set_index('subject_id')
    X = np.zeros((len(eval_df), len(feature_cols)), dtype=np.float32)
    for i, sid in enumerate(eval_df['subject_id'].values):
        if sid in idx_keyed.index:
            row = idx_keyed.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for j, col in enumerate(feature_cols):
                v = row.get(col, 0)
                X[i, j] = float(v) if not pd.isna(v) else 0.0
    return X


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('PTSD NLP — CPU INFERENCE BENCHMARK (Keyword + Structured)')
    print('=' * 70, flush=True)

    bench = BenchmarkLogger()

    # Determine CPU allocation from SLURM, fall back to os.cpu_count()
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK',
                                 os.environ.get('SLURM_CPUS_ON_NODE',
                                                os.cpu_count() or 1)))
    slurm_job = os.environ.get('SLURM_JOB_ID', 'local')
    slurm_node = os.environ.get('SLURMD_NODENAME',
                                 os.environ.get('SLURM_NODELIST', 'unknown'))
    print(f'SLURM_JOB_ID={slurm_job}  node={slurm_node}  n_cpus={n_cpus}')

    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    val_texts = val_df['note_text'].tolist()
    test_texts = test_df['note_text'].tolist()
    print(f'Val: {len(val_df):,} | Test: {len(test_df):,}')

    rows = []

    def time_block(label, fn, n_samples, stage_tag):
        t0 = time.perf_counter()
        with bench.track('cpu_bench', stage=stage_tag, device='cpu',
                         n_samples=n_samples,
                         notes=f'{label} | n_cpus={n_cpus} | node={slurm_node}'):
            fn()
        dt = time.perf_counter() - t0
        ms_per = round(dt * 1000.0 / max(n_samples, 1), 2)
        rows.append({
            'model':          label,
            'stage':          stage_tag,
            'wall_clock_s':   round(dt, 3),
            'n_samples':      n_samples,
            'ms_per_patient': ms_per,
            'n_cpus':         n_cpus,
            'slurm_node':     slurm_node,
            'slurm_job':      slurm_job,
        })
        print(f'  -> {label}/{stage_tag}: {dt:.2f}s '
              f'({ms_per} ms/patient, n_cpus={n_cpus})')

    # ── 1. Keyword DSM-5 / PCL-5 (parallel) ──────────────────────────────
    print('\n[1/2] Keyword DSM-5 / PCL-5 (parallel) ...')
    raw_specs = load_keyword_lexicon(KEYWORD_JSON)
    print(f'  Loaded {len(raw_specs)} phrases. n_jobs={n_cpus}')

    # Warm-up: spin up joblib workers once so the first measured call isn't
    # paying process-startup cost.
    _ = score_notes_keyword_parallel(val_texts[:32], raw_specs, n_cpus)

    time_block(
        'Keyword-DSM5/PCL5', lambda:
        score_notes_keyword_parallel(val_texts, raw_specs, n_cpus),
        len(val_df), 'val_inference')
    time_block(
        'Keyword-DSM5/PCL5', lambda:
        score_notes_keyword_parallel(test_texts, raw_specs, n_cpus),
        len(test_df), 'test_inference')

    # ── 2. Structured + LogReg ──────────────────────────────────────────
    print('\n[2/2] Structured + LogReg ...')
    with open(STRUCT_FEAT_JSON) as f:
        feat_info = json.load(f)
    feature_cols = feat_info['feature_columns']
    with open(STRUCT_LR_PKL, 'rb') as f:
        struct_model = pickle.load(f)
    adm = pd.read_parquet(ADM_PARQUET)

    def _struct_val():
        X = build_structured_features(val_df, adm, feature_cols)
        struct_model.predict_proba(X)[:, 1]

    def _struct_test():
        X = build_structured_features(test_df, adm, feature_cols)
        struct_model.predict_proba(X)[:, 1]

    time_block('Structured-LogReg', _struct_val,
               len(val_df), 'val_inference')
    time_block('Structured-LogReg', _struct_test,
               len(test_df), 'test_inference')

    # ── Write CSV ────────────────────────────────────────────────────────
    print('\n=== WRITING RESULTS ===')
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ['model', 'stage', 'wall_clock_s', 'n_samples',
                  'ms_per_patient', 'n_cpus', 'slurm_node', 'slurm_job']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote: {OUT_CSV}')

    print('\n=== FINAL TABLE ===')
    hdr = (f'{"model":<22} {"stage":<16} {"wall(s)":>10} '
           f'{"n":>6} {"ms/pt":>10} {"cpus":>5}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["model"]:<22} {r["stage"]:<16} '
              f'{r["wall_clock_s"]:>10.2f} {r["n_samples"]:>6} '
              f'{r["ms_per_patient"]:>10.2f} {r["n_cpus"]:>5}')
    print(f'\nNode: {slurm_node}  Job: {slurm_job}  CPUs: {n_cpus}')
    print('Done.')


if __name__ == '__main__':
    main()
