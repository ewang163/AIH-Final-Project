"""
ewang163_ptsd_admissions_extract.py
===================================
Extracts admission rows for all three cohort groups and adds computed columns
(age_at_admission, group label, index admission flag).

Inputs (from ewang163_ptsd_cohort_sets.py):
    ewang163_ptsd_subjects.txt
    ewang163_proxy_subjects.txt
    ewang163_unlabeled_subjects.txt

Outputs:
    ewang163_ptsd_adm_extract.csv      — filtered admissions, one row per admission
    ewang163_ptsd_adm_extract.parquet  — same data with computed columns + group + index flag

RUN:
    source ptsd_env/bin/activate
    python ewang163_ptsd_admissions_extract.py
"""

import csv
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC = '/oscar/data/shared/ursa/mimic-iv'
OUT   = '/oscar/data/class/biol1595_2595/students/ewang163'

PATIENTS_F      = f'{MIMIC}/hosp/3.1/patients.csv'
ADMISSIONS_F    = f'{MIMIC}/hosp/3.1/admissions.csv'
DIAGNOSES_F     = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'

PTSD_SIDS_F   = f'{OUT}/ewang163_ptsd_subjects.txt'
PROXY_SIDS_F   = f'{OUT}/ewang163_proxy_subjects.txt'
UNLAB_SIDS_F   = f'{OUT}/ewang163_unlabeled_subjects.txt'

OUT_CSV     = f'{OUT}/ewang163_ptsd_adm_extract.csv'
OUT_PARQUET = f'{OUT}/ewang163_ptsd_adm_extract.parquet'

# ── ICD codes ─────────────────────────────────────────────────────────────
PTSD_ICD10 = 'F431'
PTSD_ICD9  = '30981'


def load_subject_set(filepath):
    """Load subject_ids from a one-per-line text file."""
    with open(filepath) as f:
        return {int(line.strip()) for line in f if line.strip()}


def main():
    print('=' * 65)
    print('PTSD NLP Project — Admissions Extract')
    print('=' * 65)

    # ── Step 1: Load cohort subject_id sets ───────────────────────────────
    print('\n[1/5] Loading cohort subject_id sets ...')
    ptsd_sids  = load_subject_set(PTSD_SIDS_F)
    proxy_sids = load_subject_set(PROXY_SIDS_F)
    unlab_sids = load_subject_set(UNLAB_SIDS_F)
    all_sids   = ptsd_sids | proxy_sids | unlab_sids

    print(f'  PTSD+: {len(ptsd_sids):,} | Proxy: {len(proxy_sids):,} | '
          f'Unlabeled: {len(unlab_sids):,} | Total unique: {len(all_sids):,}')

    # ── Step 2: Load patients.csv into lookup dict ────────────────────────
    print('\n[2/5] Loading patients.csv for demographics ...')
    pts_lookup = {}  # subject_id → (gender, anchor_age, anchor_year)
    with open(PATIENTS_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['subject_id'])
            if sid in all_sids:
                pts_lookup[sid] = (
                    row['gender'],
                    int(row['anchor_age']),
                    int(row['anchor_year']),
                )
    print(f'  Loaded demographics for {len(pts_lookup):,} cohort patients')

    # ── Step 3: Stream diagnoses_icd.csv → PTSD-coded (subject_id, hadm_id) pairs
    print('\n[3/5] Streaming diagnoses_icd.csv to find PTSD-coded admissions ...')
    ptsd_coded_hadms = set()  # (subject_id, hadm_id) pairs with PTSD code
    dx_count = 0

    with open(DIAGNOSES_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dx_count += 1
            sid = int(row['subject_id'])
            if sid not in ptsd_sids:
                continue
            code = row['icd_code'].strip().replace('.', '').upper()
            version = row['icd_version'].strip()
            if (version == '10' and code.startswith(PTSD_ICD10)) or \
               (version == '9' and code == PTSD_ICD9):
                ptsd_coded_hadms.add((sid, int(row['hadm_id'])))

    print(f'  Scanned {dx_count:,} diagnosis rows')
    print(f'  PTSD-coded (subject, admission) pairs: {len(ptsd_coded_hadms):,}')

    # ── Step 4: Stream admissions.csv → extract cohort rows ───────────────
    print('\n[4/5] Streaming admissions.csv to extract cohort admissions ...')
    adm_header = None
    extracted_rows = []
    total_rows = 0
    matched_rows = 0

    with open(ADMISSIONS_F, newline='') as f:
        reader = csv.DictReader(f)
        adm_header = reader.fieldnames
        for row in reader:
            total_rows += 1
            sid = int(row['subject_id'])
            if sid not in all_sids:
                continue
            matched_rows += 1
            # Convert types for key fields
            row['subject_id'] = sid
            row['hadm_id'] = int(row['hadm_id'])
            extracted_rows.append(row)

    print(f'  Scanned {total_rows:,} admission rows')
    print(f'  Extracted {matched_rows:,} rows for cohort patients')

    # ── Step 5: Build DataFrame, compute columns, save ────────────────────
    print('\n[5/5] Computing derived columns and saving ...')
    df = pd.DataFrame(extracted_rows)

    # Parse datetime columns
    for col in ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Assign group label
    df['group'] = 'unlabeled'
    df.loc[df['subject_id'].isin(ptsd_sids), 'group'] = 'ptsd_pos'
    df.loc[df['subject_id'].isin(proxy_sids), 'group'] = 'proxy'

    # Add patient demographics
    df['gender'] = df['subject_id'].map(lambda s: pts_lookup.get(s, (None,None,None))[0])
    df['anchor_age'] = df['subject_id'].map(lambda s: pts_lookup.get(s, (None,None,None))[1])
    df['anchor_year'] = df['subject_id'].map(lambda s: pts_lookup.get(s, (None,None,None))[2])

    # Compute age_at_admission = anchor_age + (admit_year - anchor_year), clipped >=18
    df['admit_year'] = df['admittime'].dt.year
    df['age_at_admission'] = (df['anchor_age'] + (df['admit_year'] - df['anchor_year'])).clip(lower=18)

    # Determine index_admittime per patient
    # PTSD+: first admission where a PTSD ICD code appears
    ptsd_coded_hadm_set = ptsd_coded_hadms  # set of (subject_id, hadm_id)
    df['is_ptsd_coded_adm'] = df.apply(
        lambda r: (r['subject_id'], r['hadm_id']) in ptsd_coded_hadm_set, axis=1
    )

    # For PTSD+ patients: index = earliest admittime among PTSD-coded admissions
    ptsd_index_times = (
        df[df['is_ptsd_coded_adm']]
        .groupby('subject_id')['admittime']
        .min()
        .reset_index()
        .rename(columns={'admittime': 'index_admittime'})
    )

    # For proxy and unlabeled: index = earliest admittime overall
    non_ptsd_mask = df['group'].isin(['proxy', 'unlabeled'])
    non_ptsd_index_times = (
        df[non_ptsd_mask]
        .groupby('subject_id')['admittime']
        .min()
        .reset_index()
        .rename(columns={'admittime': 'index_admittime'})
    )

    # Combine index times
    index_times = pd.concat([ptsd_index_times, non_ptsd_index_times], ignore_index=True)
    df = df.merge(index_times, on='subject_id', how='left')

    # Flag whether each row IS the index admission
    df['is_index_admission'] = (df['admittime'] == df['index_admittime'])

    # Drop helper column
    df.drop(columns=['is_ptsd_coded_adm'], inplace=True)

    # Save CSV (human-readable)
    df.to_csv(OUT_CSV, index=False)
    print(f'  Saved CSV → {OUT_CSV}')

    # Save parquet (efficient for downstream)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f'  Saved parquet → {OUT_PARQUET}')

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('SUMMARY')
    print('=' * 65)
    print(f'  Total rows (admissions): {len(df):,}')
    for grp in ['ptsd_pos', 'proxy', 'unlabeled']:
        grp_df = df[df['group'] == grp]
        n_patients = grp_df['subject_id'].nunique()
        n_admissions = len(grp_df)
        n_index = grp_df['is_index_admission'].sum()
        print(f'  {grp:12s}: {n_patients:,} patients, {n_admissions:,} admissions, '
              f'{n_index:,} index admissions')

    print(f'\n  Columns: {list(df.columns)}')
    print('\nDone.')


if __name__ == '__main__':
    main()
