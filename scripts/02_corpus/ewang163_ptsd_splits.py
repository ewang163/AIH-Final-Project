"""
ewang163_ptsd_splits.py
=======================
Creates patient-level splits, stratified by ptsd_label.

Two split modes:
  1. Random 80/10/10 (default) — patient-level stratified random split
  2. Temporal (--temporal) — train on pre-2015 patients, test on post-2015
     (Fix 7: temporal validation to test generalization across coding-era shift)

Splits are on subject_id — all admissions for a patient go to the same split.

Inputs:
    ewang163_ptsd_corpus.parquet
    ewang163_ptsd_adm_extract.parquet  (for temporal mode)

Outputs:
    ewang163_split_train.parquet
    ewang163_split_val.parquet
    ewang163_split_test.parquet
    ewang163_split_subject_ids.json

    (Temporal mode adds _temporal suffix to all output filenames)

RUN:
    source ptsd_env/bin/activate
    python ewang163_ptsd_splits.py               # random split
    python ewang163_ptsd_splits.py --temporal     # temporal split (Fix 7)
"""

import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR  = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_NOTES   = f'{STUDENT_DIR}/data/notes'
DATA_SPLITS  = f'{STUDENT_DIR}/data/splits'
DATA_COHORT  = f'{STUDENT_DIR}/data/cohort'

CORPUS_PARQUET = f'{DATA_NOTES}/ewang163_ptsd_corpus.parquet'
ADM_PARQUET    = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
TRAIN_PARQUET  = f'{DATA_SPLITS}/ewang163_split_train.parquet'
VAL_PARQUET    = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET   = f'{DATA_SPLITS}/ewang163_split_test.parquet'
SPLITS_JSON    = f'{DATA_SPLITS}/ewang163_split_subject_ids.json'

TEMPORAL_CUTOFF = '2015'


def temporal_split(df, adm_parquet, cutoff):
    """Fix 7: temporal split — train on pre-cutoff, test on post-cutoff.

    MIMIC-IV uses per-patient random date shifts, so shifted admittime cannot
    be used for temporal ordering.  Instead, we join patients.csv to get
    anchor_year_group (the real 3-year window) and split on that.

    cutoff='2015' means: anchor_year_group '2008-2010' and '2011-2013' → train,
    '2014-2016' onward → test.  This places the ICD-9→ICD-10 transition and
    post-DSM-5 reclassification in the test period.
    """
    import csv
    PATIENTS_F = '/oscar/data/shared/ursa/mimic-iv/hosp/3.1/patients.csv'

    # Read anchor_year_group for cohort patients
    cohort_sids = set(df['subject_id'].unique())
    sid_to_group = {}
    with open(PATIENTS_F) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['subject_id'])
            if sid in cohort_sids:
                sid_to_group[sid] = row['anchor_year_group']

    patients = df.groupby('subject_id')['ptsd_label'].first().reset_index()
    patients['anchor_year_group'] = patients['subject_id'].map(sid_to_group)

    # Pre-cutoff: anchor_year_group starts before cutoff year
    # e.g., cutoff='2015' → pre = {'2008 - 2010', '2011 - 2013'}
    cutoff_year = int(cutoff.split('-')[0])
    patients['group_start'] = patients['anchor_year_group'].apply(
        lambda g: int(g.split(' - ')[0]) if pd.notna(g) else 9999
    )

    pre = patients[patients['group_start'] < cutoff_year]
    post = patients[patients['group_start'] >= cutoff_year]

    print(f'  Temporal cutoff: anchor_year_group starting before {cutoff_year}')
    print(f'  Pre-cutoff groups: {sorted(pre["anchor_year_group"].unique())}')
    print(f'  Post-cutoff groups: {sorted(post["anchor_year_group"].unique())}')
    print(f'  Pre-cutoff patients:  {len(pre):,} '
          f'(pos={int(pre["ptsd_label"].sum()):,})')
    print(f'  Post-cutoff patients: {len(post):,} '
          f'(pos={int(post["ptsd_label"].sum()):,})')

    if len(pre) == 0 or len(post) == 0:
        raise ValueError(f'Temporal split produced empty partition. '
                         f'Pre={len(pre)}, Post={len(post)}')

    sids_train_val = pre['subject_id'].values
    labels_train_val = pre['ptsd_label'].values
    sids_train, sids_val, _, _ = train_test_split(
        sids_train_val, labels_train_val,
        test_size=0.1, stratify=labels_train_val, random_state=42
    )

    return set(sids_train), set(sids_val), set(post['subject_id'].values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal', action='store_true',
                        help='Use temporal split (Fix 7) instead of random')
    args = parser.parse_args()

    mode = 'temporal' if args.temporal else 'random'
    suffix = '_temporal' if args.temporal else ''

    train_parquet = TRAIN_PARQUET.replace('.parquet', f'{suffix}.parquet')
    val_parquet = VAL_PARQUET.replace('.parquet', f'{suffix}.parquet')
    test_parquet = TEST_PARQUET.replace('.parquet', f'{suffix}.parquet')
    splits_json = SPLITS_JSON.replace('.json', f'{suffix}.json')

    print('=' * 65)
    print(f'PTSD NLP Project — Patient-Level Splits ({mode})')
    print('=' * 65)

    # ── Step 1: Load corpus ───────────────────────────────────────────────
    print('\n[1/3] Loading corpus ...')
    df = pd.read_parquet(CORPUS_PARQUET)
    print(f'  {len(df):,} rows, {df["subject_id"].nunique():,} patients')

    # ── Step 2: Patient-level split ───────────────────────────────────────
    if args.temporal:
        print('\n[2/3] Temporal split (Fix 7) ...')
        train_set, val_set, test_set = temporal_split(
            df, ADM_PARQUET, TEMPORAL_CUTOFF)
    else:
        print('\n[2/3] Splitting patients 80/10/10 stratified by ptsd_label ...')
        patients = (df.groupby('subject_id')['ptsd_label']
                    .first()
                    .reset_index())

        sids = patients['subject_id'].values
        labels = patients['ptsd_label'].values

        sids_train, sids_temp, labels_train, labels_temp = train_test_split(
            sids, labels, test_size=0.2, stratify=labels, random_state=42
        )

        sids_val, sids_test, labels_val, labels_test = train_test_split(
            sids_temp, labels_temp, test_size=0.5, stratify=labels_temp, random_state=42
        )

        train_set = set(sids_train)
        val_set   = set(sids_val)
        test_set  = set(sids_test)

    patients = df.groupby('subject_id')['ptsd_label'].first().reset_index()

    # Verify no overlap
    assert len(train_set & val_set) == 0, 'Train/val overlap!'
    assert len(train_set & test_set) == 0, 'Train/test overlap!'
    assert len(val_set & test_set) == 0, 'Val/test overlap!'
    assert len(train_set) + len(val_set) + len(test_set) == len(patients), \
        f'Patient count mismatch: {len(train_set)}+{len(val_set)}+{len(test_set)} != {len(patients)}'
    print('  No subject_id overlap across splits — verified.')

    # Assign rows to splits
    df_train = df[df['subject_id'].isin(train_set)].copy()
    df_val   = df[df['subject_id'].isin(val_set)].copy()
    df_test  = df[df['subject_id'].isin(test_set)].copy()

    # ── Step 3: Save ──────────────────────────────────────────────────────
    print('\n[3/3] Saving ...')
    df_train.to_parquet(train_parquet, index=False)
    df_val.to_parquet(val_parquet, index=False)
    df_test.to_parquet(test_parquet, index=False)

    splits_dict = {
        'train': sorted(int(s) for s in train_set),
        'val':   sorted(int(s) for s in val_set),
        'test':  sorted(int(s) for s in test_set),
        'split_mode': mode,
    }
    if args.temporal:
        splits_dict['temporal_cutoff'] = TEMPORAL_CUTOFF
    with open(splits_json, 'w') as f:
        json.dump(splits_dict, f)

    print(f'  {train_parquet}')
    print(f'  {val_parquet}')
    print(f'  {test_parquet}')
    print(f'  {splits_json}')

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('SPLIT SUMMARY')
    print('=' * 65)
    print(f'  {"Split":<8} {"Patients":>10} {"Rows":>8} '
          f'{"Pos (label=1)":>15} {"Unlab (label=0)":>17} {"Pos %":>7}')
    print(f'  {"-"*8} {"-"*10} {"-"*8} {"-"*15} {"-"*17} {"-"*7}')

    for name, split_df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        n_patients = split_df['subject_id'].nunique()
        n_rows = len(split_df)
        n_pos = (split_df['ptsd_label'] == 1).sum()
        n_neg = (split_df['ptsd_label'] == 0).sum()
        pct = n_pos / n_rows * 100 if n_rows > 0 else 0
        print(f'  {name:<8} {n_patients:>10,} {n_rows:>8,} '
              f'{n_pos:>15,} {n_neg:>17,} {pct:>6.1f}%')

    print(f'\n  Total patients: {len(patients):,}')
    print('\nDone.')


if __name__ == '__main__':
    main()
