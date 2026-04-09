"""
ewang163_ptsd_splits.py
=======================
Creates patient-level 80/10/10 train/val/test splits, stratified by ptsd_label.

Splits are on subject_id — all admissions for a patient go to the same split.

Inputs:
    ewang163_ptsd_corpus.parquet

Outputs:
    ewang163_split_train.parquet
    ewang163_split_val.parquet
    ewang163_split_test.parquet
    ewang163_split_subject_ids.json

RUN:
    source ptsd_env/bin/activate
    python ewang163_ptsd_splits.py
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────
OUT = '/oscar/data/class/biol1595_2595/students/ewang163'

CORPUS_PARQUET = f'{OUT}/ewang163_ptsd_corpus.parquet'
TRAIN_PARQUET  = f'{OUT}/ewang163_split_train.parquet'
VAL_PARQUET    = f'{OUT}/ewang163_split_val.parquet'
TEST_PARQUET   = f'{OUT}/ewang163_split_test.parquet'
SPLITS_JSON    = f'{OUT}/ewang163_split_subject_ids.json'


def main():
    print('=' * 65)
    print('PTSD NLP Project — Patient-Level Splits')
    print('=' * 65)

    # ── Step 1: Load corpus ───────────────────────────────────────────────
    print('\n[1/3] Loading corpus ...')
    df = pd.read_parquet(CORPUS_PARQUET)
    print(f'  {len(df):,} rows, {df["subject_id"].nunique():,} patients')

    # ── Step 2: Patient-level split ───────────────────────────────────────
    print('\n[2/3] Splitting patients 80/10/10 stratified by ptsd_label ...')

    # One row per patient with their label
    patients = (df.groupby('subject_id')['ptsd_label']
                .first()
                .reset_index())

    sids = patients['subject_id'].values
    labels = patients['ptsd_label'].values

    # 80% train, 20% temp
    sids_train, sids_temp, labels_train, labels_temp = train_test_split(
        sids, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Split temp 50/50 → 10% val, 10% test
    sids_val, sids_test, labels_val, labels_test = train_test_split(
        sids_temp, labels_temp, test_size=0.5, stratify=labels_temp, random_state=42
    )

    train_set = set(sids_train)
    val_set   = set(sids_val)
    test_set  = set(sids_test)

    # Verify no overlap
    assert len(train_set & val_set) == 0, 'Train/val overlap!'
    assert len(train_set & test_set) == 0, 'Train/test overlap!'
    assert len(val_set & test_set) == 0, 'Val/test overlap!'
    assert len(train_set) + len(val_set) + len(test_set) == len(patients), 'Patient count mismatch!'
    print('  No subject_id overlap across splits — verified.')

    # Assign rows to splits
    df_train = df[df['subject_id'].isin(train_set)].copy()
    df_val   = df[df['subject_id'].isin(val_set)].copy()
    df_test  = df[df['subject_id'].isin(test_set)].copy()

    # ── Step 3: Save ──────────────────────────────────────────────────────
    print('\n[3/3] Saving ...')
    df_train.to_parquet(TRAIN_PARQUET, index=False)
    df_val.to_parquet(VAL_PARQUET, index=False)
    df_test.to_parquet(TEST_PARQUET, index=False)

    splits_dict = {
        'train': sorted(int(s) for s in train_set),
        'val':   sorted(int(s) for s in val_set),
        'test':  sorted(int(s) for s in test_set),
    }
    with open(SPLITS_JSON, 'w') as f:
        json.dump(splits_dict, f)

    print(f'  {TRAIN_PARQUET}')
    print(f'  {VAL_PARQUET}')
    print(f'  {TEST_PARQUET}')
    print(f'  {SPLITS_JSON}')

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
