"""
ewang163_ptsd_cohort_sets.py
============================
Rebuilds the three cohort subject_id sets from scratch using streaming reads
of the large MIMIC-IV source files, then saves one-per-line text files.

Outputs (in STUDENT_DIR):
    ewang163_ptsd_subjects.txt      — PTSD+ ICD-coded (expected n=5,711)
    ewang163_proxy_subjects.txt     — Pharmacological proxy (expected n=163)
    ewang163_unlabeled_subjects.txt — Matched unlabeled pool (expected n=17,133)

RUN:
    source ptsd_env/bin/activate
    python ewang163_ptsd_cohort_sets.py
"""

import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC = '/oscar/data/shared/ursa/mimic-iv'
OUT   = '/oscar/data/class/biol1595_2595/students/ewang163'

PATIENTS_F      = f'{MIMIC}/hosp/3.1/patients.csv'
ADMISSIONS_F    = f'{MIMIC}/hosp/3.1/admissions.csv'
DIAGNOSES_F     = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'
PRESCRIPTIONS_F = f'{MIMIC}/hosp/3.1/prescriptions.csv'

# ── ICD codes (dots stripped, uppercase) ──────────────────────────────────
PTSD_ICD10   = 'F431'
PTSD_ICD9    = '30981'

EXCL_PREFIXES = ['I10', 'N40', 'I7300', 'S06']

SSRI_SNRI_DRUGS = [
    'sertraline', 'fluoxetine', 'paroxetine', 'escitalopram', 'citalopram',
    'fluvoxamine', 'venlafaxine', 'duloxetine', 'desvenlafaxine',
    'levomilnacipran', 'milnacipran'
]


# ── Helper: age → decade string (must match table1 exactly) ──────────────
def age_decade(age):
    if age is None or np.isnan(age):
        return 'Other'
    a = int(age)
    if 20 <= a <= 29: return '20s'
    if 30 <= a <= 39: return '30s'
    if 40 <= a <= 49: return '40s'
    if 50 <= a <= 59: return '50s'
    return 'Other'


def save_subjects(filepath, subject_set):
    """Write sorted subject_ids, one per line."""
    with open(filepath, 'w') as f:
        for sid in sorted(subject_set):
            f.write(f'{sid}\n')
    print(f'  Saved {len(subject_set):,} subject_ids → {filepath}')


def main():
    print('=' * 65)
    print('PTSD NLP Project — Cohort Set Reconstruction')
    print('=' * 65)

    # ── Step 1: Stream diagnoses_icd.csv → identify PTSD+ subjects ────────
    print('\n[1/6] Streaming diagnoses_icd.csv to find PTSD+ subjects ...')
    ptsd_subjects = set()
    dx_row_count = 0

    with open(DIAGNOSES_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dx_row_count += 1
            code = row['icd_code'].strip().replace('.', '').upper()
            version = row['icd_version'].strip()
            if (version == '10' and code.startswith(PTSD_ICD10)) or \
               (version == '9' and code == PTSD_ICD9):
                ptsd_subjects.add(int(row['subject_id']))

    print(f'  Scanned {dx_row_count:,} diagnosis rows')
    print(f'  PTSD+ subjects: {len(ptsd_subjects):,}')

    # ── Step 2: Stream prescriptions.csv → collect prazosin & SSRI times ──
    print('\n[2/6] Streaming prescriptions.csv to find proxy candidates ...')
    praz_times = defaultdict(list)   # subject_id → [datetime, ...]
    ssri_times = defaultdict(list)
    rx_row_count = 0

    with open(PRESCRIPTIONS_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rx_row_count += 1
            drug = row.get('drug', '').lower()
            starttime_str = row.get('starttime', '').strip()
            if not starttime_str or not drug:
                continue

            sid = int(row['subject_id'])
            is_praz = 'prazosin' in drug
            is_ssri = any(d in drug for d in SSRI_SNRI_DRUGS)

            if is_praz or is_ssri:
                try:
                    dt = datetime.fromisoformat(starttime_str)
                except ValueError:
                    continue
                if is_praz:
                    praz_times[sid].append(dt)
                if is_ssri:
                    ssri_times[sid].append(dt)

    print(f'  Scanned {rx_row_count:,} prescription rows')
    print(f'  Patients with prazosin: {len(praz_times):,}')
    print(f'  Patients with SSRI/SNRI: {len(ssri_times):,}')

    # Cross-check: prazosin × SSRI within 180 days
    proxy_candidates = set()
    overlap_sids = set(praz_times.keys()) & set(ssri_times.keys())
    for sid in overlap_sids:
        found = False
        for pt in praz_times[sid]:
            for st in ssri_times[sid]:
                if abs((pt - st).total_seconds()) / 86400 <= 180:
                    found = True
                    break
            if found:
                break
        if found:
            proxy_candidates.add(sid)

    print(f'  Proxy candidates (prazosin + SSRI/SNRI ≤180 days): {len(proxy_candidates):,}')

    # ── Step 3: Stream diagnoses_icd.csv again → find exclusion subjects ──
    print('\n[3/6] Streaming diagnoses_icd.csv for proxy ICD exclusions ...')
    # Only need to check subjects in proxy_candidates (minus PTSD+)
    candidates_to_check = proxy_candidates - ptsd_subjects
    excl_subjects = set()

    with open(DIAGNOSES_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['subject_id'])
            if sid not in candidates_to_check:
                continue
            code = row['icd_code'].strip().replace('.', '').upper()
            if any(code.startswith(p) for p in EXCL_PREFIXES):
                excl_subjects.add(sid)

    proxy_subjects = candidates_to_check - excl_subjects
    print(f'  Excluded by ICD codes: {len(excl_subjects):,}')
    print(f'  Proxy subjects (final): {len(proxy_subjects):,}')

    # ── Step 4: Load patients + admissions for unlabeled matching ──────────
    print('\n[4/6] Loading patients.csv and admissions.csv for matching ...')
    pts = pd.read_csv(PATIENTS_F, usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    adm = pd.read_csv(ADMISSIONS_F, usecols=['subject_id', 'hadm_id', 'admittime'],
                       parse_dates=['admittime'])
    print(f'  Patients: {len(pts):,} | Admissions: {len(adm):,}')

    # ── Step 5: Build unlabeled pool and do 3:1 matching ──────────────────
    print('\n[5/6] Building unlabeled pool and 3:1 matching ...')
    all_excluded = ptsd_subjects | proxy_subjects

    # First admission per subject (for age computation and as index)
    first_adm = (adm.groupby('subject_id')['admittime']
                 .min().reset_index()
                 .rename(columns={'admittime': 'first_admittime'}))
    first_adm['first_admit_year'] = first_adm['first_admittime'].dt.year

    # Merge patient demographics
    pool = first_adm.merge(pts, on='subject_id', how='left')
    pool['age_at_admission'] = (pool['anchor_age']
                                + (pool['first_admit_year'] - pool['anchor_year'])
                                ).clip(lower=18)
    pool['decade'] = pool['age_at_admission'].apply(age_decade)

    # Split into PTSD+ reference and unlabeled candidates
    ptsd_ref = pool[pool['subject_id'].isin(ptsd_subjects)].copy()
    unlab_pool = pool[~pool['subject_id'].isin(all_excluded)].copy()

    print(f'  PTSD+ reference for matching: {len(ptsd_ref):,}')
    print(f'  Unlabeled candidate pool: {len(unlab_pool):,}')

    # 3:1 stratified match on age_decade × sex
    np.random.seed(42)
    matched_rows = []
    for (dec, sex), grp in ptsd_ref.groupby(['decade', 'gender']):
        needed = len(grp) * 3
        candidates = unlab_pool[
            (unlab_pool['decade'] == dec) &
            (unlab_pool['gender'] == sex)
        ]
        take = min(needed, len(candidates))
        if take < needed:
            print(f'  WARNING: only {len(candidates)} unlabeled in stratum '
                  f'(decade={dec}, sex={sex}); needed {needed}')
        if take > 0:
            matched_rows.append(candidates.sample(n=take, random_state=42))

    unlab_matched = pd.concat(matched_rows, ignore_index=True)
    unlab_subjects = set(unlab_matched['subject_id'].unique())
    print(f'  Matched unlabeled subjects: {len(unlab_subjects):,}')

    # ── Step 6: Save output files ─────────────────────────────────────────
    print('\n[6/6] Saving subject ID files ...')
    save_subjects(f'{OUT}/ewang163_ptsd_subjects.txt', ptsd_subjects)
    save_subjects(f'{OUT}/ewang163_proxy_subjects.txt', proxy_subjects)
    save_subjects(f'{OUT}/ewang163_unlabeled_subjects.txt', unlab_subjects)

    # ── Verify counts ─────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('VERIFICATION')
    print('=' * 65)
    expected = {'PTSD+': 5711, 'Proxy': 163, 'Unlabeled': 17133}
    actual   = {'PTSD+': len(ptsd_subjects), 'Proxy': len(proxy_subjects),
                'Unlabeled': len(unlab_subjects)}

    all_ok = True
    for group in expected:
        status = 'OK' if actual[group] == expected[group] else 'MISMATCH'
        if status == 'MISMATCH':
            all_ok = False
        print(f'  {group:12s}: expected {expected[group]:,}  got {actual[group]:,}  [{status}]')

    if all_ok:
        print('\nAll counts match. Done.')
    else:
        print('\nWARNING: Some counts do not match expected values!')


if __name__ == '__main__':
    main()
