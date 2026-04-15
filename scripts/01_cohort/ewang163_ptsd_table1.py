"""
ewang163_ptsd_table1.py
=======================
Computes all values for Table 1 (Study Population Characteristics) in the
PTSD NLP project proposal (AIH 2025 / Spring 2026, Assignment 2, Part 2 Q3).

Three groups:
  - ICD-coded PTSD+    : F43.1 / 309.81 at any admission  → training positives
  - Pharmacological Proxy : prazosin + SSRI/SNRI within 180 days, no exclusion ICD
                            at ANY position → external validation set
  - Matched Unlabeled     : 3:1 match on age decade × sex → PU learning pool

SAVE THIS SCRIPT TO:
    /oscar/data/class/biol1595_2595/students/ewang163/ewang163_ptsd_table1.py

SETUP (first time only — run from your Oscar terminal):
    $ module load python
    $ cd /oscar/data/class/biol1595_2595/students/ewang163
    $ python -m venv ptsd_env
    $ source ptsd_env/bin/activate
    $ pip install pandas numpy

RUN:
    $ source ptsd_env/bin/activate   # skip if already active
    $ python ewang163_ptsd_table1.py

EXPECTED RUNTIME: 5–15 min (loading prescriptions.csv is the slow step)

OUTPUTS (saved to your student directory):
    ewang163_table1_results.csv    — full Table 1 as CSV (import into Excel etc.)
    ewang163_table1_summary.txt    — human-readable version for quick review

NOTE: All output stays within your Oscar student directory per course guidelines.
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────
MIMIC = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
RESULTS_TABLE1  = f'{STUDENT_DIR}/results/table1'

PATIENTS_F      = f'{MIMIC}/hosp/3.1/patients.csv'
ADMISSIONS_F    = f'{MIMIC}/hosp/3.1/admissions.csv'
DIAGNOSES_F     = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'
PRESCRIPTIONS_F = f'{MIMIC}/hosp/3.1/prescriptions.csv'

OUT_CSV = f'{RESULTS_TABLE1}/ewang163_table1_results.csv'
OUT_TXT = f'{RESULTS_TABLE1}/ewang163_table1_summary.txt'

# ── ICD codes (dots stripped, uppercase) ──────────────────────────────────
PTSD_ICD10   = 'F431'    # F43.1
PTSD_ICD9    = '30981'   # 309.81

# Proxy exclusion codes — any diagnosis position, not just primary
EXCL_PREFIXES = ['I10', 'N40', 'I7300', 'S06']  # HTN, BPH, Raynaud's, TBI

# Comorbidity prefixes — ICD-10 and ICD-9 equivalents
# MIMIC-IV uses ICD-9 for admissions before Oct 2015, ICD-10 after
MDD_PFX      = ['F32', 'F33',                              # ICD-10
                 '296']                                     # ICD-9 (296.xx mood disorders)
ANXIETY_PFX  = ['F41',                                     # ICD-10
                 '300']                                     # ICD-9 (300.xx anxiety/neurotic)
SUD_PFX      = ['F10', 'F11', 'F12', 'F13', 'F14',       # ICD-10
                 'F15', 'F16', 'F17', 'F18', 'F19',
                 '303', '304', '305']                       # ICD-9 (alcohol/drug dependence/abuse)
TBI_PFX      = ['S06',                                     # ICD-10
                 '800', '801', '802', '803', '804',        # ICD-9 skull fractures
                 '850', '851', '852', '853', '854']         # ICD-9 intracranial injury
PAIN_PFX     = ['G89',                                     # ICD-10
                 '338']                                     # ICD-9 (338.xx pain, not elsewhere classified)
# Suicidal: ICD-10 R45.851 (suicidal ideation); ICD-9 V62.84 (ideation), E950-E959 (attempts)
SUICIDAL_PFXS = ['R458',                                   # ICD-10 R45.8x
                  'V6284',                                  # ICD-9 V62.84 suicidal ideation
                  'E95']                                    # ICD-9 E950-E959 self-inflicted injury

# ── Drug name patterns (lowercase substring match on drug column) ──────────
SSRI_SNRI_DRUGS = [
    'sertraline', 'fluoxetine', 'paroxetine', 'escitalopram', 'citalopram',
    'fluvoxamine', 'venlafaxine', 'duloxetine', 'desvenlafaxine',
    'levomilnacipran', 'milnacipran'
]
PRAZOSIN_DRUGS = ['prazosin']
SGA_DRUGS = [
    'quetiapine', 'olanzapine', 'risperidone', 'aripiprazole', 'ziprasidone',
    'clozapine', 'lurasidone', 'asenapine', 'paliperidone', 'iloperidone',
    'brexpiprazole', 'cariprazine'
]


# ── Helper: map MIMIC race strings to 5 categories ────────────────────────
def map_race(race_str):
    if pd.isna(race_str):
        return 'Other/Unknown'
    r = str(race_str).upper()
    if 'WHITE' in r:
        return 'White/Non-Hispanic'
    elif 'BLACK' in r or 'AFRICAN' in r:
        return 'Black/African American'
    elif 'HISPANIC' in r or 'LATINO' in r:
        return 'Hispanic/Latino'
    elif 'ASIAN' in r:
        return 'Asian'
    else:
        return 'Other/Unknown'


# ── Helper: age → decade string ───────────────────────────────────────────
def age_decade(age):
    if pd.isna(age):
        return 'Other'
    a = int(age)
    if 20 <= a <= 29: return '20s'
    if 30 <= a <= 39: return '30s'
    if 40 <= a <= 49: return '40s'
    if 50 <= a <= 59: return '50s'
    return 'Other'


# ── Helper: which subject_ids have an ICD prefix match ────────────────────
def subjects_with_icd_prefix(dx_df, subject_pool, prefixes):
    if dx_df.empty or 'icd_code' not in dx_df.columns or 'subject_id' not in dx_df.columns:
        return set()
    mask = dx_df['icd_code'].apply(
        lambda c: any(str(c).upper().startswith(p) for p in prefixes)
    )
    return set(dx_df[mask]['subject_id']) & set(subject_pool)


def subjects_with_icd_exact(dx_df, subject_pool, code):
    if dx_df.empty or 'icd_code' not in dx_df.columns or 'subject_id' not in dx_df.columns:
        return set()
    return set(dx_df[dx_df['icd_code'] == code]['subject_id']) & set(subject_pool)


# ── Helper: which subject_ids have a drug pattern match ───────────────────
def subjects_with_drug(rx_df, subject_pool, drug_patterns):
    if rx_df.empty or 'drug_lower' not in rx_df.columns or 'subject_id' not in rx_df.columns:
        return set()
    mask = rx_df['drug_lower'].apply(
        lambda d: any(p in str(d) for p in drug_patterns)
    )
    return set(rx_df[mask]['subject_id']) & set(subject_pool)


# ── Core stats function ────────────────────────────────────────────────────
def compute_group_stats(label, index_adm_df, adm_counts_df,
                        prior_dx_df, prior_rx_df,
                        predx_n=None):
    """
    Computes all Table 1 statistics for one group.

    index_adm_df  : one row per patient, their 'index' admission
                    (must have: subject_id, gender, age_at_admission,
                     race, admittime, dischtime, admission_type, insurance)
    adm_counts_df : subject_id → total_admissions (global, not group-specific)
    prior_dx_df   : diagnoses from admissions BEFORE the index admission
    prior_rx_df   : prescriptions from admissions BEFORE the index admission
    predx_n       : number of PTSD+ patients with pre-diagnosis notes (optional)
    """
    df = index_adm_df.copy()
    n = len(df)
    sids = set(df['subject_id'].unique())
    s = {'Group': label, 'N': n}

    def pct(count): return round(count / n * 100, 1) if n > 0 else 0.0

    # --- Sex ---
    sex_counts = df['gender'].value_counts()
    s['Female_n']   = int(sex_counts.get('F', 0))
    s['Female_pct'] = pct(s['Female_n'])
    s['Male_n']     = int(sex_counts.get('M', 0))
    s['Male_pct']   = pct(s['Male_n'])
    s['OtherSex_n']   = int(n - s['Female_n'] - s['Male_n'])
    s['OtherSex_pct'] = pct(s['OtherSex_n'])

    # --- Age at index admission ---
    ages = df['age_at_admission'].dropna()
    s['Age_mean'] = round(float(ages.mean()), 1)
    s['Age_sd']   = round(float(ages.std()), 1)

    df['decade'] = df['age_at_admission'].apply(age_decade)
    for dec in ['20s', '30s', '40s', '50s', 'Other']:
        c = int((df['decade'] == dec).sum())
        s[f'Age_{dec}_n']   = c
        s[f'Age_{dec}_pct'] = pct(c)

    # --- Race/ethnicity ---
    df['race_cat'] = df['race'].apply(map_race)
    for cat in ['White/Non-Hispanic', 'Black/African American',
                'Hispanic/Latino', 'Asian', 'Other/Unknown']:
        key = cat.replace('/', '_').replace(' ', '_').replace('-', '_')
        c = int((df['race_cat'] == cat).sum())
        s[f'Race_{key}_n']   = c
        s[f'Race_{key}_pct'] = pct(c)

    # --- Admissions count ---
    df_counts = df.merge(adm_counts_df, on='subject_id', how='left')
    total_adm = df_counts['total_admissions'].dropna()

    s['Single_admit_n']   = int((total_adm == 1).sum())
    s['Single_admit_pct'] = pct(s['Single_admit_n'])
    s['Multi_admit_n']    = int((total_adm > 1).sum())
    s['Multi_admit_pct']  = pct(s['Multi_admit_n'])

    s['Admissions_median'] = round(float(total_adm.median()), 1)
    s['Admissions_Q1']     = round(float(total_adm.quantile(0.25)), 1)
    s['Admissions_Q3']     = round(float(total_adm.quantile(0.75)), 1)

    # --- Pre-diagnosis notes available (PTSD+ group only) ---
    if predx_n is not None:
        s['PreDx_n']   = predx_n
        s['PreDx_pct'] = pct(predx_n)
    else:
        s['PreDx_n']   = 'N/A'
        s['PreDx_pct'] = 'N/A'

    # --- LOS at index admission ---
    df['los_days'] = ((df['dischtime'] - df['admittime'])
                      .dt.total_seconds() / 86400)
    los = df['los_days'].dropna()
    s['LOS_median'] = round(float(los.median()), 1)
    s['LOS_Q1']     = round(float(los.quantile(0.25)), 1)
    s['LOS_Q3']     = round(float(los.quantile(0.75)), 1)

    # --- Emergency admission ---
    # MIMIC-IV v3.1 uses 'EW EMER.' and 'DIRECT EMER.' — not 'EMERGENCY'
    emerg = df['admission_type'].str.upper().str.contains('EMER', na=False)
    s['Emergency_n']   = int(emerg.sum())
    s['Emergency_pct'] = pct(s['Emergency_n'])

    # --- Insurance: Medicaid or self-pay ---
    med_sp = df['insurance'].str.upper().isin(
        ['MEDICAID', 'SELF PAY', 'SELF-PAY']
    )
    s['MedicaidSelfPay_n']   = int(med_sp.sum())
    s['MedicaidSelfPay_pct'] = pct(s['MedicaidSelfPay_n'])

    # --- Prior-admission comorbidities ---
    comorbidities = [
        ('MDD',         MDD_PFX),
        ('Anxiety',     ANXIETY_PFX),
        ('SUD',         SUD_PFX),
        ('TBI',         TBI_PFX),
        ('ChronicPain', PAIN_PFX),
        ('Suicidal',    SUICIDAL_PFXS),
    ]
    for name, codes in comorbidities:
        flagged = subjects_with_icd_prefix(prior_dx_df, sids, codes)
        c = len(flagged)
        s[f'Dx_{name}_n']   = c
        s[f'Dx_{name}_pct'] = pct(c)

    # --- Prior-admission medications ---
    for name, patterns in [
        ('SSRI_SNRI', SSRI_SNRI_DRUGS),
        ('Prazosin',  PRAZOSIN_DRUGS),
        ('SGA',       SGA_DRUGS),
    ]:
        flagged = subjects_with_drug(prior_rx_df, sids, patterns)
        c = len(flagged)
        s[f'Rx_{name}_n']   = c
        s[f'Rx_{name}_pct'] = pct(c)

    return s


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('PTSD NLP Project — Table 1 Computation')
    print('=' * 65)

    # ── Load tables ─────────────────────────────────────────────
    print('\n[1/7] Loading patients.csv ...')
    pts = pd.read_csv(PATIENTS_F)
    print(f'      {len(pts):,} rows | columns: {list(pts.columns)}')

    print('[2/7] Loading admissions.csv ...')
    adm = pd.read_csv(ADMISSIONS_F, parse_dates=['admittime', 'dischtime'])
    print(f'      {len(adm):,} rows | columns: {list(adm.columns)}')

    print('[3/7] Loading diagnoses_icd.csv ...')
    dx = pd.read_csv(DIAGNOSES_F, dtype={'icd_code': str, 'icd_version': str})
    print(f'      {len(dx):,} rows | columns: {list(dx.columns)}')

    print('[4/7] Loading prescriptions.csv (may take several minutes) ...')
    rx = pd.read_csv(PRESCRIPTIONS_F,
                     parse_dates=['starttime', 'stoptime'],
                     low_memory=False)
    print(f'      {len(rx):,} rows | columns: {list(rx.columns)}')

    # ── Normalize ────────────────────────────────────────────────
    dx['icd_code'] = (dx['icd_code']
                      .str.strip()
                      .str.replace('.', '', regex=False)
                      .str.upper())
    rx['drug_lower'] = rx['drug'].str.lower().fillna('')

    # Print unique race values to verify mapping
    print(f'\n  Unique race values in admissions: {sorted(adm["race"].dropna().unique()[:20].tolist())}')
    print(f'  Unique insurance values: {sorted(adm["insurance"].dropna().unique().tolist())}')
    print(f'  Unique admission_type values: {sorted(adm["admission_type"].dropna().unique()[:10].tolist())}')
    print(f'  Unique gender values: {adm.merge(pts[["subject_id","gender"]], on="subject_id")["gender"].unique().tolist()}')

    # ── Attach patient info to admissions ─────────────────────────
    adm = adm.merge(pts[['subject_id', 'gender', 'anchor_age', 'anchor_year']],
                    on='subject_id', how='left')
    adm['admit_year']        = adm['admittime'].dt.year
    adm['age_at_admission']  = (adm['anchor_age']
                                 + (adm['admit_year'] - adm['anchor_year'])
                                 ).clip(lower=18)

    # Total admissions per patient (used for all groups)
    adm_counts = (adm.groupby('subject_id')['hadm_id']
                  .nunique()
                  .reset_index()
                  .rename(columns={'hadm_id': 'total_admissions'}))

    # ── Step 1: Identify PTSD-coded patients ────────────────────
    print('\n[5/7] Building cohort groups ...')

    ptsd_mask = (
        ((dx['icd_version'] == '10') & (dx['icd_code'].str.startswith(PTSD_ICD10))) |
        ((dx['icd_version'] == '9')  & (dx['icd_code'] == PTSD_ICD9))
    )
    ptsd_dx_rows = dx[ptsd_mask][['subject_id', 'hadm_id']].drop_duplicates()
    ptsd_subjects = set(ptsd_dx_rows['subject_id'].unique())
    print(f'  PTSD-coded patients (F43.1 / 309.81): {len(ptsd_subjects):,}')

    # First PTSD-coded admission per patient
    ptsd_first = (ptsd_dx_rows
                  .merge(adm[['subject_id', 'hadm_id', 'admittime']],
                         on=['subject_id', 'hadm_id'], how='left')
                  .groupby('subject_id')['admittime']
                  .min()
                  .reset_index()
                  .rename(columns={'admittime': 'first_ptsd_admittime'}))

    # PTSD index admission = first coded admission
    ptsd_adm = adm[adm['subject_id'].isin(ptsd_subjects)].copy()
    ptsd_adm = ptsd_adm.merge(ptsd_first, on='subject_id', how='left')
    ptsd_index = (ptsd_adm[ptsd_adm['admittime'] == ptsd_adm['first_ptsd_admittime']]
                  .drop_duplicates(subset='subject_id')
                  .copy())

    # Pre-diagnosis admissions (multi-admission patients only)
    ptsd_adm_counts = ptsd_adm.merge(adm_counts, on='subject_id', how='left')
    pre_dx_adm = ptsd_adm_counts[
        (ptsd_adm_counts['total_admissions'] > 1) &
        (ptsd_adm_counts['admittime'] < ptsd_adm_counts['first_ptsd_admittime'])
    ]
    predx_subjects = set(pre_dx_adm['subject_id'].unique())
    print(f'  PTSD+ with pre-diagnosis admissions available: {len(predx_subjects):,}')
    print(f'  PTSD+ single-admission (no pre-dx notes): {len(ptsd_subjects) - len(predx_subjects):,}')

    # ── Step 2: Pharmacological proxy group ─────────────────────
    # Find patients with prazosin AND SSRI/SNRI within 180 days of each other
    praz = (rx[rx['drug_lower'].str.contains('prazosin', na=False)]
            [['subject_id', 'starttime']]
            .dropna(subset=['starttime'])
            .rename(columns={'starttime': 'praz_start'}))

    ssri_mask = rx['drug_lower'].apply(
        lambda d: any(p in str(d) for p in SSRI_SNRI_DRUGS)
    )
    ssri = (rx[ssri_mask][['subject_id', 'starttime']]
            .dropna(subset=['starttime'])
            .rename(columns={'starttime': 'ssri_start'}))

    # Cross-join per patient, filter to within 180 days
    praz_ssri = praz.merge(ssri, on='subject_id', how='inner')
    praz_ssri['day_diff'] = (
        (praz_ssri['praz_start'] - praz_ssri['ssri_start'])
        .dt.total_seconds().abs() / 86400
    )
    proxy_candidates = set(
        praz_ssri[praz_ssri['day_diff'] <= 180]['subject_id'].unique()
    )
    print(f'  Proxy candidates (prazosin + SSRI/SNRI ≤180 days): {len(proxy_candidates):,}')

    # Exclude patients with exclusion ICD codes at ANY diagnosis position
    excl_mask = dx['icd_code'].apply(
        lambda c: any(str(c).upper().startswith(p) for p in EXCL_PREFIXES)
    )
    excl_subjects = set(dx[excl_mask]['subject_id'].unique())

    proxy_subjects = proxy_candidates - ptsd_subjects - excl_subjects
    print(f'  Proxy patients (after ICD exclusions, not PTSD-coded): {len(proxy_subjects):,}')

    # Proxy index admission = first admission overall
    proxy_adm = adm[adm['subject_id'].isin(proxy_subjects)].copy()
    proxy_first_time = (proxy_adm.groupby('subject_id')['admittime']
                        .min().reset_index()
                        .rename(columns={'admittime': 'first_admittime'}))
    proxy_index = (proxy_adm
                   .merge(proxy_first_time, on='subject_id')
                   .query('admittime == first_admittime')
                   .drop_duplicates(subset='subject_id')
                   .copy())

    # ── Step 3: Unlabeled pool and 3:1 matching ──────────────────
    all_excluded = ptsd_subjects | proxy_subjects
    unlab_adm = adm[~adm['subject_id'].isin(all_excluded)].copy()
    unlab_first_time = (unlab_adm.groupby('subject_id')['admittime']
                        .min().reset_index()
                        .rename(columns={'admittime': 'first_admittime'}))
    unlab_index = (unlab_adm
                   .merge(unlab_first_time, on='subject_id')
                   .query('admittime == first_admittime')
                   .drop_duplicates(subset='subject_id')
                   .copy())
    unlab_index['decade'] = unlab_index['age_at_admission'].apply(age_decade)
    ptsd_index['decade']  = ptsd_index['age_at_admission'].apply(age_decade)

    # 3:1 stratified match on age_decade × sex
    np.random.seed(42)
    matched_rows = []
    for (dec, sex), grp in ptsd_index.groupby(['decade', 'gender']):
        needed = len(grp) * 3
        pool = unlab_index[
            (unlab_index['decade'] == dec) &
            (unlab_index['gender'] == sex)
        ]
        take = min(needed, len(pool))
        if take < needed:
            print(f'  WARNING: only {len(pool)} unlabeled in stratum '
                  f'(decade={dec}, sex={sex}); needed {needed}')
        if take > 0:
            matched_rows.append(pool.sample(n=take, random_state=42))

    unlab_matched = pd.concat(matched_rows, ignore_index=True)
    unlab_subjects = set(unlab_matched['subject_id'].unique())
    print(f'  Matched unlabeled pool (target 3:1): {len(unlab_matched):,}')

    # ── Step 4: Build prior-admission diagnosis and Rx tables ────
    print('\n[6/7] Computing prior-admission comorbidities and medications ...')

    def get_prior_dx(subject_set, index_times_df):
        """Diagnoses from admissions before each patient's index admission."""
        sub_adm = (adm[adm['subject_id'].isin(subject_set)]
                   [['subject_id', 'hadm_id', 'admittime']]
                   .merge(index_times_df, on='subject_id', how='left'))
        prior_hadms = sub_adm[sub_adm['admittime'] < sub_adm['index_time']][
            ['subject_id', 'hadm_id']].drop_duplicates()
        return dx.merge(prior_hadms, on=['subject_id', 'hadm_id'], how='inner')

    def get_prior_rx(subject_set, index_times_df):
        """Prescriptions from admissions before each patient's index admission."""
        sub_adm = (adm[adm['subject_id'].isin(subject_set)]
                   [['subject_id', 'hadm_id', 'admittime']]
                   .merge(index_times_df, on='subject_id', how='left'))
        prior_hadms = sub_adm[sub_adm['admittime'] < sub_adm['index_time']][
            ['subject_id', 'hadm_id']].drop_duplicates()
        return rx.merge(prior_hadms, on=['subject_id', 'hadm_id'], how='inner')

    # PTSD+
    ptsd_idx_times = ptsd_index[['subject_id', 'admittime']].rename(
        columns={'admittime': 'index_time'})
    ptsd_prior_dx = get_prior_dx(ptsd_subjects, ptsd_idx_times)
    ptsd_prior_rx = get_prior_rx(ptsd_subjects, ptsd_idx_times)
    print(f'  PTSD+ prior dx rows: {len(ptsd_prior_dx):,} | prior rx rows: {len(ptsd_prior_rx):,}')

    # Proxy
    proxy_idx_times = proxy_index[['subject_id', 'admittime']].rename(
        columns={'admittime': 'index_time'})
    proxy_prior_dx = get_prior_dx(proxy_subjects, proxy_idx_times)
    proxy_prior_rx = get_prior_rx(proxy_subjects, proxy_idx_times)
    print(f'  Proxy prior dx rows: {len(proxy_prior_dx):,} | prior rx rows: {len(proxy_prior_rx):,}')

    # Unlabeled matched
    unlab_idx_times = unlab_matched[['subject_id', 'admittime']].rename(
        columns={'admittime': 'index_time'})
    unlab_prior_dx = get_prior_dx(unlab_subjects, unlab_idx_times)
    unlab_prior_rx = get_prior_rx(unlab_subjects, unlab_idx_times)
    print(f'  Unlabeled prior dx rows: {len(unlab_prior_dx):,} | prior rx rows: {len(unlab_prior_rx):,}')

    # ── Step 5: Compute stats ────────────────────────────────────
    print('\n[7/7] Computing Table 1 statistics ...')

    stats_ptsd = compute_group_stats(
        label       = 'ICD-coded PTSD+ (Training Positives)',
        index_adm_df = ptsd_index,
        adm_counts_df = adm_counts,
        prior_dx_df = ptsd_prior_dx,
        prior_rx_df = ptsd_prior_rx,
        predx_n     = len(predx_subjects)
    )

    stats_proxy = compute_group_stats(
        label       = 'Pharmacological Proxy (External Validation)',
        index_adm_df = proxy_index,
        adm_counts_df = adm_counts,
        prior_dx_df = proxy_prior_dx,
        prior_rx_df = proxy_prior_rx,
        predx_n     = None
    )

    stats_unlab = compute_group_stats(
        label       = 'Matched Unlabeled Pool (PU Pool)',
        index_adm_df = unlab_matched,
        adm_counts_df = adm_counts,
        prior_dx_df = unlab_prior_dx,
        prior_rx_df = unlab_prior_rx,
        predx_n     = None
    )

    all_stats = [stats_ptsd, stats_proxy, stats_unlab]

    # ── Step 6: Save CSV ─────────────────────────────────────────
    results_df = (pd.DataFrame(all_stats)
                  .set_index('Group')
                  .T
                  .reset_index()
                  .rename(columns={'index': 'Characteristic'}))
    results_df.to_csv(OUT_CSV, index=False)
    print(f'\nCSV saved → {OUT_CSV}')

    # ── Step 7: Save human-readable summary ──────────────────────
    col_w = 50
    groups = [s['Group'] for s in all_stats]

    def row(label, keys):
        line = f'{label:<{col_w}}'
        for s in all_stats:
            vals = [str(s.get(k, 'N/A')) for k in keys]
            line += f'  {" / ".join(vals):<30}'
        return line

    sep = '─' * (col_w + 34 * len(groups))
    lines = [
        sep,
        'TABLE 1: Study Population Characteristics',
        'PTSD NLP Project — AIH 2025 (Spring 2026)',
        sep,
        row('Characteristic', ['Group']),
        sep,
        '',
        '── DEMOGRAPHICS ──────────────────────────────────────',
        row('N', ['N']),
        row('Sex: Female — n (%)', ['Female_n', 'Female_pct']),
        row('Sex: Male — n (%)', ['Male_n', 'Male_pct']),
        row('Sex: Other/Unknown — n (%)', ['OtherSex_n', 'OtherSex_pct']),
        row('Age, years — mean ± SD', ['Age_mean', 'Age_sd']),
        row('  Age 20s — n (%)', ['Age_20s_n', 'Age_20s_pct']),
        row('  Age 30s — n (%)', ['Age_30s_n', 'Age_30s_pct']),
        row('  Age 40s — n (%)', ['Age_40s_n', 'Age_40s_pct']),
        row('  Age 50s — n (%)', ['Age_50s_n', 'Age_50s_pct']),
        row('  Age other decades — n (%)', ['Age_Other_n', 'Age_Other_pct']),
        row('Race: White/Non-Hispanic — n (%)',
            ['Race_White_Non_Hispanic_n', 'Race_White_Non_Hispanic_pct']),
        row('Race: Black/African American — n (%)',
            ['Race_Black_African_American_n', 'Race_Black_African_American_pct']),
        row('Race: Hispanic/Latino — n (%)',
            ['Race_Hispanic_Latino_n', 'Race_Hispanic_Latino_pct']),
        row('Race: Asian — n (%)',
            ['Race_Asian_n', 'Race_Asian_pct']),
        row('Race: Other/Unknown — n (%)',
            ['Race_Other_Unknown_n', 'Race_Other_Unknown_pct']),
        '',
        '── HOSPITALIZATION ───────────────────────────────────',
        row('Single admission — n (%)', ['Single_admit_n', 'Single_admit_pct']),
        row('Multi-admission — n (%)', ['Multi_admit_n', 'Multi_admit_pct']),
        row('Pre-dx notes available (ICD+ only) — n (%)',
            ['PreDx_n', 'PreDx_pct']),
        row('Total admissions — median [Q1, Q3]',
            ['Admissions_median', 'Admissions_Q1', 'Admissions_Q3']),
        row('LOS index admission (days) — median [Q1, Q3]',
            ['LOS_median', 'LOS_Q1', 'LOS_Q3']),
        row('Emergency admission — n (%)',
            ['Emergency_n', 'Emergency_pct']),
        row('Medicaid or self-pay — n (%)',
            ['MedicaidSelfPay_n', 'MedicaidSelfPay_pct']),
        '',
        '── PRIOR-ADMISSION COMORBIDITIES ─────────────────────',
        row('Major depressive disorder (F32-33 / ICD-9: 296) — n (%)',
            ['Dx_MDD_n', 'Dx_MDD_pct']),
        row('Anxiety disorder (F41 / ICD-9: 300) — n (%)',
            ['Dx_Anxiety_n', 'Dx_Anxiety_pct']),
        row('Substance use disorder (F10-19 / ICD-9: 303-305) — n (%)',
            ['Dx_SUD_n', 'Dx_SUD_pct']),
        row('Traumatic brain injury (S06 / ICD-9: 800-804,850-854) — n (%)',
            ['Dx_TBI_n', 'Dx_TBI_pct']),
        row('Chronic pain (G89 / ICD-9: 338) — n (%)',
            ['Dx_ChronicPain_n', 'Dx_ChronicPain_pct']),
        row('Suicidal ideation/attempt (R458/V6284/E95) — n (%)',
            ['Dx_Suicidal_n', 'Dx_Suicidal_pct']),
        '',
        '── PRIOR-ADMISSION MEDICATIONS ───────────────────────',
        row('SSRI or SNRI — n (%)', ['Rx_SSRI_SNRI_n', 'Rx_SSRI_SNRI_pct']),
        row('Prazosin — n (%)', ['Rx_Prazosin_n', 'Rx_Prazosin_pct']),
        row('Second-generation antipsychotic — n (%)',
            ['Rx_SGA_n', 'Rx_SGA_pct']),
        '',
        sep,
        'Notes:',
        '  - Demographics are at the index admission (first PTSD-coded for PTSD+ group;',
        '    first admission overall for proxy and unlabeled groups).',
        '  - Comorbidities and medications are from admissions BEFORE the index admission.',
        '    Patients with only one admission have 0 prior admissions (all flags = 0).',
        '  - Unlabeled pool is matched 3:1 on age decade x sex to the PTSD+ group.',
        '  - Proxy exclusions applied at ANY diagnosis position (not only primary dx).',
        '  - Prazosin + SSRI/SNRI window defined as starttime difference ≤ 180 days.',
        sep,
    ]

    summary_text = '\n'.join(lines)
    print('\n' + summary_text)

    with open(OUT_TXT, 'w') as f:
        f.write(summary_text)
    print(f'\nSummary saved → {OUT_TXT}')
    print('\nDone.')


main()
