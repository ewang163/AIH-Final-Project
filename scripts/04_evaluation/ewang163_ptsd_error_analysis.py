"""
ewang163_ptsd_error_analysis.py
===============================
Qualitative error analysis of Longformer false positives and false negatives.

FPs (unlabeled patients predicted positive) may include genuine undercoded PTSD.
FNs (coded PTSD+ patients predicted negative) are the model's real misses.

Outputs:
    ewang163_fp_notes_sample.txt
    ewang163_fn_notes_sample.txt
    ewang163_error_analysis_summary.csv

Submit via SLURM:
    sbatch ewang163_ptsd_error_analysis.sh
"""

import pickle
import numpy as np
import pandas as pd
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR         = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS         = f'{STUDENT_DIR}/data/splits'
DATA_COHORT         = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR           = f'{STUDENT_DIR}/models'
RESULTS_PREDICTIONS = f'{STUDENT_DIR}/results/predictions'
RESULTS_ERROR_ANALYSIS = f'{STUDENT_DIR}/results/error_analysis'

TEST_PARQUET  = f'{DATA_SPLITS}/ewang163_split_test.parquet'
PRED_CSV      = f'{RESULTS_PREDICTIONS}/ewang163_longformer_test_predictions.csv'
ADM_PARQUET   = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
TFIDF_PKL     = f'{MODEL_DIR}/ewang163_tfidf_vectorizer.pkl'

FP_TXT        = f'{RESULTS_ERROR_ANALYSIS}/ewang163_fp_notes_sample.txt'
FN_TXT        = f'{RESULTS_ERROR_ANALYSIS}/ewang163_fn_notes_sample.txt'
SUMMARY_CSV   = f'{RESULTS_ERROR_ANALYSIS}/ewang163_error_analysis_summary.csv'

THRESHOLD = 0.38  # from evaluation: recall >= 0.85 cutoff
N_SAMPLE  = 25


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


def write_sample_file(filepath, sample_df, label):
    """Write sampled notes with metadata to a text file."""
    with open(filepath, 'w') as f:
        f.write(f'{"=" * 70}\n')
        f.write(f'  {label} — {len(sample_df)} sampled patients\n')
        f.write(f'  Threshold: {THRESHOLD} (recall >= 0.85)\n')
        f.write(f'{"=" * 70}\n\n')

        for i, (_, row) in enumerate(sample_df.iterrows(), 1):
            f.write(f'{"─" * 70}\n')
            f.write(f'  Patient {i}/{len(sample_df)}\n')
            f.write(f'  subject_id:     {row["subject_id"]}\n')
            f.write(f'  predicted_prob: {row["predicted_prob"]:.4f}\n')
            f.write(f'  true_label:     {row["ptsd_label"]} '
                    f'({"PTSD+" if row["ptsd_label"] == 1 else "unlabeled"})\n')
            f.write(f'  sex:            {row.get("gender", "Unknown")}\n')
            f.write(f'  age_group:      {row.get("age_group", "Unknown")}\n')
            f.write(f'  race:           {row.get("race_cat", "Unknown")}\n')
            f.write(f'  admission_type: {row.get("admission_type", "Unknown")}\n')
            f.write(f'  note_length:    {len(row["note_text"]):,} chars\n')
            f.write(f'{"─" * 70}\n\n')
            f.write(row['note_text'])
            f.write('\n\n')


def tfidf_top_terms(tfidf_vec, texts, n_top=20):
    """Get top-n terms by mean TF-IDF weight in a set of texts."""
    X = tfidf_vec.transform(texts)
    mean_weights = np.asarray(X.mean(axis=0)).flatten()
    feature_names = tfidf_vec.get_feature_names_out()
    top_idx = mean_weights.argsort()[::-1][:n_top]
    return [(feature_names[i], round(float(mean_weights[i]), 6)) for i in top_idx]


def main():
    print('=' * 65)
    print('PTSD NLP — Error Analysis (FP/FN Inspection)')
    print('=' * 65, flush=True)

    # ── Load data ────────────────────────────────────────────────────────
    print('\n[1/5] Loading data ...')
    test_df = pd.read_parquet(TEST_PARQUET)
    pred_df = pd.read_csv(PRED_CSV)
    adm = pd.read_parquet(ADM_PARQUET)

    # Merge predictions into test data
    test_df = test_df.merge(
        pred_df[['subject_id', 'hadm_id', 'predicted_prob']],
        on=['subject_id', 'hadm_id'],
        how='left',
    )
    print(f'  Test set: {len(test_df):,} patients')
    print(f'  Threshold: {THRESHOLD}')

    # Add demographics
    adm_demo = adm.drop_duplicates('subject_id', keep='first')[
        ['subject_id', 'gender', 'age_at_admission', 'race', 'admission_type']
    ].copy()
    adm_demo['race_cat'] = adm_demo['race'].apply(map_race)
    adm_demo['age_group'] = adm_demo['age_at_admission'].apply(age_decade)

    test_df = test_df.merge(
        adm_demo[['subject_id', 'gender', 'age_group', 'race_cat', 'admission_type']],
        on='subject_id', how='left',
    )

    # Compute note length
    test_df['note_length'] = test_df['note_text'].str.len()

    # ── Identify FP and FN ───────────────────────────────────────────────
    print('\n[2/5] Identifying false positives and false negatives ...')
    test_df['predicted'] = (test_df['predicted_prob'] >= THRESHOLD).astype(int)

    fp_mask = (test_df['predicted'] == 1) & (test_df['ptsd_label'] == 0)
    fn_mask = (test_df['predicted'] == 0) & (test_df['ptsd_label'] == 1)
    tp_mask = (test_df['predicted'] == 1) & (test_df['ptsd_label'] == 1)
    tn_mask = (test_df['predicted'] == 0) & (test_df['ptsd_label'] == 0)

    fp_df = test_df[fp_mask].copy()
    fn_df = test_df[fn_mask].copy()

    print(f'  TP: {tp_mask.sum():,}  FP: {len(fp_df):,}  '
          f'FN: {len(fn_df):,}  TN: {tn_mask.sum():,}')

    # ── Sample 25 of each ────────────────────────────────────────────────
    print('\n[3/5] Sampling {0} FPs and {0} FNs ...'.format(N_SAMPLE))

    # FPs: sample diverse range of predicted probs (not just borderline)
    if len(fp_df) > N_SAMPLE:
        fp_sample = fp_df.sample(n=N_SAMPLE, random_state=42)
    else:
        fp_sample = fp_df
    fp_sample = fp_sample.sort_values('predicted_prob', ascending=False)

    if len(fn_df) > N_SAMPLE:
        fn_sample = fn_df.sample(n=N_SAMPLE, random_state=42)
    else:
        fn_sample = fn_df
    fn_sample = fn_sample.sort_values('predicted_prob', ascending=False)

    print(f'  FP sample: {len(fp_sample)} patients')
    print(f'  FN sample: {len(fn_sample)} patients')

    # ── Write note files ─────────────────────────────────────────────────
    print('\n[4/5] Writing sample note files ...')
    write_sample_file(FP_TXT, fp_sample, 'FALSE POSITIVES (unlabeled predicted as PTSD+)')
    print(f'  → {FP_TXT}')

    write_sample_file(FN_TXT, fn_sample, 'FALSE NEGATIVES (PTSD+ predicted as negative)')
    print(f'  → {FN_TXT}')

    # ── Aggregate analysis ───────────────────────────────────────────────
    print('\n[5/5] Computing aggregate statistics ...')

    # Load TF-IDF vectorizer for term analysis
    with open(TFIDF_PKL, 'rb') as f:
        tfidf_vec = pickle.load(f)

    overall_mean_length = test_df['note_length'].mean()

    summary_rows = []

    for label, subset in [('false_positive', fp_df), ('false_negative', fn_df),
                          ('overall_test', test_df)]:
        n = len(subset)
        if n == 0:
            continue

        row = {
            'group': label,
            'n': n,
            'mean_predicted_prob': round(subset['predicted_prob'].mean(), 4),
            'median_predicted_prob': round(subset['predicted_prob'].median(), 4),
            'mean_note_length': round(subset['note_length'].mean(), 0),
            'median_note_length': round(subset['note_length'].median(), 0),
        }

        # Sex distribution
        if 'gender' in subset.columns:
            sex_counts = subset['gender'].value_counts(normalize=True)
            row['pct_female'] = round(sex_counts.get('F', 0) * 100, 1)
            row['pct_male'] = round(sex_counts.get('M', 0) * 100, 1)

        # Age group distribution
        if 'age_group' in subset.columns:
            age_counts = subset['age_group'].value_counts(normalize=True)
            for ag in ['20s', '30s', '40s', '50s', 'Other']:
                row[f'pct_age_{ag}'] = round(age_counts.get(ag, 0) * 100, 1)

        # Race distribution
        if 'race_cat' in subset.columns:
            race_counts = subset['race_cat'].value_counts(normalize=True)
            for rc in ['White', 'Black', 'Hispanic', 'Asian', 'Other/Unknown']:
                row[f'pct_race_{rc}'] = round(race_counts.get(rc, 0) * 100, 1)

        # Emergency admission rate
        if 'admission_type' in subset.columns:
            emer = subset['admission_type'].str.upper().str.contains('EMER', na=False)
            row['pct_emergency'] = round(emer.mean() * 100, 1)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f'  → {SUMMARY_CSV}')

    # TF-IDF top terms
    fp_top = tfidf_top_terms(tfidf_vec, fp_df['note_text'].tolist()) if len(fp_df) > 0 else []
    fn_top = tfidf_top_terms(tfidf_vec, fn_df['note_text'].tolist()) if len(fn_df) > 0 else []
    all_top = tfidf_top_terms(tfidf_vec, test_df['note_text'].tolist())

    # ── Print results ────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('ERROR ANALYSIS SUMMARY')
    print('=' * 65)

    # Counts
    print(f'\n  Confusion matrix (threshold={THRESHOLD}):')
    print(f'    TP={tp_mask.sum():,}  FP={len(fp_df):,}')
    print(f'    FN={len(fn_df):,}  TN={tn_mask.sum():,}')

    # FP analysis
    print(f'\n  FALSE POSITIVES (n={len(fp_df):,}):')
    print(f'    Mean predicted prob: {fp_df["predicted_prob"].mean():.4f}')
    print(f'    Mean note length:    {fp_df["note_length"].mean():,.0f} chars '
          f'(overall: {overall_mean_length:,.0f})')
    if 'gender' in fp_df.columns:
        sex_fp = fp_df['gender'].value_counts()
        print(f'    Sex: {dict(sex_fp)}')
    if 'age_group' in fp_df.columns:
        age_fp = fp_df['age_group'].value_counts()
        print(f'    Age: {dict(age_fp)}')
    if 'race_cat' in fp_df.columns:
        race_fp = fp_df['race_cat'].value_counts()
        print(f'    Race: {dict(race_fp)}')
    if 'admission_type' in fp_df.columns:
        emer_fp = fp_df['admission_type'].str.upper().str.contains('EMER', na=False).sum()
        print(f'    Emergency admissions: {emer_fp}/{len(fp_df)} '
              f'({emer_fp/len(fp_df)*100:.1f}%)')

    # FN analysis
    print(f'\n  FALSE NEGATIVES (n={len(fn_df):,}):')
    print(f'    Mean predicted prob: {fn_df["predicted_prob"].mean():.4f}')
    print(f'    Mean note length:    {fn_df["note_length"].mean():,.0f} chars '
          f'(overall: {overall_mean_length:,.0f})')
    if 'gender' in fn_df.columns:
        sex_fn = fn_df['gender'].value_counts()
        print(f'    Sex: {dict(sex_fn)}')
    if 'age_group' in fn_df.columns:
        age_fn = fn_df['age_group'].value_counts()
        print(f'    Age: {dict(age_fn)}')
    if 'race_cat' in fn_df.columns:
        race_fn = fn_df['race_cat'].value_counts()
        print(f'    Race: {dict(race_fn)}')
    if 'admission_type' in fn_df.columns:
        emer_fn = fn_df['admission_type'].str.upper().str.contains('EMER', na=False).sum()
        print(f'    Emergency admissions: {emer_fn}/{len(fn_df)} '
              f'({emer_fn/len(fn_df)*100:.1f}%)')

    # TF-IDF distinctive terms
    print(f'\n  TF-IDF top-20 terms (mean weight):')
    print(f'\n    FALSE POSITIVES:')
    for term, weight in fp_top[:20]:
        print(f'      {term:<25} {weight:.6f}')
    print(f'\n    FALSE NEGATIVES:')
    for term, weight in fn_top[:20]:
        print(f'      {term:<25} {weight:.6f}')
    print(f'\n    OVERALL TEST SET:')
    for term, weight in all_top[:20]:
        print(f'      {term:<25} {weight:.6f}')

    # FP/FN distinctive terms (over-represented vs overall)
    overall_dict = {t: w for t, w in all_top}
    print(f'\n  DISTINCTIVE FP terms (most over-represented vs overall):')
    fp_excess = []
    for term, weight in fp_top:
        baseline = overall_dict.get(term, 0.0001)
        ratio = weight / baseline if baseline > 0 else 0
        fp_excess.append((term, weight, ratio))
    fp_excess.sort(key=lambda x: -x[2])
    for term, weight, ratio in fp_excess[:15]:
        print(f'      {term:<25} {weight:.6f}  ({ratio:.1f}x overall)')

    print(f'\n  DISTINCTIVE FN terms (most over-represented vs overall):')
    fn_excess = []
    for term, weight in fn_top:
        baseline = overall_dict.get(term, 0.0001)
        ratio = weight / baseline if baseline > 0 else 0
        fn_excess.append((term, weight, ratio))
    fn_excess.sort(key=lambda x: -x[2])
    for term, weight, ratio in fn_excess[:15]:
        print(f'      {term:<25} {weight:.6f}  ({ratio:.1f}x overall)')

    # ── Qualitative summary questions ────────────────────────────────────
    print('\n' + '=' * 65)
    print('QUALITATIVE QUESTIONS TO ADDRESS')
    print('=' * 65)

    # Check for trauma-related language in FPs
    trauma_terms = ['trauma', 'abuse', 'assault', 'violence', 'ptsd',
                    'nightmare', 'flashback', 'hypervigilant', 'combat',
                    'veteran', 'sexual', 'domestic', 'anxiety', 'panic',
                    'intrusive', 'avoidance', 'hyperarousal', 'startle']
    fp_trauma_counts = Counter()
    for text in fp_df['note_text'].str.lower():
        for term in trauma_terms:
            if term in text:
                fp_trauma_counts[term] += 1

    print(f'\n  1. Do FP notes contain trauma-related language?')
    if fp_trauma_counts:
        print(f'     YES — trauma terms found in FP notes:')
        for term, count in fp_trauma_counts.most_common():
            pct = count / len(fp_df) * 100
            print(f'       "{term}": {count}/{len(fp_df)} ({pct:.0f}%)')
        print(f'     NOTE: These may be genuine undercoded PTSD cases.')
    else:
        print(f'     No trauma terms found in FP notes.')

    # Check FN note sparsity
    fn_short = (fn_df['note_length'] < test_df['note_length'].quantile(0.25)).sum()
    fn_very_short = (fn_df['note_length'] < 500).sum()
    print(f'\n  2. Are FN notes clinically sparse?')
    print(f'     FN notes below Q1 length: {fn_short}/{len(fn_df)} '
          f'({fn_short/len(fn_df)*100:.0f}%)')
    print(f'     FN notes < 500 chars: {fn_very_short}/{len(fn_df)} '
          f'({fn_very_short/len(fn_df)*100:.0f}%)')
    print(f'     Mean FN length: {fn_df["note_length"].mean():,.0f} '
          f'(overall: {overall_mean_length:,.0f})')

    fn_trauma_counts = Counter()
    for text in fn_df['note_text'].str.lower():
        for term in trauma_terms:
            if term in text:
                fn_trauma_counts[term] += 1
    print(f'     Trauma terms in FN notes:')
    for term, count in fn_trauma_counts.most_common(10):
        pct = count / len(fn_df) * 100
        print(f'       "{term}": {count}/{len(fn_df)} ({pct:.0f}%)')

    # Subgroup clustering
    print(f'\n  3. Subgroup clustering of FNs:')
    if 'gender' in fn_df.columns:
        overall_fem = (test_df['gender'] == 'F').mean()
        fn_fem = (fn_df['gender'] == 'F').mean()
        print(f'     Female: FN={fn_fem*100:.1f}% vs overall={overall_fem*100:.1f}%')
    if 'age_group' in fn_df.columns:
        for ag in ['20s', '30s', '40s', '50s', 'Other']:
            overall_ag = (test_df['age_group'] == ag).mean()
            fn_ag = (fn_df['age_group'] == ag).mean() if len(fn_df) > 0 else 0
            if abs(fn_ag - overall_ag) > 0.05:
                print(f'     Age {ag}: FN={fn_ag*100:.1f}% vs overall={overall_ag*100:.1f}%'
                      f' ({"over" if fn_ag > overall_ag else "under"}-represented)')

    print('\nDone.')


if __name__ == '__main__':
    main()
