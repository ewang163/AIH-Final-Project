"""
ewang163_ptsd_error_analysis.py
===============================
Qualitative error analysis of the primary PULSNAR Longformer's false positives
and false negatives. (BERT error analysis is produced inline by
ewang163_ptsd_bert_full_eval.py.)

Steps:
  1. Load PULSNAR Longformer test predictions, attach demographics from the
     admissions extract.
  2. Identify FP / FN at the val-derived operating threshold.
  3. Sample 25 of each, write annotated note text files for qualitative review.
  4. Compute aggregate demographic + note-length statistics, write to CSV.
  5. Compute clinical-vocabulary overrepresentation in FP/FN vs. the overall
     test set using a curated lexicon (trauma exposure, intrusion, avoidance,
     arousal, substance, psychiatric comorbidity).

Outputs:
    results/error_analysis/ewang163_fp_notes_sample.txt
    results/error_analysis/ewang163_fn_notes_sample.txt
    results/error_analysis/ewang163_error_analysis_summary.csv
    results/error_analysis/ewang163_distinctive_terms.csv

Submit via SLURM:
    sbatch ewang163_ptsd_error_analysis.sh
"""

import os
import re
from collections import Counter
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR             = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS             = f'{STUDENT_DIR}/data/splits'
DATA_COHORT             = f'{STUDENT_DIR}/data/cohort'
RESULTS_PREDICTIONS     = f'{STUDENT_DIR}/results/predictions'
RESULTS_ERROR_ANALYSIS  = f'{STUDENT_DIR}/results/error_analysis'
RESULTS_METRICS         = f'{STUDENT_DIR}/results/metrics'

TEST_PARQUET   = f'{DATA_SPLITS}/ewang163_split_test.parquet'
PRED_CSV       = f'{RESULTS_PREDICTIONS}/ewang163_longformer_test_predictions_pulsnar.csv'
ADM_PARQUET    = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
EVAL_JSON      = f'{RESULTS_METRICS}/ewang163_evaluation_results_pulsnar.json'

FP_TXT         = f'{RESULTS_ERROR_ANALYSIS}/ewang163_fp_notes_sample.txt'
FN_TXT         = f'{RESULTS_ERROR_ANALYSIS}/ewang163_fn_notes_sample.txt'
SUMMARY_CSV    = f'{RESULTS_ERROR_ANALYSIS}/ewang163_error_analysis_summary.csv'
TERMS_CSV      = f'{RESULTS_ERROR_ANALYSIS}/ewang163_distinctive_terms.csv'

N_SAMPLE = 25

# Curated clinical vocabulary (no machine-learned features needed). Each term
# is matched as a case-insensitive whole-word regex.
CLINICAL_LEXICON = {
    # PTSD-adjacent narrative content (note: PTSD itself is masked in training
    # text but raw clinical terms here are matched against unmasked test text)
    'trauma':         r'\btrauma',
    'abuse':          r'\babus',
    'assault':        r'\bassault',
    'violence':       r'\bviolen',
    'nightmare':      r'\bnightmar',
    'flashback':      r'\bflashback',
    'hypervigilant':  r'\bhypervigil',
    'startle':        r'\bstartle',
    'intrusive':      r'\bintrusiv',
    'avoidance':      r'\bavoidan',
    'combat':         r'\bcombat',
    'veteran':        r'\bveteran',
    'sexual':         r'\bsexual',
    'domestic':       r'\bdomestic',
    'mva':            r'\bmva\b|\bmvc\b',
    'gunshot':        r'\bgunshot',
    'stabbing':       r'\bstab(?:bed|bing)?\b',
    # Treatment context
    'prazosin':       r'\bprazosin',
    'sertraline':     r'\bsertraline',
    'paroxetine':     r'\bparoxetine',
    'emdr':           r'\bemdr\b',
    # Comorbidity / severity
    'depression':     r'\bdepress',
    'anxiety':        r'\banxiet',
    'panic':          r'\bpanic',
    'bipolar':        r'\bbipolar',
    'schizo':         r'\bschizo',
    'psychiatric':    r'\bpsychiatric',
    'suicide':        r'\bsuici',
    'self_harm':      r'\bself.?harm',
    'overdose':       r'\boverdose',
    # Substance use
    'heroin':         r'\bheroin',
    'cocaine':        r'\bcocaine',
    'methadone':      r'\bmethadone',
    'opioid':         r'\bopioid',
    'alcohol':        r'\balcohol',
    'detox':          r'\bdetox',
}
COMPILED_LEXICON = {name: re.compile(pat, re.IGNORECASE)
                    for name, pat in CLINICAL_LEXICON.items()}


def map_race(race_str):
    if pd.isna(race_str):
        return 'Other/Unknown'
    r = str(race_str).upper()
    if 'WHITE' in r:
        return 'White'
    if 'BLACK' in r or 'AFRICAN' in r:
        return 'Black'
    if 'HISPANIC' in r or 'LATINO' in r:
        return 'Hispanic'
    if 'ASIAN' in r:
        return 'Asian'
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


def lexicon_rates(texts):
    """For each lexicon term, fraction of texts that match it at least once."""
    if len(texts) == 0:
        return {term: 0.0 for term in COMPILED_LEXICON}
    rates = {}
    for term, regex in COMPILED_LEXICON.items():
        n_match = sum(1 for t in texts if regex.search(t or ''))
        rates[term] = n_match / len(texts)
    return rates


def write_sample_file(filepath, sample_df, label, threshold):
    with open(filepath, 'w') as f:
        f.write(f'{"=" * 70}\n')
        f.write(f'  {label} — {len(sample_df)} sampled patients\n')
        f.write(f'  Operating threshold: {threshold:.4f} (val-derived)\n')
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


def load_threshold():
    """Pull the val-derived threshold from the PULSNAR re-eval JSON."""
    import json
    with open(EVAL_JSON) as f:
        return float(json.load(f)['val_threshold'])


def main():
    print('=' * 65)
    print('PTSD NLP — Error Analysis (PULSNAR Longformer FP/FN)')
    print('=' * 65, flush=True)

    threshold = load_threshold()

    # ── Load data ────────────────────────────────────────────────────────
    print('\n[1/5] Loading data ...')
    test_df = pd.read_parquet(TEST_PARQUET)
    pred_df = pd.read_csv(PRED_CSV)
    adm = pd.read_parquet(ADM_PARQUET)

    # PULSNAR predictions are positionally aligned with test_df, no hadm_id
    if 'hadm_id' in pred_df.columns:
        test_df = test_df.merge(
            pred_df[['subject_id', 'hadm_id', 'predicted_prob']],
            on=['subject_id', 'hadm_id'], how='left',
        )
    else:
        if (pred_df['subject_id'].values == test_df['subject_id'].values).all():
            test_df = test_df.copy()
            test_df['predicted_prob'] = pred_df['predicted_prob'].values
        else:
            raise ValueError('PULSNAR predictions misaligned with test_df')
    print(f'  Test set: {len(test_df):,} patients')
    print(f'  Threshold: {threshold:.4f} (val-derived)')

    # Attach demographics
    adm_demo = adm.drop_duplicates('subject_id', keep='first')[
        ['subject_id', 'gender', 'age_at_admission', 'race', 'admission_type']
    ].copy()
    adm_demo['race_cat'] = adm_demo['race'].apply(map_race)
    adm_demo['age_group'] = adm_demo['age_at_admission'].apply(age_decade)
    test_df = test_df.merge(
        adm_demo[['subject_id', 'gender', 'age_group', 'race_cat',
                  'admission_type']],
        on='subject_id', how='left',
    )
    test_df['note_length'] = test_df['note_text'].str.len()

    # ── Identify FP / FN ─────────────────────────────────────────────────
    print('\n[2/5] Identifying false positives and false negatives ...')
    test_df['predicted'] = (test_df['predicted_prob'] >= threshold).astype(int)

    fp_df = test_df[(test_df['predicted'] == 1) & (test_df['ptsd_label'] == 0)].copy()
    fn_df = test_df[(test_df['predicted'] == 0) & (test_df['ptsd_label'] == 1)].copy()
    tp_n  = int(((test_df['predicted'] == 1) & (test_df['ptsd_label'] == 1)).sum())
    tn_n  = int(((test_df['predicted'] == 0) & (test_df['ptsd_label'] == 0)).sum())
    print(f'  TP={tp_n:,}  FP={len(fp_df):,}  FN={len(fn_df):,}  TN={tn_n:,}')

    # ── Sample 25 of each ────────────────────────────────────────────────
    print(f'\n[3/5] Sampling {N_SAMPLE} FPs and {N_SAMPLE} FNs ...')
    rng = np.random.RandomState(42)
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

    os.makedirs(RESULTS_ERROR_ANALYSIS, exist_ok=True)
    write_sample_file(FP_TXT, fp_sample,
                      'FALSE POSITIVES (unlabeled predicted as PTSD+)', threshold)
    write_sample_file(FN_TXT, fn_sample,
                      'FALSE NEGATIVES (PTSD+ predicted as negative)', threshold)
    print(f'  → {FP_TXT}')
    print(f'  → {FN_TXT}')

    # ── Aggregate demographic + length statistics ────────────────────────
    print('\n[4/5] Aggregate FP / FN / overall statistics ...')
    overall_len = test_df['note_length'].mean()
    summary_rows = []
    for label, sub in [('false_positive', fp_df),
                       ('false_negative', fn_df),
                       ('overall_test', test_df)]:
        if len(sub) == 0:
            continue
        row = {
            'group': label,
            'n': int(len(sub)),
            'mean_predicted_prob':   round(float(sub['predicted_prob'].mean()), 4),
            'median_predicted_prob': round(float(sub['predicted_prob'].median()), 4),
            'mean_note_length':      round(float(sub['note_length'].mean()), 0),
            'median_note_length':    round(float(sub['note_length'].median()), 0),
        }
        sex = sub['gender'].value_counts(normalize=True)
        row['pct_female'] = round(float(sex.get('F', 0)) * 100, 1)
        row['pct_male']   = round(float(sex.get('M', 0)) * 100, 1)
        ages = sub['age_group'].value_counts(normalize=True)
        for ag in ['20s', '30s', '40s', '50s', 'Other']:
            row[f'pct_age_{ag}'] = round(float(ages.get(ag, 0)) * 100, 1)
        races = sub['race_cat'].value_counts(normalize=True)
        for rc in ['White', 'Black', 'Hispanic', 'Asian', 'Other/Unknown']:
            row[f'pct_race_{rc}'] = round(float(races.get(rc, 0)) * 100, 1)
        emer = sub['admission_type'].astype(str).str.upper().str.contains('EMER', na=False)
        row['pct_emergency'] = round(float(emer.mean()) * 100, 1)
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print(f'  → {SUMMARY_CSV}')

    # ── Clinical-lexicon overrepresentation ──────────────────────────────
    print('\n[5/5] Clinical-vocabulary distinctive-term analysis ...')
    fp_rates  = lexicon_rates(fp_df['note_text'].tolist())
    fn_rates  = lexicon_rates(fn_df['note_text'].tolist())
    all_rates = lexicon_rates(test_df['note_text'].tolist())

    rows = []
    for term in COMPILED_LEXICON:
        rows.append({
            'term': term,
            'fp_rate':       round(fp_rates[term],  4),
            'fn_rate':       round(fn_rates[term],  4),
            'overall_rate':  round(all_rates[term], 4),
            'fp_over_overall_ratio': round(fp_rates[term] / all_rates[term], 3) if all_rates[term] > 0 else 0,
            'fn_over_overall_ratio': round(fn_rates[term] / all_rates[term], 3) if all_rates[term] > 0 else 0,
        })
    rates_df = pd.DataFrame(rows)
    rates_df.to_csv(TERMS_CSV, index=False)
    print(f'  → {TERMS_CSV}')

    # ── Print summary ────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('ERROR ANALYSIS SUMMARY')
    print('=' * 65)
    print(f'\n  Confusion at threshold={threshold:.4f}:')
    print(f'    TP={tp_n:,}  FP={len(fp_df):,}  FN={len(fn_df):,}  TN={tn_n:,}')

    for label, sub in [('FALSE POSITIVES', fp_df), ('FALSE NEGATIVES', fn_df)]:
        if len(sub) == 0:
            continue
        print(f'\n  {label} (n={len(sub):,}):')
        print(f'    mean predicted prob: {sub["predicted_prob"].mean():.4f}')
        print(f'    mean note length:    {sub["note_length"].mean():,.0f} chars '
              f'(overall: {overall_len:,.0f})')
        sex = dict(sub['gender'].value_counts())
        ages = dict(sub['age_group'].value_counts())
        races = dict(sub['race_cat'].value_counts())
        emer = sub['admission_type'].astype(str).str.upper().str.contains('EMER', na=False).sum()
        print(f'    sex:   {sex}')
        print(f'    age:   {ages}')
        print(f'    race:  {races}')
        print(f'    emergency: {emer}/{len(sub)} ({emer/len(sub)*100:.1f}%)')

    print('\n  Top 10 terms over-represented in FALSE POSITIVES (rate / overall):')
    fp_top = rates_df.sort_values('fp_over_overall_ratio', ascending=False).head(10)
    for _, r in fp_top.iterrows():
        print(f'    {r["term"]:<14} fp={r["fp_rate"]:.3f}  overall={r["overall_rate"]:.3f}  '
              f'ratio={r["fp_over_overall_ratio"]:.2f}x')

    print('\n  Top 10 terms over-represented in FALSE NEGATIVES (rate / overall):')
    fn_top = rates_df.sort_values('fn_over_overall_ratio', ascending=False).head(10)
    for _, r in fn_top.iterrows():
        print(f'    {r["term"]:<14} fn={r["fn_rate"]:.3f}  overall={r["overall_rate"]:.3f}  '
              f'ratio={r["fn_over_overall_ratio"]:.2f}x')

    # FN length sparsity
    fn_q1 = (fn_df['note_length'] < test_df['note_length'].quantile(0.25)).sum()
    fn_short = (fn_df['note_length'] < 500).sum()
    print(f'\n  FN sparsity:')
    print(f'    FN notes below test-set Q1 length: {fn_q1}/{len(fn_df)} '
          f'({fn_q1/len(fn_df)*100:.0f}%)')
    print(f'    FN notes < 500 chars:              {fn_short}/{len(fn_df)} '
          f'({fn_short/len(fn_df)*100:.0f}%)')

    print('\nDone.')


if __name__ == '__main__':
    main()
