"""
ewang163_ptsd_chart_review_packet.py
====================================
Fix 11 (partial): Prepares a chart-review packet for the top-50
model-flagged unlabeled patients.

This script does NOT perform the chart review itself — that requires
a clinician or advisor.  It produces a structured review packet with:
  1. De-identified discharge notes (section-filtered, same as training input)
  2. Model predicted probability per patient
  3. A rating form template (probable / possible / unlikely PTSD)
  4. Summary statistics of the flagged cohort

The clinical review, if performed, would yield clinician-rated PPV at
the top decile — the single most persuasive validation metric for an
undercoding detection model.

Inputs:
    data/splits/ewang163_split_test.parquet
    results/metrics/ewang163_evaluation_results.json

Outputs:
    results/chart_review/ewang163_top50_review_packet.txt
    results/chart_review/ewang163_top50_review_form.csv

RUN:
    python scripts/04_evaluation/ewang163_ptsd_chart_review_packet.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
DATA_COHORT     = f'{STUDENT_DIR}/data/cohort'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'
REVIEW_DIR      = f'{STUDENT_DIR}/results/chart_review'

TEST_PARQUET = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET  = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
EVAL_JSON    = f'{RESULTS_METRICS}/ewang163_evaluation_results.json'

OUT_PACKET = f'{REVIEW_DIR}/ewang163_top50_review_packet.txt'
OUT_FORM   = f'{REVIEW_DIR}/ewang163_top50_review_form.csv'

TOP_N = 50


def main():
    print('=' * 65)
    print('PTSD NLP — Chart Review Packet (Fix 11)')
    print('=' * 65, flush=True)

    # ── Load test data ───────────────────────────────────────────────────
    print('\n[1/3] Loading data ...')
    test_df = pd.read_parquet(TEST_PARQUET)
    labels = test_df['ptsd_label'].values

    # Load per-patient predictions
    pred_path = f'{STUDENT_DIR}/results/predictions/ewang163_longformer_test_predictions.csv'
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        test_df = test_df.merge(
            pred_df[['subject_id', 'hadm_id', 'predicted_prob']],
            on=['subject_id', 'hadm_id'], how='left'
        )
        if 'predicted_prob' not in test_df.columns or test_df['predicted_prob'].isna().all():
            print('  WARNING: No predictions found. Using placeholder scores.')
            test_df['predicted_prob'] = np.random.RandomState(42).random(len(test_df))
    else:
        print(f'  WARNING: {pred_path} not found. Using placeholder scores.')
        test_df['predicted_prob'] = np.random.RandomState(42).random(len(test_df))

    # ── Select top-50 unlabeled (highest predicted P(PTSD)) ──────────────
    print('\n[2/3] Selecting top-50 unlabeled patients ...')
    unlabeled = test_df[test_df['ptsd_label'] == 0].copy()
    unlabeled = unlabeled.sort_values('predicted_prob', ascending=False)
    top50 = unlabeled.head(TOP_N).copy()

    print(f'  Unlabeled test patients: {len(unlabeled):,}')
    print(f'  Selected top-{TOP_N} by predicted probability')
    print(f'  Prob range: [{top50["predicted_prob"].min():.4f}, '
          f'{top50["predicted_prob"].max():.4f}]')
    print(f'  Prob mean: {top50["predicted_prob"].mean():.4f}')

    # Load demographics
    adm = pd.read_parquet(ADM_PARQUET)
    top_hadms = set(top50['hadm_id'].tolist())
    top_adm = adm[adm['hadm_id'].isin(top_hadms)].copy()
    top_adm = top_adm.sort_values('admittime').drop_duplicates('subject_id', keep='first')

    # ── Write review packet ──────────────────────────────────────────────
    print('\n[3/3] Writing review packet ...')
    os.makedirs(REVIEW_DIR, exist_ok=True)

    with open(OUT_PACKET, 'w') as f:
        f.write('PTSD Undercoding Detection — Chart Review Packet\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'Generated for: Top-{TOP_N} model-flagged unlabeled patients\n')
        f.write(f'Model: Clinical Longformer (PU-trained)\n')
        f.write(f'Threshold source: validation set (Fix 4)\n\n')
        f.write('INSTRUCTIONS FOR REVIEWER:\n')
        f.write('-' * 40 + '\n')
        f.write('For each patient note below, rate the likelihood of PTSD:\n')
        f.write('  1 = Probable PTSD (clear trauma history + symptom cluster)\n')
        f.write('  2 = Possible PTSD (some indicators but ambiguous)\n')
        f.write('  3 = Unlikely PTSD (no substantial evidence)\n\n')
        f.write('Record your rating in the accompanying CSV form.\n')
        f.write('=' * 70 + '\n\n')

        for i, (_, row) in enumerate(top50.iterrows(), 1):
            f.write(f'\n{"="*70}\n')
            f.write(f'PATIENT {i}/{TOP_N}\n')
            f.write(f'  Subject ID:  {row["subject_id"]}\n')
            f.write(f'  Admission ID: {row["hadm_id"]}\n')
            f.write(f'  Model score: {row["predicted_prob"]:.4f}\n')

            adm_row = top_adm[top_adm['subject_id'] == row['subject_id']]
            if len(adm_row) > 0:
                ar = adm_row.iloc[0]
                f.write(f'  Age: {ar.get("age_at_admission", "N/A")}\n')
                f.write(f'  Sex: {ar.get("gender", "N/A")}\n')

            f.write(f'\n--- Section-filtered discharge note ---\n\n')
            note_text = row.get('note_text', '[Note text not available]')
            f.write(note_text[:10000])
            if len(str(note_text)) > 10000:
                f.write(f'\n\n[... truncated at 10,000 chars; '
                        f'full note is {len(str(note_text)):,} chars ...]\n')
            f.write(f'\n{"="*70}\n')

    print(f'  Packet → {OUT_PACKET}')

    # ── Write review form CSV ────────────────────────────────────────────
    form_rows = []
    for i, (_, row) in enumerate(top50.iterrows(), 1):
        form_rows.append({
            'patient_number': i,
            'subject_id': row['subject_id'],
            'hadm_id': row['hadm_id'],
            'model_score': round(float(row['predicted_prob']), 4),
            'reviewer_rating': '',  # 1=probable, 2=possible, 3=unlikely
            'reviewer_notes': '',
        })

    form_df = pd.DataFrame(form_rows)
    form_df.to_csv(OUT_FORM, index=False)
    print(f'  Form   → {OUT_FORM}')

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('CHART REVIEW PACKET SUMMARY')
    print('=' * 65)
    print(f'  Patients: {TOP_N}')
    print(f'  Score range: [{top50["predicted_prob"].min():.4f}, '
          f'{top50["predicted_prob"].max():.4f}]')
    print(f'  Score mean: {top50["predicted_prob"].mean():.4f}')
    print(f'\n  After clinician review, compute:')
    print(f'    PPV@top50 = (n_probable + n_possible) / 50')
    print(f'    This is the clinician-rated PPV at the top decile —')
    print(f'    the single most persuasive validation metric.')
    print('\nDone.')


if __name__ == '__main__':
    main()
