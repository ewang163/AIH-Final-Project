"""
ewang163_ptsd_fairness.py
=========================
Fix 9: Statistically defensible fairness reporting.

Replaces per-subgroup AUPRC (unreliable at small n_pos) with:
  1. Calibration-in-the-large per subgroup (mean predicted - mean observed)
  2. Equal opportunity difference (recall gap at val-derived threshold)
  3. Bootstrap 95% CI on AUPRC (1,000 resamples; only report if CI width < 0.15)
  4. White vs. non-White primary fairness contrast

Inputs:
    results/predictions/ewang163_longformer_test_predictions.parquet (or regenerated)
    results/metrics/ewang163_evaluation_results.json (for val threshold)
    data/cohort/ewang163_ptsd_adm_extract.parquet (demographics)

Outputs:
    results/metrics/ewang163_fairness_results.csv

RUN:
    sbatch --partition=batch --mem=16G --time=1:00:00 \
           --output=logs/ewang163_fairness_%j.out \
           --wrap="python scripts/04_evaluation/ewang163_ptsd_fairness.py"
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
DATA_COHORT     = f'{STUDENT_DIR}/data/cohort'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'

TEST_PARQUET = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET  = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
EVAL_JSON    = f'{RESULTS_METRICS}/ewang163_evaluation_results.json'
OUT_CSV      = f'{RESULTS_METRICS}/ewang163_fairness_results.csv'

N_BOOTSTRAP = 1000
RANDOM_SEED = 42


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


def bootstrap_auprc(labels, probs, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    auprcs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y = labels[idx]
        p = probs[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        auprcs.append(average_precision_score(y, p))
    if not auprcs:
        return None, None, None
    auprcs = sorted(auprcs)
    point = average_precision_score(labels, probs)
    lo = auprcs[int(0.025 * len(auprcs))]
    hi = auprcs[int(0.975 * len(auprcs))]
    return round(point, 4), round(lo, 4), round(hi, 4)


def wilson_ci(p_hat, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0, 0)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (round(max(centre - spread, 0), 4), round(min(centre + spread, 1), 4))


def main():
    print('=' * 65)
    print('PTSD NLP — Fairness Analysis (Fix 9)')
    print('=' * 65, flush=True)

    bench = BenchmarkLogger()

    # ── Load data ────────────────────────────────────────────────────────
    print('\n[1/3] Loading data ...')

    with open(EVAL_JSON) as f:
        eval_results = json.load(f)

    if 'val_thresholds' in eval_results:
        threshold = eval_results['val_thresholds']['longformer']
    else:
        threshold = eval_results['longformer']['threshold_recall_85']
    print(f'  Operating threshold: {threshold:.4f}')

    test_df = pd.read_parquet(TEST_PARQUET)
    labels = test_df['ptsd_label'].values.astype(np.int64)

    # Regenerate Longformer probs from evaluation results or load predictions
    # For now, use the evaluation summary to identify we need the model
    # Actually, we need the per-patient predictions. Check if they exist.
    pred_path = f'{STUDENT_DIR}/results/predictions/ewang163_longformer_test_predictions.csv'
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        probs = pred_df['predicted_prob'].values
        print(f'  Loaded predictions from {pred_path}')
    else:
        print(f'  WARNING: {pred_path} not found.')
        print(f'  Please run evaluate.py first to generate test predictions.')
        print(f'  Using random scores as placeholder (results will be meaningless).')
        rng = np.random.RandomState(RANDOM_SEED)
        probs = rng.random(len(labels))

    adm = pd.read_parquet(ADM_PARQUET)

    # ── Build demographics ───────────────────────────────────────────────
    print('\n[2/3] Building demographics ...')
    test_hadms = set(test_df['hadm_id'].tolist())
    test_adm = adm[adm['hadm_id'].isin(test_hadms)].copy()
    test_adm = test_adm.sort_values('admittime').drop_duplicates('subject_id', keep='first')
    test_adm['race_cat'] = test_adm['race'].apply(map_race)
    test_adm['age_group'] = test_adm['age_at_admission'].apply(age_decade)
    test_adm['is_emergency'] = test_adm['admission_type'].str.upper().str.contains('EMER', na=False)

    demo_map = test_adm.set_index('subject_id')[
        ['gender', 'race_cat', 'age_group', 'is_emergency']
    ].to_dict('index')

    test_df['gender'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('gender', 'Unknown'))
    test_df['race_cat'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('race_cat', 'Other/Unknown'))
    test_df['age_group'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('age_group', 'Other'))
    test_df['is_emergency'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('is_emergency', False))

    # White vs non-White
    test_df['race_binary'] = test_df['race_cat'].apply(lambda r: 'White' if r == 'White' else 'Non-White')

    # ── Compute fairness metrics ─────────────────────────────────────────
    print('\n[3/3] Computing fairness metrics ...')
    results = []

    groupings = [
        ('sex', 'gender'),
        ('age_group', 'age_group'),
        ('race', 'race_cat'),
        ('race_binary', 'race_binary'),
        ('emergency', 'is_emergency'),
    ]

    preds = (probs >= threshold).astype(int)

    with bench.track('fairness', stage='full_analysis', device='cpu',
                     n_samples=len(test_df)):
        for group_name, col in groupings:
            for val in sorted(test_df[col].unique()):
                mask = (test_df[col] == val).values
                n = int(mask.sum())
                n_pos = int(labels[mask].sum())
                n_neg = n - n_pos

                if n_pos == 0 or n_pos == n:
                    results.append({
                        'group': group_name, 'value': str(val),
                        'n': n, 'n_pos': n_pos,
                        'calibration_in_large': None,
                        'cal_ci_lo': None, 'cal_ci_hi': None,
                        'recall_at_threshold': None,
                        'AUPRC': None, 'AUPRC_ci_lo': None, 'AUPRC_ci_hi': None,
                        'AUPRC_ci_width': None, 'AUPRC_reliable': False,
                    })
                    continue

                mean_pred = probs[mask].mean()
                mean_obs = labels[mask].mean()
                cal = round(float(mean_pred - mean_obs), 4)
                cal_ci = wilson_ci(mean_obs, n)

                tp_sub = ((preds[mask] == 1) & (labels[mask] == 1)).sum()
                fn_sub = ((preds[mask] == 0) & (labels[mask] == 1)).sum()
                recall_sub = tp_sub / (tp_sub + fn_sub) if (tp_sub + fn_sub) > 0 else 0

                auprc, ci_lo, ci_hi = bootstrap_auprc(labels[mask], probs[mask])
                ci_width = round(ci_hi - ci_lo, 4) if ci_lo is not None else None
                reliable = ci_width is not None and ci_width < 0.15

                results.append({
                    'group': group_name,
                    'value': str(val),
                    'n': n,
                    'n_pos': n_pos,
                    'calibration_in_large': cal,
                    'cal_ci_lo': cal_ci[0],
                    'cal_ci_hi': cal_ci[1],
                    'recall_at_threshold': round(float(recall_sub), 4),
                    'AUPRC': auprc,
                    'AUPRC_ci_lo': ci_lo,
                    'AUPRC_ci_hi': ci_hi,
                    'AUPRC_ci_width': ci_width,
                    'AUPRC_reliable': reliable,
                })

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)
    print(f'\nResults → {OUT_CSV}')

    # Equal opportunity difference (recall gap)
    print('\n' + '=' * 65)
    print('FAIRNESS SUMMARY')
    print('=' * 65)

    print(f'\n  Operating threshold: {threshold:.4f}')

    for group_name, col in groupings:
        grp_df = results_df[results_df['group'] == group_name]
        print(f'\n  {group_name.upper()}:')
        print(f'    {"Value":<20} {"n":>6} {"n_pos":>6} {"Recall":>8} '
              f'{"Cal-in-large":>14} {"AUPRC [95% CI]":>25} {"Reliable":>10}')
        for _, row in grp_df.iterrows():
            auprc_str = (f'{row["AUPRC"]:.4f} [{row["AUPRC_ci_lo"]:.4f}, {row["AUPRC_ci_hi"]:.4f}]'
                         if row['AUPRC'] is not None else 'N/A')
            recall_str = f'{row["recall_at_threshold"]:.4f}' if row['recall_at_threshold'] is not None else 'N/A'
            cal_str = f'{row["calibration_in_large"]:+.4f}' if row['calibration_in_large'] is not None else 'N/A'
            print(f'    {str(row["value"]):<20} {row["n"]:>6} {row["n_pos"]:>6} '
                  f'{recall_str:>8} {cal_str:>14} {auprc_str:>25} '
                  f'{"Yes" if row["AUPRC_reliable"] else "No":>10}')

        recalls = grp_df['recall_at_threshold'].dropna()
        if len(recalls) >= 2:
            eq_opp_diff = recalls.max() - recalls.min()
            print(f'    Equal opportunity difference: {eq_opp_diff:.4f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
