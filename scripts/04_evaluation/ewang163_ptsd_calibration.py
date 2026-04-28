"""
ewang163_ptsd_calibration.py
============================
Fit and evaluate probability calibration for the Clinical Longformer.

Kiryo PU loss does not produce calibrated probabilities. This script:
1. Loads val + test predictions
2. Fits Platt scaling on validation set, saves calibrator
3. Applies Elkan-Noto PU correction (Fix 5): divides calibrated probabilities
   by c = P(s=1|y=1), estimated as mean model probability on known positives,
   so predicted probabilities approximate P(PTSD=1) rather than P(coded=1)
4. Produces calibration curves (raw vs. Platt-scaled vs. Elkan-Noto) with 95% CI
5. Computes Expected Calibration Error (ECE) for all three variants

Fix 5 reference: Elkan & Noto (2008), "Learning classifiers from only positive
and unlabeled data". The constant c = P(s=1|y=1) is the probability that a true
positive is observed (coded) — the labeling frequency.

Outputs:
    ewang163_platt_calibrator.pkl
    ewang163_calibration_curve.png
    ewang163_calibration_results.csv

Submit via SLURM:
    sbatch ewang163_ptsd_calibration.sh
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

RESULTS_PREDICTIONS = f'{STUDENT_DIR}/results/predictions'
RESULTS_METRICS    = f'{STUDENT_DIR}/results/metrics'
RESULTS_FIGURES    = f'{STUDENT_DIR}/results/figures'
MODEL_DIR          = f'{STUDENT_DIR}/models'

VAL_PRED_CSV  = f'{RESULTS_PREDICTIONS}/ewang163_longformer_val_predictions.csv'
TEST_PRED_CSV = f'{RESULTS_PREDICTIONS}/ewang163_longformer_test_predictions.csv'

CALIBRATOR_PKL = f'{MODEL_DIR}/ewang163_platt_calibrator.pkl'
CURVE_PNG      = f'{RESULTS_FIGURES}/ewang163_calibration_curve.png'
RESULTS_CSV    = f'{RESULTS_METRICS}/ewang163_calibration_results.csv'

N_BINS = 10


def equal_frequency_bins(probs, labels, n_bins):
    """Bin predictions into equal-frequency deciles.
    Returns list of dicts with bin stats + 95% Wilson CI."""
    order = np.argsort(probs)
    probs_sorted = probs[order]
    labels_sorted = labels[order]

    bin_size = len(probs) // n_bins
    bins = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(probs)
        p = probs_sorted[start:end]
        l = labels_sorted[start:end]
        n = len(l)
        mean_pred = float(p.mean())
        obs_frac = float(l.mean())

        # Wilson score 95% CI for observed fraction
        z = 1.96
        denom = 1 + z**2 / n
        centre = (obs_frac + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((obs_frac * (1 - obs_frac) + z**2 / (4 * n)) / n) / denom
        ci_lo = max(0, centre - margin)
        ci_hi = min(1, centre + margin)

        bins.append({
            'bin': i + 1,
            'n': n,
            'mean_predicted': round(mean_pred, 4),
            'observed_fraction': round(obs_frac, 4),
            'ci_lower': round(ci_lo, 4),
            'ci_upper': round(ci_hi, 4),
        })
    return bins


def compute_ece(bins):
    """Expected Calibration Error from binned stats."""
    total_n = sum(b['n'] for b in bins)
    ece = sum(
        (b['n'] / total_n) * abs(b['mean_predicted'] - b['observed_fraction'])
        for b in bins
    )
    return ece


def main():
    print('=' * 65)
    print('PTSD NLP — Probability Calibration Analysis')
    print('=' * 65, flush=True)

    # ── Step 1: Load predictions ─────────────────────────────────────────
    print('\n[1/4] Loading predictions ...')
    val_df = pd.read_csv(VAL_PRED_CSV)
    test_df = pd.read_csv(TEST_PRED_CSV)

    val_probs = val_df['predicted_prob'].values
    val_labels = val_df['ptsd_label'].values.astype(int)
    test_probs = test_df['predicted_prob'].values
    test_labels = test_df['ptsd_label'].values.astype(int)

    print(f'  Val:  {len(val_df):,} patients '
          f'({val_labels.sum():,} pos, {(val_labels == 0).sum():,} unlabeled)')
    print(f'  Test: {len(test_df):,} patients '
          f'({test_labels.sum():,} pos, {(test_labels == 0).sum():,} unlabeled)')

    # ── Step 2: Fit Platt scaling ────────────────────────────────────────
    print('\n[2/4] Fitting Platt scaling on validation set ...')
    platt = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
    platt.fit(val_probs.reshape(-1, 1), val_labels)

    print(f'  Platt coef: {platt.coef_[0][0]:.4f}, '
          f'intercept: {platt.intercept_[0]:.4f}')

    with open(CALIBRATOR_PKL, 'wb') as f:
        pickle.dump(platt, f)
    print(f'  → {CALIBRATOR_PKL}')

    # Apply to test set
    test_cal = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1]

    print(f'  Raw  test: mean={test_probs.mean():.4f}, '
          f'median={np.median(test_probs):.4f}')
    print(f'  Platt test: mean={test_cal.mean():.4f}, '
          f'median={np.median(test_cal):.4f}')

    # ── Step 2b: Elkan-Noto PU correction (Fix 5) ───────────────────────
    print('\n[2b/5] Applying Elkan-Noto correction (Fix 5) ...')
    # Estimate c = P(s=1|y=1) = mean predicted probability on known positives
    # in the validation set (held-out from training)
    val_pos_mask = val_labels == 1
    c_estimate = float(val_probs[val_pos_mask].mean())
    print(f'  Elkan-Noto c = P(s=1|y=1) estimate: {c_estimate:.4f}')
    print(f'    (mean raw prob on {val_pos_mask.sum()} val positives)')

    # Corrected probabilities: P(y=1|x) = P(s=1|x) / c, clipped to [0,1]
    test_en = np.clip(test_cal / c_estimate, 0, 1)
    print(f'  Elkan-Noto test: mean={test_en.mean():.4f}, '
          f'median={np.median(test_en):.4f}')

    # ── Step 3: Calibration curves ───────────────────────────────────────
    print('\n[3/5] Computing calibration bins ...')
    bins_raw = equal_frequency_bins(test_probs, test_labels, N_BINS)
    bins_cal = equal_frequency_bins(test_cal, test_labels, N_BINS)
    bins_en = equal_frequency_bins(test_en, test_labels, N_BINS)

    ece_raw = compute_ece(bins_raw)
    ece_cal = compute_ece(bins_cal)
    ece_en = compute_ece(bins_en)

    print(f'  ECE (raw):          {ece_raw:.4f}')
    print(f'  ECE (Platt-scaled): {ece_cal:.4f}')
    print(f'  ECE (Elkan-Noto):   {ece_en:.4f}')

    # Plot
    print('\n[4/5] Generating calibration plot ...')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, bins, title, ece_val in [
        (axes[0], bins_raw, 'Raw Longformer', ece_raw),
        (axes[1], bins_cal, 'Platt-scaled', ece_cal),
        (axes[2], bins_en, f'Elkan-Noto (c={c_estimate:.3f})', ece_en),
    ]:
        mean_preds = [b['mean_predicted'] for b in bins]
        obs_fracs  = [b['observed_fraction'] for b in bins]
        ci_lo      = [b['ci_lower'] for b in bins]
        ci_hi      = [b['ci_upper'] for b in bins]
        yerr_lo    = [o - lo for o, lo in zip(obs_fracs, ci_lo)]
        yerr_hi    = [hi - o for o, hi in zip(obs_fracs, ci_hi)]

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
        ax.errorbar(
            mean_preds, obs_fracs,
            yerr=[yerr_lo, yerr_hi],
            fmt='o-', color='steelblue', capsize=4, linewidth=1.5,
            markersize=6, label='Observed',
        )
        ax.set_xlabel('Mean predicted probability', fontsize=11)
        ax.set_ylabel('Fraction positive', fontsize=11)
        ax.set_title(f'{title}\nECE = {ece_val:.4f}', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=150)
    plt.close(fig)
    print(f'  → {CURVE_PNG}')

    # ── Save results CSV ─────────────────────────────────────────────────
    print('\n[5/5] Saving results ...')
    rows = []
    for btype, bins in [('raw', bins_raw), ('platt_scaled', bins_cal),
                         ('elkan_noto', bins_en)]:
        for b in bins:
            rows.append({
                'type': btype,
                'bin': b['bin'],
                'n': b['n'],
                'mean_predicted': b['mean_predicted'],
                'observed_fraction': b['observed_fraction'],
                'ci_lower': b['ci_lower'],
                'ci_upper': b['ci_upper'],
            })
    for label, ece_val in [('raw', ece_raw), ('platt_scaled', ece_cal),
                            ('elkan_noto', ece_en)]:
        rows.append({
            'type': 'ece_summary', 'bin': label,
            'n': len(test_labels), 'mean_predicted': round(ece_val, 4),
            'observed_fraction': None, 'ci_lower': None, 'ci_upper': None,
        })
    rows.append({
        'type': 'elkan_noto_c', 'bin': 'c_estimate',
        'n': int(val_pos_mask.sum()), 'mean_predicted': round(c_estimate, 4),
        'observed_fraction': None, 'ci_lower': None, 'ci_upper': None,
    })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f'  → {RESULTS_CSV}')

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('CALIBRATION SUMMARY')
    print('=' * 65)
    print(f'\n  ECE before Platt scaling:  {ece_raw:.4f}')
    print(f'  ECE after Platt scaling:   {ece_cal:.4f}')
    print(f'  ECE after Elkan-Noto:      {ece_en:.4f}')
    print(f'  Elkan-Noto c estimate:     {c_estimate:.4f}')
    improvement = ece_raw - ece_en
    if improvement > 0:
        print(f'  Total improvement (raw→EN): {improvement:.4f} '
              f'({improvement / ece_raw * 100:.1f}% reduction)')
    else:
        print(f'  Note: Elkan-Noto correction did not improve ECE '
              f'(delta = {improvement:.4f})')
    print(f'\n  Interpretation: Elkan-Noto-corrected probabilities approximate '
          f'P(PTSD=1|text)\n  rather than P(coded=1|text). Use these for '
          f'clinical decision support.')

    print(f'\n  Raw calibration (10 equal-frequency bins):')
    print(f'  {"Bin":>4} {"N":>6} {"Mean pred":>10} {"Obs frac":>10} {"95% CI":>16}')
    print(f'  {"-"*4} {"-"*6} {"-"*10} {"-"*10} {"-"*16}')
    for b in bins_raw:
        print(f'  {b["bin"]:>4} {b["n"]:>6} {b["mean_predicted"]:>10.4f} '
              f'{b["observed_fraction"]:>10.4f} '
              f'[{b["ci_lower"]:.4f}, {b["ci_upper"]:.4f}]')

    print(f'\n  Platt-scaled calibration:')
    print(f'  {"Bin":>4} {"N":>6} {"Mean pred":>10} {"Obs frac":>10} {"95% CI":>16}')
    print(f'  {"-"*4} {"-"*6} {"-"*10} {"-"*10} {"-"*16}')
    for b in bins_cal:
        print(f'  {b["bin"]:>4} {b["n"]:>6} {b["mean_predicted"]:>10.4f} '
              f'{b["observed_fraction"]:>10.4f} '
              f'[{b["ci_lower"]:.4f}, {b["ci_upper"]:.4f}]')

    print('\nDone.')


if __name__ == '__main__':
    main()
