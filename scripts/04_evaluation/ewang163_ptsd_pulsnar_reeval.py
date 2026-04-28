"""
ewang163_ptsd_pulsnar_reeval.py
==============================
Dual-winner reporting: re-run Fixes 5, 6, 9 on the PULSNAR checkpoint.

Fix 5 — calibration (ECE raw / Platt / Elkan-Noto)
Fix 6 — clinical utility (LR+, LR-, DOR, NNS, alert rate, workup reduction)
Fix 9 — fairness (calibration in large, EO diff, bootstrap CI AUPRC)

Outputs (all suffixed `_pulsnar`):
  results/predictions/ewang163_longformer_val_predictions_pulsnar.csv
  results/predictions/ewang163_longformer_test_predictions_pulsnar.csv
  results/metrics/ewang163_evaluation_results_pulsnar.json
  results/metrics/ewang163_calibration_results_pulsnar.csv
  results/metrics/ewang163_fairness_results_pulsnar.csv
  results/figures/ewang163_calibration_curve_pulsnar.png

RUN:
    sbatch --partition=gpu --gres=gpu:1 --mem=32G --time=2:00:00 \
           --output=logs/ewang163_pulsnar_reeval_%j.out \
           --wrap="python scripts/04_evaluation/ewang163_ptsd_pulsnar_reeval.py"
"""

import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
DATA_COHORT     = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR       = f'{STUDENT_DIR}/models'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'
RESULTS_PRED    = f'{STUDENT_DIR}/results/predictions'
RESULTS_FIG     = f'{STUDENT_DIR}/results/figures'

VAL_PARQUET   = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET  = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET   = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
PULSNAR_MODEL = f'{MODEL_DIR}/ewang163_longformer_pulsnar'

SUFFIX = '_pulsnar'
VAL_PRED_CSV  = f'{RESULTS_PRED}/ewang163_longformer_val_predictions{SUFFIX}.csv'
TEST_PRED_CSV = f'{RESULTS_PRED}/ewang163_longformer_test_predictions{SUFFIX}.csv'
EVAL_JSON     = f'{RESULTS_METRICS}/ewang163_evaluation_results{SUFFIX}.json'
CAL_CSV       = f'{RESULTS_METRICS}/ewang163_calibration_results{SUFFIX}.csv'
CAL_PNG       = f'{RESULTS_FIG}/ewang163_calibration_curve{SUFFIX}.png'
FAIR_CSV      = f'{RESULTS_METRICS}/ewang163_fairness_results{SUFFIX}.csv'

MAX_LEN    = 4096
BATCH_SIZE = 4
N_BINS = 10
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


# ── Inference helpers ─────────────────────────────────────────────────────
class NoteDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


@torch.no_grad()
def run_inference(model, tokenizer, texts, device):
    ds = NoteDataset(texts, tokenizer, MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


def threshold_at_recall(probs, labels, target_recall=0.85):
    for t in np.linspace(1.0, 0.0, 1001):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if recall >= target_recall:
            return float(t)
    return 0.0


def compute_metrics(probs, labels, thresh):
    preds = (probs >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return {
        'AUPRC': round(average_precision_score(labels, probs), 4),
        'AUROC': round(roc_auc_score(labels, probs), 4),
        'threshold': round(thresh, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'precision': round(prec, 4),
        'F1': round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_total': len(labels),
        'n_pos': int(labels.sum()),
        'pos_rate': round(float(labels.mean()), 4),
    }


# ── Fix 6: clinical utility ───────────────────────────────────────────────
def clinical_utility(sens, spec, alert_rate, prevalences=(0.01, 0.02, 0.05, 0.10, 0.20)):
    """Recalibrated PPV/NPV/NNS at deployment prevalences + LR+/LR-/DOR."""
    eps = 1e-9
    lr_pos = sens / (1 - spec + eps)
    lr_neg = (1 - sens) / (spec + eps)
    dor = (sens / (1 - sens + eps)) / ((1 - spec + eps) / (spec + eps))
    rows = []
    for prev in prevalences:
        ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev) + eps)
        npv = (spec * (1 - prev)) / ((1 - sens) * prev + spec * (1 - prev) + eps)
        nns = 1 / ppv if ppv > 0 else float('inf')
        rows.append({
            'prevalence': prev,
            'PPV': round(ppv, 4),
            'NPV': round(npv, 4),
            'NNS': round(nns, 2),
        })
    return {
        'LR_positive': round(lr_pos, 4),
        'LR_negative': round(lr_neg, 4),
        'DOR': round(dor, 2),
        'alert_rate': round(alert_rate, 4),
        'workup_reduction': round(1 - alert_rate, 4),
        'by_prevalence': rows,
    }


# ── Fix 5: calibration ────────────────────────────────────────────────────
def equal_frequency_bins(probs, labels, n_bins):
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
        z = 1.96
        denom = 1 + z**2 / n
        centre = (obs_frac + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((obs_frac * (1 - obs_frac) + z**2 / (4 * n)) / n) / denom
        bins.append({
            'bin': i + 1, 'n': n,
            'mean_predicted': round(mean_pred, 4),
            'observed_fraction': round(obs_frac, 4),
            'ci_lower': round(max(0, centre - margin), 4),
            'ci_upper': round(min(1, centre + margin), 4),
        })
    return bins


def compute_ece(bins):
    total_n = sum(b['n'] for b in bins)
    return sum(
        (b['n'] / total_n) * abs(b['mean_predicted'] - b['observed_fraction'])
        for b in bins
    )


# ── Fix 9: fairness helpers ───────────────────────────────────────────────
def map_race(race_str):
    if pd.isna(race_str):
        return 'Other/Unknown'
    r = str(race_str).upper()
    if 'WHITE' in r: return 'White'
    if 'BLACK' in r or 'AFRICAN' in r: return 'Black'
    if 'HISPANIC' in r or 'LATINO' in r: return 'Hispanic'
    if 'ASIAN' in r: return 'Asian'
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
    if n == 0: return (0, 0)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (round(max(centre - spread, 0), 4), round(min(centre + spread, 1), 4))


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('PTSD NLP — PULSNAR Re-Evaluation (Fixes 5, 6, 9)')
    print('=' * 65, flush=True)

    bench = BenchmarkLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    print(f'Model:  {PULSNAR_MODEL}')

    os.makedirs(RESULTS_PRED, exist_ok=True)
    os.makedirs(RESULTS_METRICS, exist_ok=True)
    os.makedirs(RESULTS_FIG, exist_ok=True)

    # ── Step 1: Load data + model ────────────────────────────────────────
    print('\n[1/6] Loading data + PULSNAR model ...')
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f'  Val:  n={len(val_df)} pos_rate={val_df["ptsd_label"].mean():.3f}')
    print(f'  Test: n={len(test_df)} pos_rate={test_df["ptsd_label"].mean():.3f}')

    tokenizer = AutoTokenizer.from_pretrained(PULSNAR_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        PULSNAR_MODEL, num_labels=2)
    model.to(device)

    # ── Step 2: Inference on val + test ──────────────────────────────────
    print('\n[2/6] Running PULSNAR inference ...')
    val_labels = val_df['ptsd_label'].values.astype(np.int64)
    test_labels = test_df['ptsd_label'].values.astype(np.int64)

    with bench.track('pulsnar_reeval', stage='val_inference', device='gpu',
                     n_samples=len(val_df)):
        val_probs = run_inference(model, tokenizer, val_df['note_text'].tolist(), device)
    with bench.track('pulsnar_reeval', stage='test_inference', device='gpu',
                     n_samples=len(test_df)):
        test_probs = run_inference(model, tokenizer, test_df['note_text'].tolist(), device)

    pd.DataFrame({
        'subject_id': val_df['subject_id'].values,
        'ptsd_label': val_labels,
        'predicted_prob': val_probs,
    }).to_csv(VAL_PRED_CSV, index=False)
    pd.DataFrame({
        'subject_id': test_df['subject_id'].values,
        'ptsd_label': test_labels,
        'predicted_prob': test_probs,
    }).to_csv(TEST_PRED_CSV, index=False)
    print(f'  → {VAL_PRED_CSV}')
    print(f'  → {TEST_PRED_CSV}')

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Step 3: Threshold + base metrics + Fix 6 utility ─────────────────
    print('\n[3/6] Computing val-derived threshold + metrics + utility ...')
    val_thresh = threshold_at_recall(val_probs, val_labels, 0.85)
    val_metrics = compute_metrics(val_probs, val_labels, val_thresh)
    test_metrics = compute_metrics(test_probs, test_labels, val_thresh)
    alert_rate = float((test_probs >= val_thresh).mean())
    utility = clinical_utility(test_metrics['sensitivity'],
                               test_metrics['specificity'], alert_rate)

    print(f'  Val-derived threshold (recall>=0.85): {val_thresh:.4f}')
    print(f'  Val:  AUPRC={val_metrics["AUPRC"]:.4f} AUROC={val_metrics["AUROC"]:.4f}')
    print(f'  Test: AUPRC={test_metrics["AUPRC"]:.4f} AUROC={test_metrics["AUROC"]:.4f}')
    print(f'        sens={test_metrics["sensitivity"]:.3f} spec={test_metrics["specificity"]:.3f} '
          f'F1={test_metrics["F1"]:.3f}')
    print(f'  LR+ = {utility["LR_positive"]:.3f} | LR- = {utility["LR_negative"]:.3f} '
          f'| DOR = {utility["DOR"]:.2f}')
    print(f'  Alert rate = {utility["alert_rate"]:.4f} '
          f'| Workup reduction = {utility["workup_reduction"]:.4f}')
    for r in utility['by_prevalence']:
        print(f'    Prev={r["prevalence"]:.2f}  PPV={r["PPV"]:.4f}  '
              f'NPV={r["NPV"]:.4f}  NNS={r["NNS"]:.2f}')

    # ── Step 4: Fix 5 calibration ────────────────────────────────────────
    print('\n[4/6] Calibration analysis (Fix 5) ...')
    platt = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
    platt.fit(val_probs.reshape(-1, 1), val_labels)
    test_cal = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1]

    val_pos_mask = val_labels == 1
    c_estimate = float(val_probs[val_pos_mask].mean())
    test_en = np.clip(test_cal / c_estimate, 0, 1)

    bins_raw = equal_frequency_bins(test_probs, test_labels, N_BINS)
    bins_cal = equal_frequency_bins(test_cal, test_labels, N_BINS)
    bins_en  = equal_frequency_bins(test_en,  test_labels, N_BINS)
    ece_raw = compute_ece(bins_raw)
    ece_cal = compute_ece(bins_cal)
    ece_en  = compute_ece(bins_en)

    print(f'  Elkan-Noto c = {c_estimate:.4f}')
    print(f'  ECE raw          = {ece_raw:.4f}')
    print(f'  ECE Platt-scaled = {ece_cal:.4f}')
    print(f'  ECE Elkan-Noto   = {ece_en:.4f}')

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, bins, title, ece_val in [
        (axes[0], bins_raw, 'Raw PULSNAR', ece_raw),
        (axes[1], bins_cal, 'Platt-scaled', ece_cal),
        (axes[2], bins_en,  f'Elkan-Noto (c={c_estimate:.3f})', ece_en),
    ]:
        mp = [b['mean_predicted'] for b in bins]
        of = [b['observed_fraction'] for b in bins]
        lo = [o - b['ci_lower'] for o, b in zip(of, bins)]
        hi = [b['ci_upper'] - o for o, b in zip(of, bins)]
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
        ax.errorbar(mp, of, yerr=[lo, hi], fmt='o-', color='steelblue',
                    capsize=4, linewidth=1.5, markersize=6, label='Observed')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction positive')
        ax.set_title(f'{title}\nECE = {ece_val:.4f}')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CAL_PNG, dpi=150)
    plt.close(fig)
    print(f'  → {CAL_PNG}')

    cal_rows = []
    for btype, bs in [('raw', bins_raw), ('platt_scaled', bins_cal),
                      ('elkan_noto', bins_en)]:
        for b in bs:
            cal_rows.append({'type': btype, **b})
    for label, ece_val in [('raw', ece_raw), ('platt_scaled', ece_cal),
                           ('elkan_noto', ece_en)]:
        cal_rows.append({'type': 'ece_summary', 'bin': label,
                         'n': len(test_labels),
                         'mean_predicted': round(ece_val, 4),
                         'observed_fraction': None,
                         'ci_lower': None, 'ci_upper': None})
    cal_rows.append({'type': 'elkan_noto_c', 'bin': 'c_estimate',
                     'n': int(val_pos_mask.sum()),
                     'mean_predicted': round(c_estimate, 4),
                     'observed_fraction': None,
                     'ci_lower': None, 'ci_upper': None})
    pd.DataFrame(cal_rows).to_csv(CAL_CSV, index=False)
    print(f'  → {CAL_CSV}')

    # ── Step 5: Fix 9 fairness ──────────────────────────────────────────
    print('\n[5/6] Fairness analysis (Fix 9) ...')
    adm = pd.read_parquet(ADM_PARQUET)
    test_hadms = set(test_df['hadm_id'].tolist())
    test_adm = adm[adm['hadm_id'].isin(test_hadms)].copy()
    test_adm = test_adm.sort_values('admittime').drop_duplicates(
        'subject_id', keep='first')
    test_adm['race_cat'] = test_adm['race'].apply(map_race)
    test_adm['age_group'] = test_adm['age_at_admission'].apply(age_decade)
    test_adm['is_emergency'] = test_adm['admission_type'].str.upper().str.contains('EMER', na=False)

    demo_map = test_adm.set_index('subject_id')[
        ['gender', 'race_cat', 'age_group', 'is_emergency']
    ].to_dict('index')

    test_df = test_df.copy()
    test_df['gender'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('gender', 'Unknown'))
    test_df['race_cat'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('race_cat', 'Other/Unknown'))
    test_df['age_group'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('age_group', 'Other'))
    test_df['is_emergency'] = test_df['subject_id'].map(lambda s: demo_map.get(s, {}).get('is_emergency', False))
    test_df['race_binary'] = test_df['race_cat'].apply(lambda r: 'White' if r == 'White' else 'Non-White')

    preds = (test_probs >= val_thresh).astype(int)
    groupings = [
        ('sex', 'gender'), ('age_group', 'age_group'),
        ('race', 'race_cat'), ('race_binary', 'race_binary'),
        ('emergency', 'is_emergency'),
    ]

    fair_rows = []
    with bench.track('pulsnar_reeval', stage='fairness', device='cpu',
                     n_samples=len(test_df)):
        for gname, col in groupings:
            for val in sorted(test_df[col].unique()):
                mask = (test_df[col] == val).values
                n = int(mask.sum())
                n_pos = int(test_labels[mask].sum())
                if n_pos == 0 or n_pos == n:
                    fair_rows.append({
                        'group': gname, 'value': str(val), 'n': n, 'n_pos': n_pos,
                        'calibration_in_large': None,
                        'cal_ci_lo': None, 'cal_ci_hi': None,
                        'recall_at_threshold': None,
                        'AUPRC': None, 'AUPRC_ci_lo': None, 'AUPRC_ci_hi': None,
                        'AUPRC_ci_width': None, 'AUPRC_reliable': False,
                    })
                    continue
                mean_pred = test_probs[mask].mean()
                mean_obs = test_labels[mask].mean()
                cal = round(float(mean_pred - mean_obs), 4)
                cal_ci = wilson_ci(mean_obs, n)
                tp_sub = ((preds[mask] == 1) & (test_labels[mask] == 1)).sum()
                fn_sub = ((preds[mask] == 0) & (test_labels[mask] == 1)).sum()
                recall_sub = tp_sub / (tp_sub + fn_sub) if (tp_sub + fn_sub) > 0 else 0
                auprc, lo, hi = bootstrap_auprc(test_labels[mask], test_probs[mask])
                ci_w = round(hi - lo, 4) if lo is not None else None
                fair_rows.append({
                    'group': gname, 'value': str(val), 'n': n, 'n_pos': n_pos,
                    'calibration_in_large': cal,
                    'cal_ci_lo': cal_ci[0], 'cal_ci_hi': cal_ci[1],
                    'recall_at_threshold': round(float(recall_sub), 4),
                    'AUPRC': auprc, 'AUPRC_ci_lo': lo, 'AUPRC_ci_hi': hi,
                    'AUPRC_ci_width': ci_w,
                    'AUPRC_reliable': ci_w is not None and ci_w < 0.15,
                })

    fair_df = pd.DataFrame(fair_rows)
    fair_df.to_csv(FAIR_CSV, index=False)
    print(f'  → {FAIR_CSV}')

    print('\n  Equal-opportunity differences (recall gap at threshold):')
    for gname, _ in groupings:
        recalls = fair_df[fair_df['group'] == gname]['recall_at_threshold'].dropna()
        if len(recalls) >= 2:
            eo_diff = recalls.max() - recalls.min()
            print(f'    {gname:<14}  EO diff = {eo_diff:.4f}  '
                  f'(min={recalls.min():.3f}, max={recalls.max():.3f})')

    # ── Step 6: Save consolidated JSON ───────────────────────────────────
    print('\n[6/6] Saving consolidated results JSON ...')
    out = {
        'model': 'Clinical Longformer (PULSNAR, alpha=0.1957)',
        'checkpoint': PULSNAR_MODEL,
        'val_threshold': val_thresh,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'utility': utility,
        'calibration': {
            'ece_raw': round(ece_raw, 4),
            'ece_platt': round(ece_cal, 4),
            'ece_elkan_noto': round(ece_en, 4),
            'elkan_noto_c': round(c_estimate, 4),
        },
        'fairness': {
            gname: {
                'eo_diff': round(
                    float(fair_df[fair_df['group'] == gname]['recall_at_threshold']
                          .dropna().max()
                          - fair_df[fair_df['group'] == gname]['recall_at_threshold']
                          .dropna().min()),
                    4)
                if len(fair_df[fair_df['group'] == gname]['recall_at_threshold']
                       .dropna()) >= 2 else None
            }
            for gname, _ in groupings
        },
    }
    with open(EVAL_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'  → {EVAL_JSON}')

    print('\nDone.')


if __name__ == '__main__':
    main()
