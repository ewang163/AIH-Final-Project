"""
ewang163_ptsd_decision_curves.py
================================
Decision Curve Analysis (DCA) for the Clinical Longformer.

1. Runs Longformer inference on val + test sets (saves predictions to CSV)
2. Platt-scales probabilities using validation set
3. Recalibrates to 2% and 5% deployment prevalences
4. Computes net benefit curves for threshold range [0.01, 0.40]
5. Produces DCA plots (one per prevalence)

Outputs:
    ewang163_longformer_val_predictions.csv
    ewang163_longformer_test_predictions.csv
    ewang163_dca_results.csv
    ewang163_dca_2pct.png
    ewang163_dca_5pct.png

Submit via SLURM:
    sbatch ewang163_ptsd_decision_curves.sh
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_PREDICTIONS = f'{STUDENT_DIR}/results/predictions'
RESULTS_METRICS    = f'{STUDENT_DIR}/results/metrics'
RESULTS_FIGURES    = f'{STUDENT_DIR}/results/figures'

VAL_PARQUET   = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET  = f'{DATA_SPLITS}/ewang163_split_test.parquet'
LONGFORMER_DIR = f'{MODEL_DIR}/ewang163_longformer_best'

VAL_PRED_CSV  = f'{RESULTS_PREDICTIONS}/ewang163_longformer_val_predictions.csv'
TEST_PRED_CSV = f'{RESULTS_PREDICTIONS}/ewang163_longformer_test_predictions.csv'
DCA_CSV       = f'{RESULTS_METRICS}/ewang163_dca_results.csv'
DCA_2PCT_PNG  = f'{RESULTS_FIGURES}/ewang163_dca_2pct.png'
DCA_5PCT_PNG  = f'{RESULTS_FIGURES}/ewang163_dca_5pct.png'

MAX_LEN    = 4096
BATCH_SIZE = 4


# ── Dataset for transformer inference ────────────────────────────────────
class NoteDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


@torch.no_grad()
def run_inference(model, tokenizer, texts, device):
    """Run inference, return predicted probabilities (positive class)."""
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


def save_predictions(df, probs, out_path, label):
    """Save per-patient predictions to CSV."""
    pred_df = pd.DataFrame({
        'subject_id': df['subject_id'].values,
        'hadm_id': df['hadm_id'].values,
        'ptsd_label': df['ptsd_label'].values,
        'predicted_prob': np.round(probs, 6),
    })
    pred_df.to_csv(out_path, index=False)
    print(f'  Saved {label} predictions → {out_path}')
    return pred_df


# ── Net benefit computation ──────────────────────────────────────────────
def compute_net_benefit(labels, probs, threshold):
    """Net benefit at a single threshold."""
    N = len(labels)
    preds = (probs >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    nb = tp / N - fp / N * (threshold / (1 - threshold))
    return nb


def compute_treat_all_nb(labels, threshold):
    """Net benefit for treat-all strategy."""
    prevalence = labels.mean()
    nb = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
    return nb


def main():
    print('=' * 65)
    print('PTSD NLP — Decision Curve Analysis')
    print('=' * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Step 1: Load or generate predictions ─────────────────────────────
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    if os.path.exists(VAL_PRED_CSV) and os.path.exists(TEST_PRED_CSV):
        print('\n[1/5] Loading cached predictions ...')
        val_pred = pd.read_csv(VAL_PRED_CSV)
        test_pred = pd.read_csv(TEST_PRED_CSV)
        val_probs = val_pred['predicted_prob'].values
        test_probs = test_pred['predicted_prob'].values
        print(f'  Val: {len(val_pred):,} patients')
        print(f'  Test: {len(test_pred):,} patients')
    else:
        print('\n[1/5] Running Longformer inference on val + test sets ...')
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(LONGFORMER_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            LONGFORMER_DIR, num_labels=2
        )
        model.to(device)

        print('  Inference on validation set ...')
        val_probs = run_inference(model, tokenizer, val_df['note_text'].tolist(), device)
        val_pred = save_predictions(val_df, val_probs, VAL_PRED_CSV, 'val')

        print('  Inference on test set ...')
        test_probs = run_inference(model, tokenizer, test_df['note_text'].tolist(), device)
        test_pred = save_predictions(test_df, test_probs, TEST_PRED_CSV, 'test')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'  Inference completed in {time.time() - t0:.0f}s')

    val_labels = val_df['ptsd_label'].values.astype(int)
    test_labels = test_df['ptsd_label'].values.astype(int)
    print(f'  Test set: {len(test_labels):,} patients '
          f'({test_labels.sum():,} pos, {(test_labels == 0).sum():,} unlabeled)')

    # ── Step 2: Platt scaling on validation set ──────────────────────────
    print('\n[2/5] Platt scaling (fit on validation set) ...')
    platt = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
    platt.fit(val_probs.reshape(-1, 1), val_labels)

    val_cal = platt.predict_proba(val_probs.reshape(-1, 1))[:, 1]
    test_cal = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1]

    print(f'  Platt coef: {platt.coef_[0][0]:.4f}, '
          f'intercept: {platt.intercept_[0]:.4f}')
    print(f'  Raw  test probs: mean={test_probs.mean():.4f}, '
          f'median={np.median(test_probs):.4f}')
    print(f'  Calibrated test: mean={test_cal.mean():.4f}, '
          f'median={np.median(test_cal):.4f}')

    # Save calibrated predictions alongside raw
    test_pred_full = pd.DataFrame({
        'subject_id': test_df['subject_id'].values,
        'hadm_id': test_df['hadm_id'].values,
        'ptsd_label': test_labels,
        'predicted_prob': np.round(test_probs, 6),
        'calibrated_prob': np.round(test_cal, 6),
    })
    test_pred_full.to_csv(TEST_PRED_CSV, index=False)
    print(f'  Updated {TEST_PRED_CSV} with calibrated_prob column')

    # ── Step 3: Prevalence recalibration ─────────────────────────────────
    print('\n[3/5] Prevalence recalibration ...')
    p_study = test_labels.mean()
    print(f'  Study prevalence (test set): {p_study:.4f}')

    deploy_prevs = {'2pct': 0.02, '5pct': 0.05}

    recalibrated = {}
    for tag, p_deploy in deploy_prevs.items():
        cal_deploy = (test_cal * p_deploy / p_study) / \
            (test_cal * p_deploy / p_study +
             (1 - test_cal) * (1 - p_deploy) / (1 - p_study))
        recalibrated[tag] = cal_deploy
        print(f'  {tag}: mean recalibrated prob = {cal_deploy.mean():.4f}')

    # ── Step 4: Compute DCA net benefit curves ───────────────────────────
    print('\n[4/5] Computing decision curves ...')
    thresholds = np.arange(0.01, 0.405, 0.005)
    thresholds = np.round(thresholds, 4)

    dca_rows = []
    for t in thresholds:
        row = {'threshold': t}
        for tag, p_deploy in deploy_prevs.items():
            probs_deploy = recalibrated[tag]

            # Simulate labels at deployment prevalence using recalibrated probs
            # Net benefit is computed on the study population but with
            # prevalence-adjusted probabilities
            nb_model = compute_net_benefit(test_labels, probs_deploy, t)
            nb_treat_all = compute_treat_all_nb(test_labels, t)

            row[f'net_benefit_model_{tag}'] = round(nb_model, 6)
            row[f'net_benefit_treatall_{tag}'] = round(nb_treat_all, 6)
        dca_rows.append(row)

    dca_df = pd.DataFrame(dca_rows)
    dca_df.to_csv(DCA_CSV, index=False)
    print(f'  → {DCA_CSV}')

    # ── Step 5: DCA plots ────────────────────────────────────────────────
    print('\n[5/5] Generating DCA plots ...')

    for tag, p_deploy in deploy_prevs.items():
        pct_label = f'{int(p_deploy * 100)}%'
        fig, ax = plt.subplots(figsize=(8, 5))

        nb_model = dca_df[f'net_benefit_model_{tag}'].values
        nb_treat_all = dca_df[f'net_benefit_treatall_{tag}'].values
        ts = dca_df['threshold'].values

        ax.plot(ts, nb_model, 'b-', linewidth=2, label='Longformer')
        ax.plot(ts, nb_treat_all, 'r--', linewidth=1.5, label='Treat all')
        ax.axhline(y=0, color='k', linestyle=':', linewidth=1, label='Treat none')

        # Find threshold range where model beats treat-all
        model_better = nb_model > nb_treat_all
        if model_better.any():
            # Find contiguous ranges
            ranges = []
            in_range = False
            for i, better in enumerate(model_better):
                if better and not in_range:
                    start_idx = i
                    in_range = True
                elif not better and in_range:
                    ranges.append((ts[start_idx], ts[i - 1]))
                    in_range = False
            if in_range:
                ranges.append((ts[start_idx], ts[-1]))

            # Shade the region where model > treat-all
            for r_start, r_end in ranges:
                ax.axvspan(r_start, r_end, alpha=0.1, color='blue')

            range_str = ', '.join(f'{r[0]:.3f}–{r[1]:.3f}' for r in ranges)
            ax.annotate(
                f'Model > treat-all:\n{range_str}',
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9),
            )

        ax.set_xlabel('Threshold probability', fontsize=12)
        ax.set_ylabel('Net benefit', fontsize=12)
        ax.set_title(f'Decision Curve Analysis — {pct_label} deployment prevalence',
                     fontsize=13)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(0.01, 0.40)

        # Set y-axis limits sensibly
        all_vals = np.concatenate([nb_model, nb_treat_all, [0]])
        y_min = min(all_vals.min(), -0.02)
        y_max = max(all_vals.max(), 0.02) * 1.15
        ax.set_ylim(y_min, y_max)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_png = DCA_2PCT_PNG if tag == '2pct' else DCA_5PCT_PNG
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f'  → {out_png}')

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('DECISION CURVE ANALYSIS SUMMARY')
    print('=' * 65)

    for tag, p_deploy in deploy_prevs.items():
        pct_label = f'{int(p_deploy * 100)}%'
        nb_model = dca_df[f'net_benefit_model_{tag}'].values
        nb_treat_all = dca_df[f'net_benefit_treatall_{tag}'].values
        ts = dca_df['threshold'].values

        model_better = nb_model > nb_treat_all
        if model_better.any():
            better_ts = ts[model_better]
            print(f'\n  {pct_label} prevalence:')
            print(f'    Model beats treat-all for thresholds '
                  f'{better_ts.min():.3f} – {better_ts.max():.3f}')
            # Peak net benefit
            peak_idx = np.argmax(nb_model)
            print(f'    Peak net benefit: {nb_model[peak_idx]:.4f} '
                  f'at threshold {ts[peak_idx]:.3f}')
        else:
            print(f'\n  {pct_label} prevalence:')
            print(f'    Model does not beat treat-all in [{ts[0]:.2f}, {ts[-1]:.2f}]')

    print('\nDone.')


if __name__ == '__main__':
    main()
