"""
ewang163_ptsd_pip_sweep_eval.py
===============================
Fix 2: Aggregates results from the pi_p sweep.  For each trained checkpoint
(one per pi_p value), runs proxy validation and picks the best pi_p by
proxy-vs-unlabeled Mann-Whitney AUC.

The proxy AUC is the only PU-uncontaminated metric available, so it is
the correct selection criterion.

Inputs:
    models/ewang163_longformer_best_pip{XX}/   — one per sweep value
    data/notes/ewang163_proxy_notes.parquet
    data/notes/ewang163_ptsd_corpus.parquet

Outputs:
    results/metrics/ewang163_pip_sweep_results.csv
    results/figures/ewang163_pip_sweep_proxy_auc.png

RUN:
    sbatch --partition=gpu --gres=gpu:1 --mem=32G --time=4:00:00 \
           --output=logs/ewang163_pip_sweep_eval_%j.out \
           --wrap="python scripts/03_training/ewang163_ptsd_pip_sweep_eval.py"
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
MODEL_DIR       = f'{STUDENT_DIR}/models'
DATA_NOTES      = f'{STUDENT_DIR}/data/notes'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'
RESULTS_FIGURES = f'{STUDENT_DIR}/results/figures'

PROXY_PARQUET  = f'{DATA_NOTES}/ewang163_proxy_notes.parquet'
CORPUS_PARQUET = f'{DATA_NOTES}/ewang163_ptsd_corpus.parquet'
SPLITS_JSON    = f'{DATA_SPLITS}/ewang163_split_subject_ids.json'
VAL_PARQUET    = f'{DATA_SPLITS}/ewang163_split_val.parquet'

OUT_CSV = f'{RESULTS_METRICS}/ewang163_pip_sweep_results.csv'
OUT_PNG = f'{RESULTS_FIGURES}/ewang163_pip_sweep_proxy_auc.png'

# Explicit checkpoint list: (checkpoint_suffix, label, pi_p_used_for_training)
# Covers all 7 sweep values + the empirical-pi retrain + the PULSNAR checkpoint
CHECKPOINTS = [
    ('_pip005', 'pi_p=0.05',   0.05),
    ('_pip008', 'pi_p=0.08',   0.08),
    ('_pip010', 'pi_p=0.10',   0.10),
    ('_pip012', 'pi_p=0.12',   0.12),
    ('_pip015', 'pi_p=0.15',   0.15),
    ('_pip020', 'pi_p=0.20',   0.20),
    ('_pip025', 'pi_p=0.25',   0.25),
    ('',        'retrain (empirical 0.398)', 0.398),
    ('_pulsnar', 'PULSNAR (alpha=0.196)', 0.196),
]
MAX_LEN = 4096
BATCH_SIZE = 4
N_UNLABELED_SAMPLE = 500
RANDOM_SEED = 42


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


def main():
    print('=' * 65)
    print('PTSD NLP — pi_p Sweep Evaluation (Fix 2)')
    print('=' * 65, flush=True)

    bench = BenchmarkLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    # ── Load proxy + unlabeled sample ────────────────────────────────────
    print('\n[1/3] Loading proxy and unlabeled notes ...')
    proxy_df = pd.read_parquet(PROXY_PARQUET)
    proxy_texts = proxy_df['note_text'].tolist()
    print(f'  Proxy: {len(proxy_texts)} notes')

    corpus = pd.read_parquet(CORPUS_PARQUET)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    train_sids = set(splits['train'])
    unlab = corpus[(corpus['group'] == 'unlabeled') &
                   (corpus['subject_id'].isin(train_sids))].copy()
    rng = np.random.RandomState(RANDOM_SEED)
    sample_sids = rng.choice(unlab['subject_id'].unique(),
                             size=min(N_UNLABELED_SAMPLE, unlab['subject_id'].nunique()),
                             replace=False)
    unlab_sample = unlab[unlab['subject_id'].isin(set(sample_sids))].drop_duplicates(
        'subject_id', keep='first')
    unlab_texts = unlab_sample['note_text'].tolist()
    print(f'  Unlabeled sample: {len(unlab_texts)} notes')

    # ── Load val set for AUPRC ───────────────────────────────────────────
    val_df = pd.read_parquet(VAL_PARQUET)
    val_texts = val_df['note_text'].tolist()
    val_labels = val_df['ptsd_label'].values

    # ── Evaluate each checkpoint ─────────────────────────────────────────
    print(f'\n[2/3] Evaluating {len(CHECKPOINTS)} checkpoints ...')
    results = []

    for suffix, label, pi_p in CHECKPOINTS:
        # Handle special suffixes: '' = retrain, '_pulsnar' = PULSNAR checkpoint
        if suffix == '_pulsnar':
            checkpoint_dir = f'{MODEL_DIR}/ewang163_longformer_pulsnar'
        else:
            checkpoint_dir = f'{MODEL_DIR}/ewang163_longformer_best{suffix}'

        if not os.path.isdir(checkpoint_dir):
            print(f'\n  {label}: checkpoint not found at {checkpoint_dir} — skipping')
            continue

        print(f'\n  {label} ({checkpoint_dir})')

        config_path = os.path.join(checkpoint_dir, 'training_config.json')
        best_val_auprc = None
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            best_val_auprc = cfg.get('best_val_auprc')
            print(f'    Training val AUPRC: {best_val_auprc}')

        with bench.track('pip_sweep_eval', stage=label,
                         device='gpu' if device.type == 'cuda' else 'cpu',
                         n_samples=len(proxy_texts) + len(unlab_texts) + len(val_texts)):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_dir, num_labels=2)
            model.to(device)

            proxy_probs = run_inference(model, tokenizer, proxy_texts, device)
            unlab_probs = run_inference(model, tokenizer, unlab_texts, device)
            val_probs = run_inference(model, tokenizer, val_texts, device)

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        U_stat, mw_p = mannwhitneyu(proxy_probs, unlab_probs, alternative='greater')
        proxy_auc = U_stat / (len(proxy_probs) * len(unlab_probs))
        val_auprc = average_precision_score(val_labels, val_probs)

        proxy_frac_above_05 = (proxy_probs >= 0.5).mean()
        unlab_frac_above_05 = (unlab_probs >= 0.5).mean()

        print(f'    Proxy AUC: {proxy_auc:.4f}  (MW p={mw_p:.2e})')
        print(f'    Val AUPRC (re-eval): {val_auprc:.4f}')
        print(f'    Proxy above 0.5: {proxy_frac_above_05:.1%}  '
              f'Unlab above 0.5: {unlab_frac_above_05:.1%}')

        results.append({
            'checkpoint_label': label,
            'checkpoint_suffix': suffix,
            'pi_p': pi_p,
            'proxy_auc': round(proxy_auc, 4),
            'mw_p_value': float(f'{mw_p:.4e}'),
            'val_auprc_reeval': round(val_auprc, 4),
            'val_auprc_training': best_val_auprc,
            'proxy_median_score': round(float(np.median(proxy_probs)), 4),
            'unlab_median_score': round(float(np.median(unlab_probs)), 4),
            'proxy_frac_above_05': round(float(proxy_frac_above_05), 4),
            'unlab_frac_above_05': round(float(unlab_frac_above_05), 4),
        })

    if not results:
        print('\nNo checkpoints found — sweep may not have completed yet.')
        return

    # ── Save + plot ──────────────────────────────────────────────────────
    print('\n[3/3] Saving results ...')
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)
    print(f'  Results → {OUT_CSV}')

    # Pick best by proxy AUC (must have MW p < 0.01 for validity)
    valid = results_df[results_df['mw_p_value'] < 0.01].copy()
    if len(valid) == 0:
        print('\n  WARNING: no checkpoint has MW p < 0.01; using all for ranking')
        valid = results_df

    best_idx = valid['proxy_auc'].idxmax()
    best_row = valid.loc[best_idx]
    print(f'\n  BEST checkpoint: {best_row["checkpoint_label"]}')
    print(f'    pi_p_training={best_row["pi_p"]}')
    print(f'    proxy AUC={best_row["proxy_auc"]:.4f} (MW p={best_row["mw_p_value"]:.2e})')
    print(f'    val AUPRC={best_row["val_auprc_reeval"]:.4f}')

    # Save winner as JSON for downstream scripts
    winner = {
        'checkpoint_label': best_row['checkpoint_label'],
        'checkpoint_suffix': best_row['checkpoint_suffix'],
        'pi_p': float(best_row['pi_p']),
        'proxy_auc': float(best_row['proxy_auc']),
        'mw_p_value': float(best_row['mw_p_value']),
        'val_auprc_reeval': float(best_row['val_auprc_reeval']),
    }
    winner_json = f'{RESULTS_METRICS}/ewang163_best_pi_p.json'
    with open(winner_json, 'w') as f:
        json.dump(winner, f, indent=2)
    print(f'    Winner saved → {winner_json}')

    # Plot — two panels: proxy AUC by model, val AUPRC by model
    sweep_df = results_df[results_df['checkpoint_suffix'].str.startswith('_pip')].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sweep-only pi_p vs proxy AUC
    if len(sweep_df) > 0:
        sweep_df_sorted = sweep_df.sort_values('pi_p')
        ax1 = axes[0]
        ax1.plot(sweep_df_sorted['pi_p'], sweep_df_sorted['proxy_auc'], 'o-b', label='Proxy AUC')
        ax1.set_xlabel('Class prior π_p (training-time)')
        ax1.set_ylabel('Proxy vs. Unlabeled AUC', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1b = ax1.twinx()
        ax1b.plot(sweep_df_sorted['pi_p'], sweep_df_sorted['val_auprc_reeval'], 's--r', label='Val AUPRC')
        ax1b.set_ylabel('Validation AUPRC', color='r')
        ax1b.tick_params(axis='y', labelcolor='r')
        ax1.set_title('Fix 2: π_p Sweep')
        ax1.grid(True, alpha=0.3)

    # Right: all checkpoints ranked by proxy AUC
    ax2 = axes[1]
    all_sorted = results_df.sort_values('proxy_auc', ascending=True)
    y_pos = np.arange(len(all_sorted))
    ax2.barh(y_pos, all_sorted['proxy_auc'], color='steelblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(all_sorted['checkpoint_label'])
    ax2.axvline(0.5, color='gray', linestyle=':', label='Chance')
    ax2.set_xlabel('Proxy vs. Unlabeled AUC')
    ax2.set_title('All 9 checkpoints ranked by proxy AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f'  Figure  → {OUT_PNG}')

    print('\n' + '=' * 65)
    print('CHECKPOINT COMPARISON — sorted by proxy AUC')
    print('=' * 65)
    show_cols = ['checkpoint_label', 'pi_p', 'proxy_auc', 'mw_p_value',
                 'val_auprc_reeval', 'proxy_median_score', 'unlab_median_score',
                 'proxy_frac_above_05', 'unlab_frac_above_05']
    sorted_df = results_df.sort_values('proxy_auc', ascending=False)
    print(sorted_df[show_cols].to_string(index=False))
    print(f'\nRecommended: {best_row["checkpoint_label"]} (suffix="{best_row["checkpoint_suffix"]}")')
    print('Done.')


if __name__ == '__main__':
    main()
