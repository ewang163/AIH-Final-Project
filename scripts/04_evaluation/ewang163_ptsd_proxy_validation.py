"""
ewang163_ptsd_proxy_validation.py
=================================
External validation using the pharmacological proxy group (n=163 patients,
102 with notes) that was held out from ALL training and evaluation.

Compares Longformer-predicted PTSD probability distributions:
  - Proxy group (prazosin + SSRI/SNRI overlap, no ICD PTSD code)
  - Random sample of 500 unlabeled patients from the training pool

If the model detects real PTSD signal (not just ICD-code artifacts),
proxy patients should score substantially higher than unlabeled patients.

Outputs:
    ewang163_proxy_validation.png           — box plot + histogram
    ewang163_proxy_validation_results.csv   — summary statistics

Submit via SLURM:
    sbatch ewang163_ptsd_proxy_validation.sh
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_NOTES         = f'{STUDENT_DIR}/data/notes'
DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_METRICS    = f'{STUDENT_DIR}/results/metrics'
RESULTS_FIGURES    = f'{STUDENT_DIR}/results/figures'

PROXY_PARQUET    = f'{DATA_NOTES}/ewang163_proxy_notes.parquet'
CORPUS_PARQUET   = f'{DATA_NOTES}/ewang163_ptsd_corpus.parquet'
SPLITS_JSON      = f'{DATA_SPLITS}/ewang163_split_subject_ids.json'
EVAL_JSON        = f'{RESULTS_METRICS}/ewang163_evaluation_results.json'
BEST_DIR         = f'{MODEL_DIR}/ewang163_longformer_best'

OUT_PNG = f'{RESULTS_FIGURES}/ewang163_proxy_validation.png'
OUT_CSV = f'{RESULTS_METRICS}/ewang163_proxy_validation_results.csv'

MAX_LEN    = 4096
BATCH_SIZE = 4
N_UNLABELED_SAMPLE = 500
RANDOM_SEED = 42


# ── Dataset ───────────────────────────────────────────────────────────────
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
    """Return predicted P(PTSD=1) for each text."""
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
    print('PTSD NLP — Proxy Group Validation')
    print('=' * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Load recall-0.85 threshold from evaluation results ────────────────
    print('\n[1/5] Loading evaluation threshold ...')
    try:
        with open(EVAL_JSON) as f:
            eval_results = json.load(f)
        threshold = eval_results['longformer']['threshold_recall_85']
        print(f'  Threshold (recall>=0.85 from test set): {threshold:.4f}')
    except (FileNotFoundError, KeyError):
        threshold = 0.5
        print(f'  WARNING: evaluation results not found, using default threshold=0.5')

    # ── Load proxy notes ──────────────────────────────────────────────────
    print('\n[2/5] Loading proxy and unlabeled notes ...')
    proxy_df = pd.read_parquet(PROXY_PARQUET)
    proxy_texts = proxy_df['note_text'].tolist()
    print(f'  Proxy patients: {proxy_df["subject_id"].nunique():,}, '
          f'notes: {len(proxy_df):,}')

    # Sample 500 unlabeled patients from training split
    corpus = pd.read_parquet(CORPUS_PARQUET)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    train_sids = set(splits['train'])

    unlab = corpus[(corpus['group'] == 'unlabeled') &
                   (corpus['subject_id'].isin(train_sids))].copy()
    print(f'  Unlabeled in train pool: {len(unlab):,}')

    rng = np.random.RandomState(RANDOM_SEED)
    unlab_patients = unlab['subject_id'].unique()
    sample_sids = rng.choice(unlab_patients,
                             size=min(N_UNLABELED_SAMPLE, len(unlab_patients)),
                             replace=False)
    unlab_sample = unlab[unlab['subject_id'].isin(set(sample_sids))].copy()
    # One note per patient (first)
    unlab_sample = unlab_sample.drop_duplicates('subject_id', keep='first')
    unlab_texts = unlab_sample['note_text'].tolist()
    print(f'  Unlabeled sample: {len(unlab_sample):,} patients')

    # ── Load model and run inference ──────────────────────────────────────
    print('\n[3/5] Loading Longformer and running inference ...')
    tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_DIR, num_labels=2
    )
    model.to(device)

    t0 = time.time()
    proxy_probs = run_inference(model, tokenizer, proxy_texts, device)
    t_proxy = time.time() - t0
    print(f'  Proxy inference: {len(proxy_probs)} notes ({t_proxy:.0f}s)')

    t0 = time.time()
    unlab_probs = run_inference(model, tokenizer, unlab_texts, device)
    t_unlab = time.time() - t0
    print(f'  Unlabeled inference: {len(unlab_probs)} notes ({t_unlab:.0f}s)')

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Statistical comparisons ───────────────────────────────────────────
    print('\n[4/5] Computing statistics ...')

    # Mann-Whitney U test
    U_stat, mw_p = mannwhitneyu(proxy_probs, unlab_probs, alternative='greater')
    # AUC = U / (n1 * n2) — probability that a random proxy patient scores
    # higher than a random unlabeled patient
    auc_proxy_vs_unlab = U_stat / (len(proxy_probs) * len(unlab_probs))

    # Fraction exceeding threshold
    proxy_above = (proxy_probs >= threshold).sum()
    unlab_above = (unlab_probs >= threshold).sum()
    proxy_frac = proxy_above / len(proxy_probs)
    unlab_frac = unlab_above / len(unlab_probs)

    print(f'  Proxy scores:     median={np.median(proxy_probs):.4f}, '
          f'mean={np.mean(proxy_probs):.4f}, '
          f'IQR=[{np.percentile(proxy_probs, 25):.4f}, '
          f'{np.percentile(proxy_probs, 75):.4f}]')
    print(f'  Unlabeled scores: median={np.median(unlab_probs):.4f}, '
          f'mean={np.mean(unlab_probs):.4f}, '
          f'IQR=[{np.percentile(unlab_probs, 25):.4f}, '
          f'{np.percentile(unlab_probs, 75):.4f}]')
    print(f'\n  Mann-Whitney U: U={U_stat:.0f}, p={mw_p:.2e}')
    print(f'  AUC (proxy vs unlabeled): {auc_proxy_vs_unlab:.4f}')
    print(f'\n  Threshold = {threshold:.4f}')
    print(f'  Proxy above threshold:     {proxy_above}/{len(proxy_probs)} '
          f'({proxy_frac:.1%})')
    print(f'  Unlabeled above threshold: {unlab_above}/{len(unlab_probs)} '
          f'({unlab_frac:.1%})')

    # ── Save results CSV ──────────────────────────────────────────────────
    rows = [
        {
            'group': 'Proxy',
            'n': len(proxy_probs),
            'median_score': round(float(np.median(proxy_probs)), 4),
            'mean_score': round(float(np.mean(proxy_probs)), 4),
            'q25': round(float(np.percentile(proxy_probs, 25)), 4),
            'q75': round(float(np.percentile(proxy_probs, 75)), 4),
            'frac_above_threshold': round(float(proxy_frac), 4),
            'n_above_threshold': int(proxy_above),
        },
        {
            'group': 'Unlabeled sample',
            'n': len(unlab_probs),
            'median_score': round(float(np.median(unlab_probs)), 4),
            'mean_score': round(float(np.mean(unlab_probs)), 4),
            'q25': round(float(np.percentile(unlab_probs, 25)), 4),
            'q75': round(float(np.percentile(unlab_probs, 75)), 4),
            'frac_above_threshold': round(float(unlab_frac), 4),
            'n_above_threshold': int(unlab_above),
        },
        {
            'group': 'Comparison statistics',
            'n': None,
            'median_score': None,
            'mean_score': None,
            'q25': None,
            'q75': None,
            'frac_above_threshold': None,
            'n_above_threshold': None,
        },
    ]
    results_df = pd.DataFrame(rows)
    # Add stats as extra columns to the third row
    results_df.loc[2, 'group'] = (
        f'Mann-Whitney U={U_stat:.0f}, p={mw_p:.2e}, '
        f'AUC={auc_proxy_vs_unlab:.4f}, threshold={threshold:.4f}'
    )
    results_df.to_csv(OUT_CSV, index=False)
    print(f'\n  Results → {OUT_CSV}')

    # ── Plot ──────────────────────────────────────────────────────────────
    print('\n[5/5] Generating plots ...')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: box plot
    ax = axes[0]
    bp = ax.boxplot([proxy_probs, unlab_probs],
                    labels=['Proxy\n(n={})'.format(len(proxy_probs)),
                            'Unlabeled\n(n={})'.format(len(unlab_probs))],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#E8747C')
    bp['boxes'][1].set_facecolor('#7CA1E8')
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Threshold ({threshold:.3f})')
    ax.set_ylabel('Predicted P(PTSD)')
    ax.set_title('Score Distribution: Proxy vs. Unlabeled')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    # Right: overlapping histograms
    ax = axes[1]
    bins = np.linspace(0, 1, 31)
    ax.hist(proxy_probs, bins=bins, alpha=0.6, color='#E8747C',
            label=f'Proxy (n={len(proxy_probs)})', density=True, edgecolor='white')
    ax.hist(unlab_probs, bins=bins, alpha=0.6, color='#7CA1E8',
            label=f'Unlabeled (n={len(unlab_probs)})', density=True, edgecolor='white')
    ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Threshold ({threshold:.3f})')
    ax.set_xlabel('Predicted P(PTSD)')
    ax.set_ylabel('Density')
    ax.set_title('Score Histograms')
    ax.legend(fontsize=9)

    fig.suptitle(
        f'Proxy Validation — Mann-Whitney p={mw_p:.2e}, '
        f'AUC={auc_proxy_vs_unlab:.3f}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot → {OUT_PNG}')

    # ── Interpretation ────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('INTERPRETATION')
    print('=' * 65)
    if auc_proxy_vs_unlab > 0.65 and mw_p < 0.01:
        print('  The proxy group scores significantly higher than unlabeled')
        print('  patients (AUC={:.3f}, p={:.2e}). This provides non-circular'.format(
            auc_proxy_vs_unlab, mw_p))
        print('  evidence that the model detects real PTSD clinical signal,')
        print('  not just ICD-coding artifacts.')
    elif auc_proxy_vs_unlab > 0.55:
        print('  Modest separation between proxy and unlabeled groups.')
        print('  Consistent with some true PTSD detection, but the proxy')
        print('  group definition has ~15-20% expected false positive rate.')
    else:
        print('  Weak or no separation. The model may not generalize well')
        print('  to PTSD cases that are not ICD-coded.')

    print(f'\n  {proxy_frac:.0%} of proxy patients exceed the recall=0.85 threshold')
    print(f'  vs. {unlab_frac:.0%} of unlabeled patients.')

    print('\nDone.')


if __name__ == '__main__':
    main()
