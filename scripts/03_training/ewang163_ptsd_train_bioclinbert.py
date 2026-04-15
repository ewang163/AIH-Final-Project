"""
ewang163_ptsd_train_bioclinbert.py
==================================
Fine-tunes BioClinicalBERT (Alsentzer et al. 2019) with Kiryo PU loss,
as a direct comparison to the Clinical Longformer.

Key difference: max_length=512 (vs. Longformer's 4096). Discharge notes
are truncated to the first 512 tokens. This is a known limitation.

Same training setup otherwise:
  - Kiryo et al. (2017) non-negative PU loss
  - AdamW, lr=2e-5, weight_decay=0.01, warmup=10%
  - 5 epochs, best checkpoint by val AUPRC

Outputs:
    ewang163_bioclinbert_best/               — best checkpoint
    ewang163_bioclinbert_training_log.csv    — epoch-level metrics
    ewang163_bioclinbert_eval_results.json   — test set evaluation

Submit via SLURM:
    sbatch ewang163_ptsd_train_bioclinbert.sh
"""

import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import average_precision_score, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR      = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS      = f'{STUDENT_DIR}/data/splits'
MODEL_DIR        = f'{STUDENT_DIR}/models'
RESULTS_METRICS  = f'{STUDENT_DIR}/results/metrics'
RESULTS_PREDS    = f'{STUDENT_DIR}/results/predictions'

TRAIN_PARQUET = f'{DATA_SPLITS}/ewang163_split_train.parquet'
VAL_PARQUET   = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET  = f'{DATA_SPLITS}/ewang163_split_test.parquet'

BEST_DIR    = f'{MODEL_DIR}/ewang163_bioclinbert_best'
LOG_CSV     = f'{RESULTS_METRICS}/ewang163_bioclinbert_training_log.csv'
EVAL_JSON   = f'{RESULTS_METRICS}/ewang163_bioclinbert_eval_results.json'

# Longformer predictions for McNemar comparison
LF_PRED_CSV = f'{RESULTS_PREDS}/ewang163_longformer_test_predictions.csv'

MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'

# ── Hyperparameters ──────────────────────────────────────────────────────
MAX_LEN          = 512
BATCH_SIZE       = 16
GRAD_ACCUM_STEPS = 2     # effective batch size = 16 * 2 = 32
LR               = 2e-5
EPOCHS           = 5
WARMUP_FRAC      = 0.1
WEIGHT_DECAY     = 0.01


# ── PU Loss: Kiryo et al. (2017) — identical to Longformer script ───────
def pu_loss(logits, labels, pi_p):
    """
    R_pu(f) = pi_p * E_P[l(f(x), +1)]
            + max(0, E_U[l(f(x), -1)] - pi_p * E_P[l(f(x), -1)])
    """
    pos_mask = (labels == 1)
    unl_mask = (labels == 0)

    n_pos = pos_mask.sum()
    n_unl = unl_mask.sum()

    if n_pos == 0 or n_unl == 0:
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    loss_pos = F.binary_cross_entropy_with_logits(
        logits[pos_mask], torch.ones_like(logits[pos_mask]), reduction='mean'
    )
    loss_unl = F.binary_cross_entropy_with_logits(
        logits[unl_mask], torch.zeros_like(logits[unl_mask]), reduction='mean'
    )
    loss_pos_as_neg = F.binary_cross_entropy_with_logits(
        logits[pos_mask], torch.zeros_like(logits[pos_mask]), reduction='mean'
    )

    pu = pi_p * loss_pos + torch.clamp(
        loss_unl - pi_p * loss_pos_as_neg, min=0.0
    )
    return pu


# ── Dataset ──────────────────────────────────────────────────────────────
class NoteDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df['note_text'].tolist()
        self.labels = df['ptsd_label'].values.astype(np.int64)
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
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training loop ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, pi_p, device,
                    grad_accum_steps):
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, 1] - outputs.logits[:, 0]
            loss = pu_loss(logits.float(), labels, pi_p) / grad_accum_steps

        scaler.scale(loss).backward()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            avg = total_loss / n_batches
            print(f'    step {step+1}/{len(loader)}  avg_loss={avg:.4f}',
                  flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auprc = average_precision_score(all_labels, all_probs)
    return auprc, all_probs, all_labels


# ── Evaluation metrics ───────────────────────────────────────────────────
def threshold_at_recall(probs, labels, target_recall=0.85):
    """Find lowest threshold achieving recall >= target_recall."""
    for t in np.linspace(1.0, 0.0, 1001):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if recall >= target_recall:
            return t
    return 0.0


def mcnemar_test(probs_a, probs_b, labels, thresh_a, thresh_b):
    """McNemar's test comparing two models' binary predictions."""
    preds_a = (probs_a >= thresh_a).astype(int)
    preds_b = (probs_b >= thresh_b).astype(int)
    correct_a = (preds_a == labels).astype(int)
    correct_b = (preds_b == labels).astype(int)
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    if b + c == 0:
        return 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return round(p_value, 6), b, c


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('PTSD NLP — BioClinicalBERT + PU Loss Training')
    print('=' * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB'
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # ── Load data ────────────────────────────────────────────────────────
    print('\n[1/5] Loading data ...')
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df   = pd.read_parquet(VAL_PARQUET)
    test_df  = pd.read_parquet(TEST_PARQUET)

    n_pos = (train_df['ptsd_label'] == 1).sum()
    n_unl = (train_df['ptsd_label'] == 0).sum()
    pi_p  = n_pos / (n_pos + n_unl)

    print(f'  Train: {len(train_df):,} rows  (pos={n_pos:,}, unl={n_unl:,})')
    print(f'  Val:   {len(val_df):,} rows')
    print(f'  Test:  {len(test_df):,} rows')
    print(f'  Estimated class prior pi_p = {pi_p:.4f}')

    # ── Tokenizer & model ────────────────────────────────────────────────
    print('\n[2/5] Loading BioClinicalBERT ...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,} total, {n_train_params:,} trainable')
    print(f'  Max sequence length: {MAX_LEN} (vs. Longformer 4096)')
    print(f'  Gradient checkpointing: enabled')

    # ── Datasets & loaders ───────────────────────────────────────────────
    print('\n[3/5] Preparing datasets ...')
    train_ds = NoteDataset(train_df, tokenizer, MAX_LEN)
    val_ds   = NoteDataset(val_df, tokenizer, MAX_LEN)
    test_ds  = NoteDataset(test_df, tokenizer, MAX_LEN)

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2 if use_cuda else 0,
                              pin_memory=use_cuda)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2 if use_cuda else 0,
                              pin_memory=use_cuda)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2 if use_cuda else 0,
                              pin_memory=use_cuda)

    # ── Optimizer & scheduler ────────────────────────────────────────────
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f'  Batch size: {BATCH_SIZE} × {GRAD_ACCUM_STEPS} accum = '
          f'{BATCH_SIZE * GRAD_ACCUM_STEPS} effective')
    print(f'  Total optimizer steps: {total_steps:,}')
    print(f'  Warmup steps: {warmup_steps:,}', flush=True)

    # ── Training ─────────────────────────────────────────────────────────
    print('\n[4/5] Training ...')
    os.makedirs(BEST_DIR, exist_ok=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_auprc = -1.0
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f'\n--- Epoch {epoch}/{EPOCHS} ---', flush=True)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            pi_p, device, GRAD_ACCUM_STEPS
        )
        train_time = time.time() - t0

        t1 = time.time()
        val_auprc, _, _ = run_eval(model, val_loader, device)
        val_time = time.time() - t1

        print(f'  Train loss: {train_loss:.4f}  ({train_time:.0f}s)')
        print(f'  Val AUPRC:  {val_auprc:.4f}  ({val_time:.0f}s)', flush=True)

        log_rows.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_auprc': round(val_auprc, 6),
            'train_time_s': round(train_time, 1),
            'val_time_s': round(val_time, 1),
        })

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            model.save_pretrained(BEST_DIR)
            tokenizer.save_pretrained(BEST_DIR)
            print(f'  >> New best model saved (AUPRC={best_auprc:.4f})',
                  flush=True)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_CSV, index=False)
    print(f'\nTraining log → {LOG_CSV}')

    # ── Test evaluation with best checkpoint ─────────────────────────────
    print('\n[5/5] Evaluating best checkpoint on test set ...')
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_DIR, num_labels=2
    )
    model.to(device)

    test_auprc, test_probs, test_labels = run_eval(model, test_loader, device)
    test_auroc = roc_auc_score(test_labels, test_probs)

    # Use same recall target as Longformer (0.85)
    thresh = threshold_at_recall(test_probs, test_labels, 0.85)
    preds = (test_probs >= thresh).astype(int)
    tp = int(((preds == 1) & (test_labels == 1)).sum())
    fp = int(((preds == 1) & (test_labels == 0)).sum())
    tn = int(((preds == 0) & (test_labels == 0)).sum())
    fn = int(((preds == 0) & (test_labels == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # McNemar's test vs. Longformer
    mcnemar_p, mcnemar_b, mcnemar_c = None, None, None
    lf_thresh = 0.38  # from CLAUDE.md / evaluation_results.json

    if os.path.exists(LF_PRED_CSV):
        lf_pred = pd.read_csv(LF_PRED_CSV)
        # Align by subject_id to ensure same order
        test_sids = test_df['subject_id'].values
        lf_map = dict(zip(lf_pred['subject_id'], lf_pred['predicted_prob']))
        lf_probs_aligned = np.array([lf_map.get(s, np.nan) for s in test_sids])

        if not np.any(np.isnan(lf_probs_aligned)):
            mcnemar_p, mcnemar_b, mcnemar_c = mcnemar_test(
                lf_probs_aligned, test_probs, test_labels, lf_thresh, thresh
            )

    # Build results
    results = {
        'model': 'BioClinicalBERT (PU)',
        'max_length': MAX_LEN,
        'best_val_auprc': round(best_auprc, 4),
        'test_AUPRC': round(test_auprc, 4),
        'test_AUROC': round(test_auroc, 4),
        'threshold_recall_85': round(thresh, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'precision': round(prec, 4),
        'F1': round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_samples': len(test_labels),
        'n_pos': int(test_labels.sum()),
        'mcnemar_p_vs_longformer': mcnemar_p,
        'mcnemar_b': mcnemar_b,
        'mcnemar_c': mcnemar_c,
    }

    with open(EVAL_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  → {EVAL_JSON}')

    # Save test predictions for downstream use
    pred_df = pd.DataFrame({
        'subject_id': test_df['subject_id'].values,
        'hadm_id': test_df['hadm_id'].values,
        'ptsd_label': test_labels,
        'predicted_prob': np.round(test_probs, 6),
    })
    pred_csv = f'{RESULTS_PREDS}/ewang163_bioclinbert_test_predictions.csv'
    pred_df.to_csv(pred_csv, index=False)
    print(f'  → {pred_csv}')

    # ── Side-by-side comparison ──────────────────────────────────────────
    print('\n' + '=' * 65)
    print('SIDE-BY-SIDE COMPARISON: Longformer vs BioClinicalBERT')
    print('=' * 65)

    # Longformer numbers from CLAUDE.md
    lf = {
        'AUPRC': 0.8827, 'AUROC': 0.8913,
        'sensitivity': 0.85, 'specificity': 0.7464,
        'precision': 0.7128, 'F1': 0.7754,
        'threshold': 0.38, 'max_len': 4096,
    }
    bert = {
        'AUPRC': results['test_AUPRC'], 'AUROC': results['test_AUROC'],
        'sensitivity': results['sensitivity'], 'specificity': results['specificity'],
        'precision': results['precision'], 'F1': results['F1'],
        'threshold': results['threshold_recall_85'], 'max_len': MAX_LEN,
    }

    print(f'\n  {"Metric":<15} {"Longformer":>12} {"BioClinBERT":>12} {"Diff":>10}')
    print(f'  {"-"*15} {"-"*12} {"-"*12} {"-"*10}')
    for metric in ['AUPRC', 'AUROC', 'sensitivity', 'specificity', 'precision', 'F1']:
        lf_val = lf[metric]
        bert_val = bert[metric]
        diff = bert_val - lf_val
        sign = '+' if diff >= 0 else ''
        print(f'  {metric:<15} {lf_val:>12.4f} {bert_val:>12.4f} {sign}{diff:>9.4f}')

    print(f'  {"max_length":<15} {lf["max_len"]:>12} {bert["max_len"]:>12}')
    print(f'  {"threshold":<15} {lf["threshold"]:>12.4f} {bert["threshold"]:>12.4f}')

    if mcnemar_p is not None:
        sig = '***' if mcnemar_p < 0.001 else '**' if mcnemar_p < 0.01 else '*' if mcnemar_p < 0.05 else 'ns'
        print(f'\n  McNemar\'s test: p = {mcnemar_p:.6f} ({sig})')
        print(f'    b (LF correct, BERT wrong) = {mcnemar_b}')
        print(f'    c (LF wrong, BERT correct) = {mcnemar_c}')
        auprc_diff = bert['AUPRC'] - lf['AUPRC']
        sign = '+' if auprc_diff >= 0 else ''
        print(f'\n  AUPRC difference: {sign}{auprc_diff:.4f} — '
              f'{"statistically significant" if mcnemar_p < 0.05 else "not significant"}')
    else:
        print('\n  McNemar\'s test: could not run (Longformer predictions not found)')

    print('\nDone.')


if __name__ == '__main__':
    main()
