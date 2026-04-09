"""
ewang163_ptsd_train_longformer.py
=================================
Fine-tunes Clinical Longformer with Kiryo et al. (2017) non-negative
PU loss for PTSD detection in discharge notes.

Submit via SLURM:
    sbatch ewang163_ptsd_train_longformer.sbatch

Or run interactively on a GPU node:
    source ptsd_env/bin/activate
    python ewang163_ptsd_train_longformer.py

Outputs:
    ewang163_longformer_best/            — best checkpoint by val AUPRC
    ewang163_longformer_training_log.csv — epoch-level metrics
"""

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
from sklearn.metrics import average_precision_score

# ── Paths ─────────────────────────────────────────────────────────────────
OUT = '/oscar/data/class/biol1595_2595/students/ewang163'

TRAIN_PARQUET = f'{OUT}/ewang163_split_train.parquet'
VAL_PARQUET   = f'{OUT}/ewang163_split_val.parquet'
BEST_DIR      = f'{OUT}/ewang163_longformer_best'
LOG_CSV       = f'{OUT}/ewang163_longformer_training_log.csv'

MODEL_NAME = 'yikuan8/Clinical-Longformer'

# ── Hyperparameters (from CLAUDE.md) ──────────────────────────────────────
MAX_LEN          = 4096
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 16     # effective batch size = 2 * 16 = 32
LR               = 2e-5
EPOCHS           = 5
WARMUP_FRAC      = 0.1
WEIGHT_DECAY     = 0.01


# ── PU Loss: Kiryo et al. (2017) non-negative risk estimator ─────────────
def pu_loss(logits, labels, pi_p):
    """
    R_pu(f) = pi_p * E_P[l(f(x), +1)]
            + max(0, E_U[l(f(x), -1)] - pi_p * E_P[l(f(x), -1)])

    Args:
        logits: raw model outputs (before sigmoid), shape (B,)
        labels: 1 for positive, 0 for unlabeled, shape (B,)
        pi_p:   class prior P(Y=1)
    """
    pos_mask = (labels == 1)
    unl_mask = (labels == 0)

    n_pos = pos_mask.sum()
    n_unl = unl_mask.sum()

    # If a batch has no positives or no unlabeled, fall back to standard BCE
    if n_pos == 0 or n_unl == 0:
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    # E_P[l(f(x), +1)]
    loss_pos = F.binary_cross_entropy_with_logits(
        logits[pos_mask], torch.ones_like(logits[pos_mask]), reduction='mean'
    )

    # E_U[l(f(x), -1)]
    loss_unl = F.binary_cross_entropy_with_logits(
        logits[unl_mask], torch.zeros_like(logits[unl_mask]), reduction='mean'
    )

    # E_P[l(f(x), -1)]
    loss_pos_as_neg = F.binary_cross_entropy_with_logits(
        logits[pos_mask], torch.zeros_like(logits[pos_mask]), reduction='mean'
    )

    # PU risk with non-negative correction
    pu = pi_p * loss_pos + torch.clamp(
        loss_unl - pi_p * loss_pos_as_neg, min=0.0
    )

    return pu


# ── Dataset ───────────────────────────────────────────────────────────────
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


# ── Training loop ─────────────────────────────────────────────────────────
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
            # Take difference of class logits as a single scalar logit
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

        if (step + 1) % 50 == 0:
            avg = total_loss / n_batches
            print(f'    step {step+1}/{len(loader)}  avg_loss={avg:.4f}',
                  flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
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


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('PTSD NLP — Clinical Longformer + PU Loss Training')
    print('=' * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # ── Load data ─────────────────────────────────────────────────────────
    print('\n[1/4] Loading data ...')
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df   = pd.read_parquet(VAL_PARQUET)

    n_pos = (train_df['ptsd_label'] == 1).sum()
    n_unl = (train_df['ptsd_label'] == 0).sum()
    pi_p  = n_pos / (n_pos + n_unl)

    print(f'  Train: {len(train_df):,} rows  (pos={n_pos:,}, unl={n_unl:,})')
    print(f'  Val:   {len(val_df):,} rows')
    print(f'  Estimated class prior pi_p = {pi_p:.4f}')

    # ── Tokenizer & model ────────────────────────────────────────────────
    print('\n[2/4] Loading Clinical Longformer ...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,} total, {n_train_params:,} trainable')
    print(f'  Gradient checkpointing: enabled')
    print(f'  Mixed precision (fp16): enabled')

    # ── Datasets & loaders ────────────────────────────────────────────────
    print('\n[3/4] Preparing datasets ...')
    train_ds = NoteDataset(train_df, tokenizer, MAX_LEN)
    val_ds   = NoteDataset(val_df, tokenizer, MAX_LEN)

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2 if use_cuda else 0,
                              pin_memory=use_cuda)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2 if use_cuda else 0,
                              pin_memory=use_cuda)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f'  Total optimizer steps: {total_steps:,}')
    print(f'  Warmup steps: {warmup_steps:,}', flush=True)

    # ── Training loop ────────────────────────────────────────────────────
    print('\n[4/4] Training ...')
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
        val_auprc, _, _ = evaluate(model, val_loader, device)
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

    # ── Save training log ─────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_CSV, index=False)
    print(f'\nTraining log saved → {LOG_CSV}')

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('TRAINING COMPLETE')
    print('=' * 65)
    print(f'  Best validation AUPRC: {best_auprc:.4f}')
    print(f'  Best checkpoint: {BEST_DIR}')
    print(log_df.to_string(index=False))
    print('\nDone.')


if __name__ == '__main__':
    main()
