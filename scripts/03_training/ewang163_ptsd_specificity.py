"""
ewang163_ptsd_specificity.py
============================
PTSD-specificity check: retrain with MDD/anxiety patients as explicit
negatives (hard psychiatric controls) instead of the unlabeled pool.

If the PU model's AUPRC drops substantially with psychiatric controls,
it suggests the model was partly learning "psychiatric language" rather
than PTSD-specific signal. A small drop is reassuring.

Pipeline:
  1. Identify MDD/anxiety patients from MIMIC-IV (exclude PTSD+, proxy)
  2. 1:1 age_decade × sex match to PTSD+ (up to 5,711)
  3. Extract section-filtered notes for controls
  4. Build corpus: PTSD+ (label=1) vs psychiatric controls (label=0)
  5. 80/10/10 patient-level split, standard BCE (not PU loss)
  6. Fine-tune Clinical Longformer, evaluate

Outputs:
    ewang163_psych_control_subjects.txt
    ewang163_psych_control_notes.parquet
    ewang163_specificity_longformer_best/
    ewang163_specificity_eval_results.json

Submit via SLURM:
    sbatch ewang163_ptsd_specificity.sh
"""

import csv
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

csv.field_size_limit(sys.maxsize)

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC           = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_COHORT     = f'{STUDENT_DIR}/data/cohort'
DATA_NOTES      = f'{STUDENT_DIR}/data/notes'
MODEL_DIR       = f'{STUDENT_DIR}/models'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'

DIAGNOSES_F     = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'
PATIENTS_F      = f'{MIMIC}/hosp/3.1/patients.csv'
ADMISSIONS_F    = f'{MIMIC}/hosp/3.1/admissions.csv'
DISCHARGE_F     = f'{MIMIC}/note/2.2/discharge.csv'
ADM_PARQUET     = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'
CORPUS_PARQUET  = f'{DATA_NOTES}/ewang163_ptsd_corpus.parquet'
PROXY_PARQUET   = f'{DATA_NOTES}/ewang163_proxy_notes.parquet'

CTRL_SUBJECTS_F = f'{DATA_COHORT}/ewang163_psych_control_subjects.txt'
CTRL_NOTES_F    = f'{DATA_NOTES}/ewang163_psych_control_notes.parquet'
BEST_DIR        = f'{MODEL_DIR}/ewang163_specificity_longformer_best'
EVAL_JSON       = f'{RESULTS_METRICS}/ewang163_specificity_eval_results.json'

MODEL_NAME = 'yikuan8/Clinical-Longformer'

# ── Hyperparameters (same as main model) ─────────────────────────────────
MAX_LEN          = 4096
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 16
LR               = 2e-5
EPOCHS           = 5
WARMUP_FRAC      = 0.1
WEIGHT_DECAY     = 0.01

# ── ICD prefixes for MDD and anxiety (from CLAUDE.md) ───────────────────
MDD_PFX     = ['F32', 'F33', '296']
ANXIETY_PFX = ['F41', '300']
PSYCH_PFX   = MDD_PFX + ANXIETY_PFX

# PTSD prefixes (for exclusion)
PTSD_ICD10  = 'F431'
PTSD_ICD9   = '30981'

# ── Section filtering (same as notes_extract.py) ────────────────────────
INCLUDE_SECTIONS = {
    'history of present illness', 'social history',
    'past medical history', 'brief hospital course',
}
SECTION_ORDER = [
    'history of present illness', 'social history',
    'past medical history', 'brief hospital course',
]
SECTION_HEADER_RE = re.compile(
    r'^([A-Z][A-Za-z /&\-]+):[ ]*$', re.MULTILINE,
)


def parse_sections(text):
    if not text:
        return {}
    headers = []
    for m in SECTION_HEADER_RE.finditer(text):
        headers.append((m.start(), m.end(), m.group(1).strip().lower()))
    result = {}
    for i, (start, end, name) in enumerate(headers):
        if name not in INCLUDE_SECTIONS:
            continue
        body_start = end
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()
        if body and name not in result:
            result[name] = body
    return result


def concatenate_sections(sections_dict):
    parts = []
    for name in SECTION_ORDER:
        if name in sections_dict:
            parts.append(sections_dict[name])
    return '\n\n'.join(parts)


def age_decade(age):
    if pd.isna(age):
        return 'Other'
    a = int(age)
    if 20 <= a <= 29: return '20s'
    if 30 <= a <= 39: return '30s'
    if 40 <= a <= 49: return '40s'
    if 50 <= a <= 59: return '50s'
    return 'Other'


# ── Dataset ──────────────────────────────────────────────────────────────
class NoteDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
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


# ── Training with weighted BCE ───────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, class_weights,
                    device, grad_accum_steps):
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
            # Standard BCE with class weights
            logits = outputs.logits  # (B, 2)
            weight = class_weights[labels]  # per-sample weight
            loss_per_sample = F.cross_entropy(
                logits.float(), labels, reduction='none'
            )
            loss = (loss_per_sample * weight).mean() / grad_accum_steps

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
def run_eval(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch['label'].numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auprc = average_precision_score(all_labels, all_probs)
    return auprc, all_probs, all_labels


def threshold_at_recall(probs, labels, target_recall=0.85):
    for t in np.linspace(1.0, 0.0, 1001):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if recall >= target_recall:
            return t
    return 0.0


def mcnemar_test(probs_a, probs_b, labels, thresh_a, thresh_b):
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
    print('PTSD NLP — Specificity Analysis (Psychiatric Controls)')
    print('=' * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Step 1: Identify excluded subjects ───────────────────────────────
    print('\n[1/8] Loading existing cohort to identify exclusions ...')
    adm_ext = pd.read_parquet(ADM_PARQUET)
    ptsd_subjects = set(adm_ext[adm_ext['group'] == 'ptsd_pos']['subject_id'].unique())
    proxy_subjects = set(adm_ext[adm_ext['group'] == 'proxy']['subject_id'].unique())
    excluded = ptsd_subjects | proxy_subjects
    print(f'  PTSD+ subjects: {len(ptsd_subjects):,}')
    print(f'  Proxy subjects: {len(proxy_subjects):,}')
    print(f'  Total excluded: {len(excluded):,}')

    # PTSD+ demographics for matching
    ptsd_adm = adm_ext[adm_ext['group'] == 'ptsd_pos'].copy()
    ptsd_index = (ptsd_adm[ptsd_adm['is_index_admission']]
                  .drop_duplicates('subject_id', keep='first')
                  .copy())
    ptsd_index['decade'] = ptsd_index['age_at_admission'].apply(age_decade)

    # ── Step 2: Stream diagnoses to find MDD/anxiety patients ────────────
    print('\n[2/8] Streaming diagnoses_icd.csv for MDD/anxiety patients ...')
    psych_subjects = set()
    n_lines = 0
    with open(DIAGNOSES_F) as f:
        next(f)  # skip header
        for line in f:
            n_lines += 1
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            sid = int(parts[0])
            if sid in excluded:
                continue
            icd = parts[3].strip().replace('.', '').upper()
            # Also exclude anyone with a PTSD code (belt and suspenders)
            if icd.startswith(PTSD_ICD10) or icd == PTSD_ICD9:
                excluded.add(sid)
                psych_subjects.discard(sid)
                continue
            if any(icd.startswith(p) for p in PSYCH_PFX):
                psych_subjects.add(sid)

    # Remove any newly-found PTSD patients
    psych_subjects -= excluded
    print(f'  Scanned {n_lines:,} diagnosis lines')
    print(f'  MDD/anxiety patients (excluding PTSD+/proxy): {len(psych_subjects):,}')

    # ── Step 3: Load demographics for matching ───────────────────────────
    print('\n[3/8] Loading demographics for age × sex matching ...')
    # Stream patients.csv and admissions.csv for psych control demographics
    patient_info = {}  # sid → (gender, anchor_age, anchor_year)
    with open(PATIENTS_F) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['subject_id'])
            if sid in psych_subjects:
                patient_info[sid] = {
                    'gender': row['gender'],
                    'anchor_age': int(row['anchor_age']),
                    'anchor_year': int(row['anchor_year']),
                }

    # Get first admission year for each psych patient to compute age
    psych_first_admit = {}  # sid → first admittime year
    psych_first_hadm = {}   # sid → first hadm_id
    with open(ADMISSIONS_F) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['subject_id'])
            if sid not in psych_subjects:
                continue
            hadm_id = int(row['hadm_id'])
            admittime = row['admittime']
            if sid not in psych_first_admit or admittime < psych_first_admit[sid]:
                psych_first_admit[sid] = admittime
                psych_first_hadm[sid] = hadm_id

    # Build DataFrame for matching
    psych_rows = []
    for sid in psych_subjects:
        if sid not in patient_info or sid not in psych_first_admit:
            continue
        info = patient_info[sid]
        admit_year = int(psych_first_admit[sid][:4])
        age_at_adm = info['anchor_age'] + (admit_year - info['anchor_year'])
        psych_rows.append({
            'subject_id': sid,
            'gender': info['gender'],
            'age_at_admission': age_at_adm,
            'decade': age_decade(age_at_adm),
            'first_hadm_id': psych_first_hadm[sid],
        })

    psych_df = pd.DataFrame(psych_rows)
    print(f'  Psychiatric controls with demographics: {len(psych_df):,}')

    # ── Step 4: 1:1 age_decade × sex matching to PTSD+ ──────────────────
    print('\n[4/8] Matching 1:1 on age_decade × sex ...')
    np.random.seed(42)
    matched_rows = []
    for (dec, sex), grp in ptsd_index.groupby(['decade', 'gender']):
        needed = len(grp)  # 1:1 ratio
        pool = psych_df[
            (psych_df['decade'] == dec) & (psych_df['gender'] == sex)
        ]
        take = min(needed, len(pool))
        if take < needed:
            print(f'  WARNING: only {len(pool)} psych controls in stratum '
                  f'(decade={dec}, sex={sex}); needed {needed}')
        if take > 0:
            matched_rows.append(pool.sample(n=take, random_state=42))

    ctrl_matched = pd.concat(matched_rows, ignore_index=True)
    ctrl_subjects = set(ctrl_matched['subject_id'].unique())
    print(f'  Matched psychiatric controls: {len(ctrl_subjects):,} '
          f'(target {len(ptsd_subjects):,})')

    # Save subject IDs
    with open(CTRL_SUBJECTS_F, 'w') as f:
        for sid in sorted(ctrl_subjects):
            f.write(f'{sid}\n')
    print(f'  → {CTRL_SUBJECTS_F}')

    # ── Step 5: Extract section-filtered notes for controls ──────────────
    print('\n[5/8] Extracting notes for psychiatric controls ...')
    # Use index admission = first admission with MDD/anxiety code
    # We need to find the first admission where MDD/anxiety was coded
    psych_index_hadms = {}  # sid → hadm_id of first MDD/anxiety admission
    with open(DIAGNOSES_F) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            sid = int(parts[0])
            if sid not in ctrl_subjects:
                continue
            icd = parts[3].strip().replace('.', '').upper()
            if any(icd.startswith(p) for p in PSYCH_PFX):
                hadm_id = int(parts[1])
                if sid not in psych_index_hadms:
                    psych_index_hadms[sid] = set()
                psych_index_hadms[sid].add(hadm_id)

    # For each patient, find the earliest admission among their MDD/anxiety hadms
    # We need admittime to pick the earliest
    psych_hadm_times = {}  # hadm_id → admittime string
    all_psych_hadms = set()
    for hadms in psych_index_hadms.values():
        all_psych_hadms |= hadms

    with open(ADMISSIONS_F) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hadm_id = int(row['hadm_id'])
            if hadm_id in all_psych_hadms:
                psych_hadm_times[hadm_id] = row['admittime']

    # Pick earliest hadm_id per patient
    ctrl_index_hadm = {}  # sid → hadm_id (earliest MDD/anxiety admission)
    for sid, hadms in psych_index_hadms.items():
        earliest_hadm = min(hadms, key=lambda h: psych_hadm_times.get(h, '9999'))
        ctrl_index_hadm[sid] = earliest_hadm

    target_hadms = set(ctrl_index_hadm.values())
    print(f'  Target hadm_ids for notes: {len(target_hadms):,}')

    # Stream discharge.csv
    collected = []
    with open(DISCHARGE_F, newline='') as f:
        reader = csv.DictReader(f)
        n_scanned = 0
        for row in reader:
            n_scanned += 1
            if n_scanned % 50000 == 0:
                print(f'    ... scanned {n_scanned:,} notes, '
                      f'matched {len(collected):,}')
            hadm_id = int(row['hadm_id'])
            if hadm_id not in target_hadms:
                continue
            sections = parse_sections(row['text'])
            note_text = concatenate_sections(sections)
            if not note_text.strip():
                continue
            # Find which subject this belongs to
            sid = None
            for s, h in ctrl_index_hadm.items():
                if h == hadm_id:
                    sid = s
                    break
            if sid is None:
                continue
            collected.append({
                'subject_id': sid,
                'hadm_id': hadm_id,
                'group': 'psych_control',
                'note_text': note_text,
            })
            target_hadms.discard(hadm_id)
            if not target_hadms:
                break

    ctrl_notes = pd.DataFrame(collected)
    ctrl_notes.to_parquet(CTRL_NOTES_F, index=False)
    print(f'  Extracted {len(ctrl_notes):,} notes → {CTRL_NOTES_F}')

    # ── Step 6: Build corpus and splits ──────────────────────────────────
    print('\n[6/8] Building specificity corpus ...')

    # Load PTSD+ notes from existing corpus (label=1)
    corpus = pd.read_parquet(CORPUS_PARQUET)
    ptsd_corpus = corpus[corpus['ptsd_label'] == 1][
        ['subject_id', 'hadm_id', 'note_text']
    ].copy()
    ptsd_corpus['label'] = 1
    print(f'  PTSD+ notes: {len(ptsd_corpus):,} '
          f'({ptsd_corpus["subject_id"].nunique():,} patients)')

    # Control notes (label=0)
    ctrl_corpus = ctrl_notes[['subject_id', 'hadm_id', 'note_text']].copy()
    ctrl_corpus['label'] = 0
    print(f'  Psych control notes: {len(ctrl_corpus):,} '
          f'({ctrl_corpus["subject_id"].nunique():,} patients)')

    # Combine
    full = pd.concat([ptsd_corpus, ctrl_corpus], ignore_index=True)
    print(f'  Combined: {len(full):,} rows')

    # Patient-level 80/10/10 split
    patients = full.groupby('subject_id')['label'].first().reset_index()
    sids = patients['subject_id'].values
    labels = patients['label'].values

    sids_train, sids_temp, _, labels_temp = train_test_split(
        sids, labels, test_size=0.2, stratify=labels, random_state=42
    )
    sids_val, sids_test, _, _ = train_test_split(
        sids_temp, labels_temp, test_size=0.5, stratify=labels_temp, random_state=42
    )

    train_set, val_set, test_set = set(sids_train), set(sids_val), set(sids_test)
    assert len(train_set & val_set) == 0 and len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0

    df_train = full[full['subject_id'].isin(train_set)].copy()
    df_val   = full[full['subject_id'].isin(val_set)].copy()
    df_test  = full[full['subject_id'].isin(test_set)].copy()

    for name, split in [('train', df_train), ('val', df_val), ('test', df_test)]:
        n_pos = (split['label'] == 1).sum()
        n_neg = (split['label'] == 0).sum()
        print(f'  {name}: {len(split):,} rows (pos={n_pos:,}, neg={n_neg:,})')

    # Compute class weights for balanced BCE
    n_pos_train = (df_train['label'] == 1).sum()
    n_neg_train = (df_train['label'] == 0).sum()
    n_total_train = len(df_train)
    w_pos = n_total_train / (2 * n_pos_train)
    w_neg = n_total_train / (2 * n_neg_train)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device)
    print(f'  Class weights: neg={w_neg:.4f}, pos={w_pos:.4f}')

    # ── Step 7: Fine-tune Longformer ─────────────────────────────────────
    print('\n[7/8] Fine-tuning Clinical Longformer ...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    train_ds = NoteDataset(df_train['note_text'].tolist(),
                           df_train['label'].values.astype(np.int64),
                           tokenizer, MAX_LEN)
    val_ds   = NoteDataset(df_val['note_text'].tolist(),
                           df_val['label'].values.astype(np.int64),
                           tokenizer, MAX_LEN)
    test_ds  = NoteDataset(df_test['note_text'].tolist(),
                           df_test['label'].values.astype(np.int64),
                           tokenizer, MAX_LEN)

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

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f'  Total optimizer steps: {total_steps:,}')
    print(f'  Warmup steps: {warmup_steps:,}')

    os.makedirs(BEST_DIR, exist_ok=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_auprc = -1.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f'\n--- Epoch {epoch}/{EPOCHS} ---', flush=True)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            class_weights, device, GRAD_ACCUM_STEPS
        )
        val_auprc, _, _ = run_eval(model, val_loader, device)

        print(f'  Train loss: {train_loss:.4f}  ({time.time()-t0:.0f}s)')
        print(f'  Val AUPRC:  {val_auprc:.4f}', flush=True)

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            model.save_pretrained(BEST_DIR)
            tokenizer.save_pretrained(BEST_DIR)
            print(f'  >> New best model saved (AUPRC={best_auprc:.4f})')

    # ── Step 8: Evaluate ─────────────────────────────────────────────────
    print('\n[8/8] Evaluating best checkpoint ...')
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_DIR, num_labels=2
    )
    model.to(device)

    # A) Evaluate on specificity test set
    test_auprc, test_probs, test_labels = run_eval(model, test_loader, device)
    test_auroc = roc_auc_score(test_labels, test_probs)
    thresh = threshold_at_recall(test_probs, test_labels, 0.85)
    preds = (test_probs >= thresh).astype(int)
    tp = int(((preds == 1) & (test_labels == 1)).sum())
    fp = int(((preds == 1) & (test_labels == 0)).sum())
    tn = int(((preds == 0) & (test_labels == 0)).sum())
    fn = int(((preds == 0) & (test_labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # B) Apply to proxy patients (external check)
    proxy_auprc = None
    proxy_mean_prob = None
    if os.path.exists(PROXY_PARQUET):
        proxy_df = pd.read_parquet(PROXY_PARQUET)
        if len(proxy_df) > 0:
            proxy_ds = NoteDataset(
                proxy_df['note_text'].tolist(),
                np.ones(len(proxy_df), dtype=np.int64),  # assume positive for scoring
                tokenizer, MAX_LEN,
            )
            proxy_loader = DataLoader(proxy_ds, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=0)
            _, proxy_probs, _ = run_eval(model, proxy_loader, device)
            proxy_mean_prob = float(np.mean(proxy_probs))
            print(f'  Proxy patients: mean predicted prob = {proxy_mean_prob:.4f} '
                  f'(n={len(proxy_df)})')

    # Build results
    results = {
        'model': 'Specificity Longformer (BCE, psych controls)',
        'specificity_test': {
            'AUPRC': round(test_auprc, 4),
            'AUROC': round(test_auroc, 4),
            'threshold_recall_85': round(thresh, 4),
            'sensitivity': round(sens, 4),
            'specificity': round(spec, 4),
            'precision': round(prec, 4),
            'F1': round(f1, 4),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'n_samples': len(test_labels),
            'n_pos': int(test_labels.sum()),
        },
        'proxy_validation': {
            'n_patients': len(proxy_df) if os.path.exists(PROXY_PARQUET) else 0,
            'mean_predicted_prob': round(proxy_mean_prob, 4) if proxy_mean_prob is not None else None,
        },
        'psych_controls': {
            'n_matched': len(ctrl_subjects),
            'n_notes_extracted': len(ctrl_notes),
        },
        'comparison_to_pu_model': {
            'pu_model_AUPRC': 0.8827,
            'specificity_model_AUPRC': round(test_auprc, 4),
            'delta_AUPRC': round(test_auprc - 0.8827, 4),
        },
    }

    with open(EVAL_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  → {EVAL_JSON}')

    # ── Results table ────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('SPECIFICITY ANALYSIS RESULTS')
    print('=' * 65)

    print(f'\n  Specificity test set (PTSD+ vs psychiatric controls):')
    print(f'    AUPRC:       {test_auprc:.4f}')
    print(f'    AUROC:       {test_auroc:.4f}')
    print(f'    Sensitivity: {sens:.4f}')
    print(f'    Specificity: {spec:.4f}')
    print(f'    Precision:   {prec:.4f}')
    print(f'    F1:          {f1:.4f}')

    if proxy_mean_prob is not None:
        print(f'\n  Proxy patients (external check):')
        print(f'    Mean predicted PTSD probability: {proxy_mean_prob:.4f}')

    delta = test_auprc - 0.8827
    print(f'\n  COMPARISON TO PU MODEL:')
    print(f'    PU model AUPRC (unlabeled pool):       0.8827')
    print(f'    Specificity model AUPRC (psych ctrl):  {test_auprc:.4f}')
    print(f'    Delta:                                 {delta:+.4f}')

    if delta > -0.05:
        print(f'\n  INTERPRETATION: Small or no drop ({delta:+.4f}).')
        print(f'  The model appears to learn PTSD-specific signal, not just')
        print(f'  general psychiatric language. This is reassuring.')
    elif delta > -0.15:
        print(f'\n  INTERPRETATION: Moderate drop ({delta:+.4f}).')
        print(f'  Some contribution from general psychiatric signal, but')
        print(f'  the model retains substantial PTSD-specific discrimination.')
    else:
        print(f'\n  INTERPRETATION: Large drop ({delta:+.4f}).')
        print(f'  The PU model may have been partly learning psychiatric')
        print(f'  language rather than PTSD-specific signal. Consider this')
        print(f'  limitation when interpreting the main results.')

    print('\nDone.')


if __name__ == '__main__':
    main()
