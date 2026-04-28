"""
ewang163_unified_inference_bench.py
===================================
Re-measures inference runtime for the four PTSD-detection models on a SINGLE
SLURM allocation (one physical GPU) so the head-to-head comparison in
ewang163_model_comparison.md §2 is apples-to-apples.

Sequentially loads and runs inference for:
  1. Clinical Longformer (PULSNAR)              — models/ewang163_longformer_pulsnar/
  2. BioClinicalBERT (PU)  truncated 512        — models/ewang163_bioclinbert_best/
  3. BioClinicalBERT (PU)  chunk-pool 512x256   — same checkpoint
  4. Keyword DSM-5/PCL-5                        — pure CPU regex
  5. Structured + LogReg                        — pure CPU sklearn

Each model is timed on BOTH the validation split (n=1,471) AND the test split
(n=1,551).  Timings are logged with BenchmarkLogger and also written to a
dedicated CSV (ewang163_unified_inference_bench.csv) with hardware identifiers.
"""

import csv
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC              = '/oscar/data/shared/ursa/mimic-iv'
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'

DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
DATA_COHORT        = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_METRICS    = f'{STUDENT_DIR}/results/metrics'

VAL_PARQUET        = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET       = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET        = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'

LONGFORMER_DIR     = f'{MODEL_DIR}/ewang163_longformer_pulsnar'
BIOCLINBERT_DIR    = f'{MODEL_DIR}/ewang163_bioclinbert_best'
KEYWORD_JSON       = f'{MODEL_DIR}/ewang163_keyword_weights.json'
STRUCT_LR_PKL      = f'{MODEL_DIR}/ewang163_structured_logreg.pkl'
STRUCT_FEAT_JSON   = f'{MODEL_DIR}/ewang163_structured_features.json'

DIAGNOSES_F        = f'{MIMIC}/hosp/3.1/diagnoses_icd.csv'
PRESCRIPTIONS_F    = f'{MIMIC}/hosp/3.1/prescriptions.csv'

OUT_CSV            = f'{RESULTS_METRICS}/ewang163_unified_inference_bench.csv'

LF_MAX_LEN     = 4096
LF_BATCH_SIZE  = 4
BERT_MAX_LEN   = 512
BERT_BATCH     = 16

# ── Comorbidity / drug constants (from CLAUDE.md / structured trainer) ───
COMORBIDITY_PFXS = {
    'dx_MDD':      ['F32', 'F33', '296'],
    'dx_anxiety':  ['F41', '300'],
    'dx_SUD':      ['F10','F11','F12','F13','F14','F15','F16','F17','F18','F19',
                    '303','304','305'],
    'dx_TBI':      ['S06','800','801','802','803','804','850','851','852','853','854'],
    'dx_pain':     ['G89', '338'],
    'dx_suicidal': ['R458', 'V6284', 'E95'],
}

DRUG_PATTERNS = {
    'rx_ssri_snri': [
        'sertraline', 'fluoxetine', 'paroxetine', 'escitalopram', 'citalopram',
        'fluvoxamine', 'venlafaxine', 'duloxetine', 'desvenlafaxine',
        'levomilnacipran', 'milnacipran',
    ],
    'rx_prazosin': ['prazosin'],
    'rx_SGA': [
        'quetiapine', 'olanzapine', 'risperidone', 'aripiprazole', 'ziprasidone',
        'clozapine', 'lurasidone', 'asenapine', 'paliperidone', 'iloperidone',
        'brexpiprazole', 'cariprazine',
    ],
}


def map_race(r):
    if pd.isna(r):
        return 'Other/Unknown'
    s = str(r).upper()
    if 'WHITE' in s:    return 'White'
    if 'BLACK' in s or 'AFRICAN' in s: return 'Black'
    if 'HISPANIC' in s or 'LATINO' in s: return 'Hispanic'
    if 'ASIAN' in s:    return 'Asian'
    return 'Other/Unknown'


# ── Transformer dataset / inference ───────────────────────────────────────
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
def run_truncated_inference(model, tokenizer, texts, device, max_len, batch_size):
    ds = NoteDataset(texts, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(out.logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


@torch.no_grad()
def run_chunk_pool_inference(model, tokenizer, texts, device,
                             chunk_len=512, stride=256, pool='max'):
    model.eval()
    out = np.zeros(len(texts))
    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102
    usable = chunk_len - 2

    for i, text in enumerate(texts):
        toks = tokenizer(text, return_tensors='pt', truncation=False,
                         add_special_tokens=False)
        ids_full = toks['input_ids'].squeeze(0)
        n = len(ids_full)

        if n <= usable:
            enc = tokenizer(text, max_length=chunk_len, padding='max_length',
                            truncation=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attn = enc['attention_mask'].to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                o = model(input_ids=input_ids, attention_mask=attn)
            out[i] = F.softmax(o.logits.float(), dim=-1)[0, 1].item()
            continue

        chunk_probs = []
        start = 0
        while start < n:
            end = min(start + usable, n)
            chunk_ids = torch.cat([
                torch.tensor([cls_id]),
                ids_full[start:end],
                torch.tensor([sep_id]),
            ])
            attn_mask = torch.ones(len(chunk_ids), dtype=torch.long)
            pad_len = chunk_len - len(chunk_ids)
            if pad_len > 0:
                chunk_ids = torch.cat([chunk_ids, torch.zeros(pad_len, dtype=torch.long)])
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])

            chunk_ids = chunk_ids.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                o = model(input_ids=chunk_ids, attention_mask=attn_mask)
            chunk_probs.append(F.softmax(o.logits.float(), dim=-1)[0, 1].item())

            if end >= n:
                break
            start += stride

        out[i] = max(chunk_probs) if pool == 'max' else float(np.mean(chunk_probs))
    return out


# ── Keyword scoring (DSM-5/PCL-5) ─────────────────────────────────────────
def load_keyword_lexicon(path):
    with open(path) as f:
        spec = json.load(f)
    compiled = []
    for entry in spec['phrases']:
        compiled.append((re.compile(entry['pattern'], re.IGNORECASE),
                         float(entry['weight'])))
    return compiled


def score_notes_keyword(texts, compiled):
    raw = np.zeros(len(texts))
    for i, t in enumerate(texts):
        for regex, w in compiled:
            raw[i] += w * len(regex.findall(t))
    return raw


# ── Structured features rebuild (for val + test in one pass) ─────────────
def build_structured_features(eval_df, adm, feature_cols):
    eval_sids = set(eval_df['subject_id'].unique())
    eval_hadms = set(eval_df['hadm_id'].unique())

    idx_rows = adm[adm['hadm_id'].isin(eval_hadms)].copy()
    idx_rows = idx_rows.sort_values('admittime').drop_duplicates('subject_id', keep='first')

    idx_rows['sex_female'] = (idx_rows['gender'] == 'F').astype(int)
    idx_rows['race_cat'] = idx_rows['race'].apply(map_race)
    idx_rows['emergency_admission'] = (
        idx_rows['admission_type'].str.upper().str.contains('EMER', na=False).astype(int)
    )
    idx_rows['medicaid_selfpay'] = (
        idx_rows['insurance'].str.upper().isin(['MEDICAID', 'SELF PAY']).astype(int)
    )
    idx_rows['los_days'] = (
        (idx_rows['dischtime'] - idx_rows['admittime']).dt.total_seconds() / 86400.0
    )

    race_dummies = pd.get_dummies(idx_rows['race_cat'], prefix='race')
    idx_rows = pd.concat([idx_rows, race_dummies], axis=1)
    idx_rows = idx_rows.drop(columns=['race_cat'])

    adm_cohort = adm[adm['subject_id'].isin(eval_sids)].copy()
    index_adm = idx_rows[['subject_id', 'admittime']].rename(
        columns={'admittime': 'idx_admittime'})
    prior_adm = adm_cohort.merge(index_adm, on='subject_id', how='inner')
    prior_adm = prior_adm[prior_adm['admittime'] < prior_adm['idx_admittime']].copy()

    prior_counts = prior_adm.groupby('subject_id').size().reset_index(name='n_prior_admissions')
    idx_rows = idx_rows.merge(prior_counts, on='subject_id', how='left')
    idx_rows['n_prior_admissions'] = idx_rows['n_prior_admissions'].fillna(0).astype(int)

    prior_hadm_to_sid = dict(zip(
        prior_adm['hadm_id'].astype(str), prior_adm['subject_id'].astype(str)
    ))

    dx_flags = {feat: set() for feat in COMORBIDITY_PFXS}
    with open(DIAGNOSES_F) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            hadm_str = parts[1]
            if hadm_str not in prior_hadm_to_sid:
                continue
            sid = prior_hadm_to_sid[hadm_str]
            icd = parts[3].strip().replace('.', '').upper()
            for feat, pfxs in COMORBIDITY_PFXS.items():
                if any(icd.startswith(p) for p in pfxs):
                    dx_flags[feat].add(sid)

    for feat in COMORBIDITY_PFXS:
        idx_rows[feat] = idx_rows['subject_id'].apply(
            lambda s, f=feat: 1 if str(s) in dx_flags[f] else 0
        )

    rx_flags = {feat: set() for feat in DRUG_PATTERNS}
    with open(PRESCRIPTIONS_F) as f:
        header_line = next(f)
        cols = [c.strip() for c in header_line.split(',')]
        hadm_idx = cols.index('hadm_id')
        drug_idx = cols.index('drug')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(hadm_idx, drug_idx):
                continue
            hadm_str = parts[hadm_idx].strip()
            if hadm_str not in prior_hadm_to_sid:
                continue
            sid = prior_hadm_to_sid[hadm_str]
            drug_lower = parts[drug_idx].strip().lower()
            for feat, patterns in DRUG_PATTERNS.items():
                if any(p in drug_lower for p in patterns):
                    rx_flags[feat].add(sid)

    for feat in DRUG_PATTERNS:
        idx_rows[feat] = idx_rows['subject_id'].apply(
            lambda s, f=feat: 1 if str(s) in rx_flags[f] else 0
        )

    for col in feature_cols:
        if col not in idx_rows.columns:
            idx_rows[col] = 0

    idx_keyed = idx_rows.set_index('subject_id')
    X = np.zeros((len(eval_df), len(feature_cols)), dtype=np.float32)
    for i, sid in enumerate(eval_df['subject_id'].values):
        if sid in idx_keyed.index:
            row = idx_keyed.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for j, col in enumerate(feature_cols):
                v = row.get(col, 0)
                X[i, j] = float(v) if not pd.isna(v) else 0.0
    return X


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('PTSD NLP — UNIFIED INFERENCE BENCHMARK (single GPU)')
    print('=' * 70, flush=True)

    bench = BenchmarkLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        print(f'  GPU: {gpu_name}  ({gpu_mem_gb} GB)')
    else:
        gpu_name, gpu_mem_gb = 'cpu', 0.0

    # ── Load splits ──────────────────────────────────────────────────────
    val_df  = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    val_texts  = val_df['note_text'].tolist()
    test_texts = test_df['note_text'].tolist()
    print(f'Val:  {len(val_df):,} | Test: {len(test_df):,}')

    rows = []  # output rows for OUT_CSV

    def time_block(label, fn, n_samples, stage_tag, device_tag):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with bench.track('unified_bench', stage=stage_tag, device=device_tag,
                         n_samples=n_samples, notes=label):
            fn()
        dt = time.perf_counter() - t0
        peak_gb = (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                   if device.type == 'cuda' else 0.0)
        ms_per = round(dt * 1000.0 / max(n_samples, 1), 2)
        rows.append({
            'model':            label,
            'stage':            stage_tag,
            'wall_clock_s':     round(dt, 3),
            'n_samples':        n_samples,
            'ms_per_patient':   ms_per,
            'gpu_name':         gpu_name,
            'gpu_total_mem_gb': gpu_mem_gb,
            'peak_mem_gb':      peak_gb,
        })
        print(f'  -> {label}/{stage_tag}: {dt:.2f}s '
              f'({ms_per} ms/patient, peak {peak_gb} GB)')

    # ── 1. Clinical Longformer (PULSNAR) ─────────────────────────────────
    print('\n[1/5] Clinical Longformer (PULSNAR) ...')
    lf_tok = AutoTokenizer.from_pretrained(LONGFORMER_DIR)
    lf_model = AutoModelForSequenceClassification.from_pretrained(
        LONGFORMER_DIR, num_labels=2).to(device)

    time_block(
        'Longformer-PULSNAR', lambda:
        run_truncated_inference(lf_model, lf_tok, val_texts, device,
                                LF_MAX_LEN, LF_BATCH_SIZE),
        len(val_df), 'val_inference', 'gpu' if device.type == 'cuda' else 'cpu')
    time_block(
        'Longformer-PULSNAR', lambda:
        run_truncated_inference(lf_model, lf_tok, test_texts, device,
                                LF_MAX_LEN, LF_BATCH_SIZE),
        len(test_df), 'test_inference', 'gpu' if device.type == 'cuda' else 'cpu')

    del lf_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 2. BioClinicalBERT (truncated 512) ───────────────────────────────
    print('\n[2/5] BioClinicalBERT (truncated 512) ...')
    bert_tok = AutoTokenizer.from_pretrained(BIOCLINBERT_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        BIOCLINBERT_DIR, num_labels=2).to(device)

    time_block(
        'BERT-truncated-512', lambda:
        run_truncated_inference(bert_model, bert_tok, val_texts, device,
                                BERT_MAX_LEN, BERT_BATCH),
        len(val_df), 'val_inference', 'gpu' if device.type == 'cuda' else 'cpu')
    time_block(
        'BERT-truncated-512', lambda:
        run_truncated_inference(bert_model, bert_tok, test_texts, device,
                                BERT_MAX_LEN, BERT_BATCH),
        len(test_df), 'test_inference', 'gpu' if device.type == 'cuda' else 'cpu')

    # ── 3. BioClinicalBERT (chunk-pool 512x256) ──────────────────────────
    print('\n[3/5] BioClinicalBERT (chunk-pool 512x256) ...')
    time_block(
        'BERT-chunk-pool-512x256', lambda:
        run_chunk_pool_inference(bert_model, bert_tok, val_texts, device,
                                 chunk_len=512, stride=256, pool='max'),
        len(val_df), 'val_inference', 'gpu' if device.type == 'cuda' else 'cpu')
    time_block(
        'BERT-chunk-pool-512x256', lambda:
        run_chunk_pool_inference(bert_model, bert_tok, test_texts, device,
                                 chunk_len=512, stride=256, pool='max'),
        len(test_df), 'test_inference', 'gpu' if device.type == 'cuda' else 'cpu')

    del bert_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 4. Keyword DSM-5 / PCL-5 ─────────────────────────────────────────
    print('\n[4/5] Keyword baseline ...')
    kw_lex = load_keyword_lexicon(KEYWORD_JSON)
    print(f'  Loaded {len(kw_lex)} compiled phrases.')

    time_block(
        'Keyword-DSM5/PCL5', lambda:
        score_notes_keyword(val_texts, kw_lex),
        len(val_df), 'val_inference', 'cpu')
    time_block(
        'Keyword-DSM5/PCL5', lambda:
        score_notes_keyword(test_texts, kw_lex),
        len(test_df), 'test_inference', 'cpu')

    # ── 5. Structured + LogReg ───────────────────────────────────────────
    print('\n[5/5] Structured + LogReg ...')
    with open(STRUCT_FEAT_JSON) as f:
        feat_info = json.load(f)
    feature_cols = feat_info['feature_columns']
    with open(STRUCT_LR_PKL, 'rb') as f:
        struct_model = pickle.load(f)

    adm = pd.read_parquet(ADM_PARQUET)

    # Feature build is part of inference time (it's CPU-bound preprocessing).
    def _struct_val():
        X = build_structured_features(val_df, adm, feature_cols)
        struct_model.predict_proba(X)[:, 1]

    def _struct_test():
        X = build_structured_features(test_df, adm, feature_cols)
        struct_model.predict_proba(X)[:, 1]

    time_block('Structured-LogReg', _struct_val,
               len(val_df), 'val_inference', 'cpu')
    time_block('Structured-LogReg', _struct_test,
               len(test_df), 'test_inference', 'cpu')

    # ── Write CSV ────────────────────────────────────────────────────────
    print('\n=== WRITING RESULTS ===')
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ['model', 'stage', 'wall_clock_s', 'n_samples',
                  'ms_per_patient', 'gpu_name', 'gpu_total_mem_gb',
                  'peak_mem_gb']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote: {OUT_CSV}')

    # Pretty-print final table
    print('\n=== FINAL TABLE ===')
    hdr = (f'{"model":<28} {"stage":<16} {"wall(s)":>10} '
           f'{"n":>6} {"ms/pt":>10} {"peak GB":>9}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["model"]:<28} {r["stage"]:<16} '
              f'{r["wall_clock_s"]:>10.2f} {r["n_samples"]:>6} '
              f'{r["ms_per_patient"]:>10.2f} {r["peak_mem_gb"]:>9.2f}')
    print(f'\nGPU: {gpu_name}  ({gpu_mem_gb} GB)')
    print('Done.')


if __name__ == '__main__':
    main()
