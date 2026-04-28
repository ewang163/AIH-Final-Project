"""
ewang163_ptsd_temporal_eval.py
==============================
Fix 3: Evaluate temporal generalization.

Two models, two test sets:
  1. Temporal model on temporal test (post-2015):
     measures generalization to later years when trained on earlier years
  2. Random-split model on temporal test (post-2015):
     measures generalization penalty (how much does the random-split model
     degrade when tested only on later years?)

The val-derived threshold (recall >= 0.85) is computed on the temporal val
set (pre-2015) for the temporal model, and on the random val set for the
random model.

Inputs:
  models/ewang163_longformer_best_temporal/  (temporal model)
  models/ewang163_longformer_best/           (random-split model)
  data/splits/ewang163_split_{val,test}_temporal.parquet
  data/splits/ewang163_split_val.parquet     (for random model's threshold)

Outputs:
  results/metrics/ewang163_temporal_eval_results.json

RUN:
    sbatch --partition=gpu --gres=gpu:1 --mem=32G --time=1:00:00 \
           --output=logs/ewang163_temporal_eval_%j.out \
           --wrap="python scripts/04_evaluation/ewang163_ptsd_temporal_eval.py"
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
MODEL_DIR       = f'{STUDENT_DIR}/models'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'

TEMPORAL_VAL    = f'{DATA_SPLITS}/ewang163_split_val_temporal.parquet'
TEMPORAL_TEST   = f'{DATA_SPLITS}/ewang163_split_test_temporal.parquet'
RANDOM_VAL      = f'{DATA_SPLITS}/ewang163_split_val.parquet'
RANDOM_TEST     = f'{DATA_SPLITS}/ewang163_split_test.parquet'

TEMPORAL_MODEL  = f'{MODEL_DIR}/ewang163_longformer_best_temporal'
RANDOM_MODEL    = f'{MODEL_DIR}/ewang163_longformer_best'

OUT_JSON = f'{RESULTS_METRICS}/ewang163_temporal_eval_results.json'

MAX_LEN    = 4096
BATCH_SIZE = 4


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


def evaluate_model_on_splits(model_dir, val_parquet, test_parquet, device, bench, label):
    """Load model, compute val threshold, evaluate on test."""
    print(f'\n--- {label} ---')
    print(f'  Model:     {model_dir}')
    print(f'  Val split: {val_parquet}')
    print(f'  Test split:{test_parquet}')

    val_df = pd.read_parquet(val_parquet)
    test_df = pd.read_parquet(test_parquet)
    val_texts = val_df['note_text'].tolist()
    test_texts = test_df['note_text'].tolist()
    val_labels = val_df['ptsd_label'].values.astype(np.int64)
    test_labels = test_df['ptsd_label'].values.astype(np.int64)

    print(f'  Val: n={len(val_df)} pos_rate={val_labels.mean():.3f}')
    print(f'  Test: n={len(test_df)} pos_rate={test_labels.mean():.3f}')

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    model.to(device)

    with bench.track('temporal_eval', stage=f'{label}_val',
                     device='gpu' if device.type == 'cuda' else 'cpu',
                     n_samples=len(val_df)):
        val_probs = run_inference(model, tokenizer, val_texts, device)

    with bench.track('temporal_eval', stage=f'{label}_test',
                     device='gpu' if device.type == 'cuda' else 'cpu',
                     n_samples=len(test_df)):
        test_probs = run_inference(model, tokenizer, test_texts, device)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    val_thresh = threshold_at_recall(val_probs, val_labels, 0.85)
    print(f'  Val-derived threshold (recall>=0.85): {val_thresh:.4f}')

    val_metrics = compute_metrics(val_probs, val_labels, val_thresh)
    test_metrics = compute_metrics(test_probs, test_labels, val_thresh)

    print(f'  Val  AUPRC={val_metrics["AUPRC"]:.4f} AUROC={val_metrics["AUROC"]:.4f}')
    print(f'  Test AUPRC={test_metrics["AUPRC"]:.4f} AUROC={test_metrics["AUROC"]:.4f} '
          f'(sens={test_metrics["sensitivity"]:.3f}, spec={test_metrics["specificity"]:.3f})')

    return {
        'model_dir': model_dir,
        'val_parquet': val_parquet,
        'test_parquet': test_parquet,
        'val_threshold': val_thresh,
        'val': val_metrics,
        'test': test_metrics,
    }


def main():
    print('=' * 65)
    print('PTSD NLP — Temporal Generalization Evaluation (Fix 3)')
    print('=' * 65, flush=True)

    bench = BenchmarkLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    results = {}

    # 1. Temporal model on temporal test set (the intended comparison)
    if os.path.isdir(TEMPORAL_MODEL):
        results['temporal_model_temporal_test'] = evaluate_model_on_splits(
            TEMPORAL_MODEL, TEMPORAL_VAL, TEMPORAL_TEST, device, bench,
            'Temporal model, temporal test')
    else:
        print(f'\nWARNING: {TEMPORAL_MODEL} not found — skipping')
        results['temporal_model_temporal_test'] = None

    # 2. Random-split model on temporal test set (control — generalization penalty)
    if os.path.isdir(RANDOM_MODEL):
        results['random_model_temporal_test'] = evaluate_model_on_splits(
            RANDOM_MODEL, RANDOM_VAL, TEMPORAL_TEST, device, bench,
            'Random model, temporal test')
    else:
        print(f'\nWARNING: {RANDOM_MODEL} not found — skipping')
        results['random_model_temporal_test'] = None

    # 3. Random-split model on random test set (baseline for comparison)
    if os.path.isdir(RANDOM_MODEL):
        results['random_model_random_test'] = evaluate_model_on_splits(
            RANDOM_MODEL, RANDOM_VAL, RANDOM_TEST, device, bench,
            'Random model, random test')

    # ── Comparison ───────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('TEMPORAL GENERALIZATION SUMMARY')
    print('=' * 65)

    if (results.get('temporal_model_temporal_test') and
            results.get('random_model_temporal_test')):
        tm = results['temporal_model_temporal_test']['test']
        rm = results['random_model_temporal_test']['test']
        delta_auprc = tm['AUPRC'] - rm['AUPRC']
        delta_auroc = tm['AUROC'] - rm['AUROC']

        print(f'\n  On temporal test (2017-2019):')
        print(f'    Temporal model: AUPRC={tm["AUPRC"]:.4f}  AUROC={tm["AUROC"]:.4f}')
        print(f'    Random model:   AUPRC={rm["AUPRC"]:.4f}  AUROC={rm["AUROC"]:.4f}')
        print(f'    Delta (temp-random): AUPRC={delta_auprc:+.4f}  AUROC={delta_auroc:+.4f}')

        results['delta'] = {
            'temporal_minus_random_AUPRC': round(delta_auprc, 4),
            'temporal_minus_random_AUROC': round(delta_auroc, 4),
            'interpretation': (
                'Positive delta: temporal training helps generalization. '
                'Negative delta: random-split training is sufficient.'
            ),
        }

    if results.get('random_model_random_test'):
        rr = results['random_model_random_test']['test']
        rm = results.get('random_model_temporal_test', {}).get('test', {})
        print(f'\n  Random model — random vs. temporal test:')
        print(f'    Random test:   AUPRC={rr.get("AUPRC", "N/A")}')
        print(f'    Temporal test: AUPRC={rm.get("AUPRC", "N/A")}')
        if 'AUPRC' in rm:
            print(f'    Generalization penalty: {rr["AUPRC"] - rm["AUPRC"]:+.4f}')
            results['generalization_penalty'] = {
                'random_random_AUPRC': rr['AUPRC'],
                'random_temporal_AUPRC': rm['AUPRC'],
                'penalty': round(rr['AUPRC'] - rm['AUPRC'], 4),
            }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults → {OUT_JSON}')
    print('\nDone.')


if __name__ == '__main__':
    main()
