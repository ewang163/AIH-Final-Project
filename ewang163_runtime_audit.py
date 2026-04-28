"""
ewang163_runtime_audit.py
=========================
Captures missing val_inference + test_inference times for the keyword and
structured baselines so the runtime comparison table has exact numbers.

Logs to results/metrics/ewang163_runtime_benchmarks.csv via BenchmarkLogger.
CPU-only — submitted via SLURM batch partition.
"""
import sys
import os
import pickle
import json
import numpy as np
import pandas as pd

STUDENT_DIR = '/oscar/data/class/biol1595_2595/students/ewang163'
sys.path.insert(0, STUDENT_DIR)

from scripts.common.ewang163_bench_utils import BenchmarkLogger  # noqa: E402

DATA_SPLITS = f'{STUDENT_DIR}/data/splits'
DATA_COHORT = f'{STUDENT_DIR}/data/cohort'
MODEL_DIR = f'{STUDENT_DIR}/models'

VAL_PARQUET = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'

STRUCT_LR_PKL = f'{MODEL_DIR}/ewang163_structured_logreg.pkl'

# Reuse the evaluator's structured-feature builder + keyword scorer
sys.path.insert(0, f'{STUDENT_DIR}/scripts/04_evaluation')
from ewang163_ptsd_evaluate import build_structured_features, score_notes_keyword  # noqa: E402


def main():
    bench = BenchmarkLogger()

    print('Loading splits ...')
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    val_texts = val_df['note_text'].tolist()
    test_texts = test_df['note_text'].tolist()
    print(f'  Val: {len(val_df):,}   Test: {len(test_df):,}')

    adm = pd.read_parquet(ADM_PARQUET)
    with open(STRUCT_LR_PKL, 'rb') as f:
        struct_lr = pickle.load(f)

    # ── Keyword: val ──
    with bench.track('runtime_audit', stage='keyword_val_inference', device='cpu',
                     n_samples=len(val_df)):
        _ = score_notes_keyword(val_texts)

    # ── Keyword: test (re-bench for consistent log) ──
    with bench.track('runtime_audit', stage='keyword_test_inference', device='cpu',
                     n_samples=len(test_df)):
        _ = score_notes_keyword(test_texts)

    # ── Structured: val ──
    with bench.track('runtime_audit', stage='structured_val_inference', device='cpu',
                     n_samples=len(val_df)):
        X_val, _ = build_structured_features(val_df, adm)
        _ = struct_lr.predict_proba(X_val)[:, 1]

    # ── Structured: test ──
    with bench.track('runtime_audit', stage='structured_test_inference', device='cpu',
                     n_samples=len(test_df)):
        X_test, _ = build_structured_features(test_df, adm)
        _ = struct_lr.predict_proba(X_test)[:, 1]

    print('Done.')


if __name__ == '__main__':
    main()
