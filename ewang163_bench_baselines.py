"""
ewang163_bench_baselines.py
===========================
One-shot benchmark wrapper that re-runs the TF-IDF and Structured baseline
training pipelines under BenchmarkLogger so wall-clock training time is
captured in the canonical runtime CSV.

This was created to fill missing entries discovered during the runtime audit:
- TF-IDF training time (was never benchmarked; SLURM logs do not exist)
- Keyword (DSM-5/PCL-5) "training" time (zero-train scoring; only scoring_all
  was logged previously, but we add a fresh entry for completeness)

Both scripts are CPU-only and run quickly; combined wall-clock < 5 min.

RUN:
  sbatch --partition=batch --mem=16G --time=15:00 \
         --output=/oscar/data/class/biol1595_2595/students/ewang163/logs/ewang163_bench_baselines_%j.out \
         --wrap="source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate && \
                python /oscar/data/class/biol1595_2595/students/ewang163/ewang163_bench_baselines.py"
"""

import sys

STUDENT_DIR = '/oscar/data/class/biol1595_2595/students/ewang163'
sys.path.insert(0, STUDENT_DIR)

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.common.ewang163_bench_utils import BenchmarkLogger

DATA_SPLITS = f'{STUDENT_DIR}/data/splits'
TRAIN_PARQUET = f'{DATA_SPLITS}/ewang163_split_train.parquet'
VAL_PARQUET = f'{DATA_SPLITS}/ewang163_split_val.parquet'

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)


def main():
    bench = BenchmarkLogger()

    print('Loading splits ...')
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)
    X_train_text = train_df['note_text'].fillna('').values
    y_train = train_df['ptsd_label'].values
    X_val_text = val_df['note_text'].fillna('').values
    y_val = val_df['ptsd_label'].values
    print(f'  Train: {len(train_df):,}   Val: {len(val_df):,}')

    # === TF-IDF: vectorize-only ===
    with bench.track('train_tfidf', stage='vectorize_fit_transform',
                     device='cpu', n_samples=len(train_df),
                     notes=f'max_features={MAX_FEATURES}, ngram={NGRAM_RANGE}'):
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=True,
            strip_accents='unicode',
            min_df=3,
            dtype=np.float32,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)
        print(f'  Vocab: {len(vectorizer.vocabulary_):,}, '
              f'matrix={X_train.shape}')

    # === TF-IDF: full LR sweep over C ===
    with bench.track('train_tfidf', stage='logreg_C_sweep',
                     device='cpu', n_samples=len(train_df),
                     notes=f'C_grid={C_GRID}, balanced'):
        best_auprc = -1
        for C in C_GRID:
            m = LogisticRegression(
                C=C, class_weight='balanced',
                solver='lbfgs', max_iter=1000, random_state=42,
            )
            m.fit(X_train, y_train)
            p = m.predict_proba(X_val)[:, 1]
            ap = average_precision_score(y_val, p)
            if ap > best_auprc:
                best_auprc = ap
        print(f'  Best AUPRC over C grid: {best_auprc:.4f}')

    # === TF-IDF: full pipeline (vectorize + sweep) end-to-end ===
    with bench.track('train_tfidf', stage='full_training',
                     device='cpu', n_samples=len(train_df),
                     notes=f'fit + C_sweep over {len(C_GRID)} C values'):
        v2 = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=True,
            strip_accents='unicode',
            min_df=3,
            dtype=np.float32,
        )
        Xt = v2.fit_transform(X_train_text)
        Xv = v2.transform(X_val_text)
        for C in C_GRID:
            m = LogisticRegression(
                C=C, class_weight='balanced',
                solver='lbfgs', max_iter=1000, random_state=42,
            )
            m.fit(Xt, y_train)
            _ = m.predict_proba(Xv)[:, 1]
        print('  TF-IDF full_training complete.')

    print('Done.')


if __name__ == '__main__':
    main()
