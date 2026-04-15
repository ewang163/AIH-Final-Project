"""
ewang163_ptsd_train_tfidf.py
============================
TF-IDF + logistic regression baseline for PTSD detection (PU setting).

Uses class_weight='balanced' as an acceptable approximation for a simple baseline.
Tunes regularization parameter C on validation AUPRC.

Inputs:
    ewang163_split_train.parquet
    ewang163_split_val.parquet

Outputs:
    ewang163_tfidf_vectorizer.pkl
    ewang163_tfidf_logreg.pkl
    ewang163_tfidf_val_results.txt

RUN:
    source ptsd_env/bin/activate
    python ewang163_ptsd_train_tfidf.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
MODEL_DIR       = f'{STUDENT_DIR}/models'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'

TRAIN_PARQUET = f'{DATA_SPLITS}/ewang163_split_train.parquet'
VAL_PARQUET   = f'{DATA_SPLITS}/ewang163_split_val.parquet'

VECTORIZER_PKL = f'{MODEL_DIR}/ewang163_tfidf_vectorizer.pkl'
LOGREG_PKL     = f'{MODEL_DIR}/ewang163_tfidf_logreg.pkl'
RESULTS_TXT    = f'{RESULTS_METRICS}/ewang163_tfidf_val_results.txt'

# ── Hyperparameters ───────────────────────────────────────────────────────
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)


def precision_at_recall(y_true, y_score, target_recall=0.85):
    """Compute precision at the threshold where recall >= target_recall."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns recall in decreasing order
    # Find the first index where recall drops below target
    valid = recall >= target_recall
    if valid.any():
        # Among points with recall >= target, pick the one with highest precision
        return precision[valid].max(), thresholds[valid[:-1]].min() if valid[:-1].any() else 0.5
    return 0.0, 0.5


def main():
    print('=' * 65)
    print('PTSD NLP Project — TF-IDF + Logistic Regression Baseline')
    print('=' * 65)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print('\n[1/4] Loading train/val splits ...')
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df   = pd.read_parquet(VAL_PARQUET)

    X_train_text = train_df['note_text'].fillna('').values
    y_train      = train_df['ptsd_label'].values
    X_val_text   = val_df['note_text'].fillna('').values
    y_val        = val_df['ptsd_label'].values

    print(f'  Train: {len(train_df):,} rows ({y_train.sum():,} pos, '
          f'{(y_train == 0).sum():,} unlabeled)')
    print(f'  Val:   {len(val_df):,} rows ({y_val.sum():,} pos, '
          f'{(y_val == 0).sum():,} unlabeled)')

    # ── Step 2: Fit TF-IDF ────────────────────────────────────────────────
    print('\n[2/4] Fitting TF-IDF vectorizer ...')
    print(f'  max_features={MAX_FEATURES:,}, ngram_range={NGRAM_RANGE}, '
          f'sublinear_tf=True')

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents='unicode',
        min_df=3,
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val   = vectorizer.transform(X_val_text)
    print(f'  Vocabulary size: {len(vectorizer.vocabulary_):,}')
    print(f'  Train matrix: {X_train.shape}')

    # ── Step 3: Tune C on validation AUPRC ────────────────────────────────
    print('\n[3/4] Tuning C on validation AUPRC ...')
    print(f'  C grid: {C_GRID}')

    best_auprc = -1
    best_C = None
    best_model = None
    results = []

    for C in C_GRID:
        model = LogisticRegression(
            C=C,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, y_prob)
        auroc = roc_auc_score(y_val, y_prob)

        results.append({'C': C, 'AUPRC': auprc, 'AUROC': auroc})
        marker = ' ← best' if auprc > best_auprc else ''
        print(f'  C={C:<8} AUPRC={auprc:.4f}  AUROC={auroc:.4f}{marker}')

        if auprc > best_auprc:
            best_auprc = auprc
            best_C = C
            best_model = model

    # ── Step 4: Final evaluation + save ───────────────────────────────────
    print(f'\n[4/4] Best C={best_C} — final validation metrics ...')

    y_prob_best = best_model.predict_proba(X_val)[:, 1]
    auprc_final = average_precision_score(y_val, y_prob_best)
    auroc_final = roc_auc_score(y_val, y_prob_best)
    prec_at_r85, threshold_r85 = precision_at_recall(y_val, y_prob_best, 0.85)

    print(f'  Validation AUPRC:            {auprc_final:.4f}')
    print(f'  Validation AUROC:            {auroc_final:.4f}')
    print(f'  Precision @ recall>=0.85:    {prec_at_r85:.4f}  (threshold={threshold_r85:.4f})')

    # Save vectorizer and model
    with open(VECTORIZER_PKL, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f'\n  Saved: {VECTORIZER_PKL}')

    with open(LOGREG_PKL, 'wb') as f:
        pickle.dump(best_model, f)
    print(f'  Saved: {LOGREG_PKL}')

    # Save results text
    lines = [
        '=' * 65,
        'TF-IDF + Logistic Regression — Validation Results',
        '=' * 65,
        '',
        f'TF-IDF: max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE}, sublinear_tf=True',
        f'LogReg: class_weight=balanced, solver=lbfgs, max_iter=1000',
        f'Best C: {best_C}',
        '',
        'C tuning results:',
    ]
    for r in results:
        lines.append(f'  C={r["C"]:<8}  AUPRC={r["AUPRC"]:.4f}  AUROC={r["AUROC"]:.4f}')
    lines += [
        '',
        f'Final validation metrics (C={best_C}):',
        f'  AUPRC:                    {auprc_final:.4f}',
        f'  AUROC:                    {auroc_final:.4f}',
        f'  Precision @ recall>=0.85: {prec_at_r85:.4f}  (threshold={threshold_r85:.4f})',
        '',
        f'Train: {len(train_df):,} rows ({y_train.sum():,} pos, {(y_train==0).sum():,} unl)',
        f'Val:   {len(val_df):,} rows ({y_val.sum():,} pos, {(y_val==0).sum():,} unl)',
    ]

    with open(RESULTS_TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved: {RESULTS_TXT}')

    print('\nDone.')


if __name__ == '__main__':
    main()
