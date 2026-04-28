"""
ewang163_ptsd_train_keyword.py
==============================
Naive keyword/phrase-lookup baseline for PTSD detection.

Zero-training model: scans section-filtered discharge notes for a curated
list of PTSD-related phrases derived from DSM-5 diagnostic criteria and the
PCL-5 screening instrument.  Produces a per-note score (weighted phrase count)
that can be evaluated on the same splits as all other models.

Two scoring variants:
  1. Raw weighted count — sum of weights for each matched phrase
  2. TF-normalized — raw count divided by note length in words

The phrase list is intentionally hand-curated from published clinical
instruments, NOT derived from model attribution (which would be circular).

Phrase categories (DSM-5 Criterion mapping):
  A — Traumatic exposure: combat, assault, abuse, MVA, rape, etc.
  B — Intrusion: flashback, nightmare, re-experiencing, intrusive
  C — Avoidance: avoidance, numbing, detachment
  D — Negative cognition/mood: guilt, blame, diminished interest
  E — Arousal/reactivity: hypervigilance, startle, insomnia, irritability

Inputs:
    data/splits/ewang163_split_{train,val,test}.parquet

Outputs:
    models/ewang163_keyword_weights.json         — phrase list + weights
    results/metrics/ewang163_keyword_val_results.txt
    results/predictions/ewang163_keyword_test_predictions.csv

RUN (CPU only — no training required):
    sbatch --partition=batch --mem=8G --time=1:00:00 \
           --output=logs/ewang163_keyword_%j.out \
           --wrap="python scripts/03_training/ewang163_ptsd_train_keyword.py"
"""

import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/oscar/data/class/biol1595_2595/students/ewang163')
from scripts.common.ewang163_bench_utils import BenchmarkLogger

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR        = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS        = f'{STUDENT_DIR}/data/splits'
MODEL_DIR          = f'{STUDENT_DIR}/models'
RESULTS_METRICS    = f'{STUDENT_DIR}/results/metrics'
RESULTS_PREDICTIONS = f'{STUDENT_DIR}/results/predictions'

TRAIN_PARQUET = f'{DATA_SPLITS}/ewang163_split_train.parquet'
VAL_PARQUET   = f'{DATA_SPLITS}/ewang163_split_val.parquet'
TEST_PARQUET  = f'{DATA_SPLITS}/ewang163_split_test.parquet'

WEIGHTS_JSON  = f'{MODEL_DIR}/ewang163_keyword_weights.json'
VAL_RESULTS   = f'{RESULTS_METRICS}/ewang163_keyword_val_results.txt'
TEST_PREDS    = f'{RESULTS_PREDICTIONS}/ewang163_keyword_test_predictions.csv'

# ── DSM-5 / PCL-5 Phrase Lexicon ─────────────────────────────────────────
# Each entry: (regex_pattern, weight, dsm5_criterion)
# Weights reflect clinical specificity for PTSD vs. general psych:
#   3.0 = highly PTSD-specific (e.g., flashback, hypervigilance)
#   2.0 = moderately specific (e.g., trauma, nightmare)
#   1.0 = shared with other conditions (e.g., anxiety, insomnia)
#   0.5 = weak/contextual signal (e.g., guilt, irritable)

PHRASE_LEXICON = [
    # Criterion A — Traumatic exposure
    (r'\bptsd\b',                            3.0, 'A'),
    (r'\bpost[- ]?traumatic\s+stress\b',     3.0, 'A'),
    (r'\bposttraumatic\s+stress\b',          3.0, 'A'),
    (r'\btrauma\b',                          2.0, 'A'),
    (r'\btraumatic\b',                       2.0, 'A'),
    (r'\bcombat\s+(veteran|exposure|related)', 3.0, 'A'),
    (r'\bsexual\s+assault\b',               3.0, 'A'),
    (r'\bphysical\s+assault\b',             2.5, 'A'),
    (r'\bassault(?:ed)?\b',                  2.0, 'A'),
    (r'\brape[d]?\b',                        3.0, 'A'),
    (r'\bdomestic\s+violence\b',            2.5, 'A'),
    (r'\bintimate\s+partner\s+violence\b',  2.5, 'A'),
    (r'\bipv\b',                             2.5, 'A'),
    (r'\babuse[d]?\b',                       1.5, 'A'),
    (r'\babusive\b',                         1.5, 'A'),
    (r'\bchild(?:hood)?\s+abuse\b',         2.5, 'A'),
    (r'\bmva\b',                             2.0, 'A'),
    (r'\bmotor\s+vehicle\s+accident\b',     2.0, 'A'),
    (r'\bgunshot\b',                         1.5, 'A'),
    (r'\bstab(?:bing|bed)\b',               1.5, 'A'),
    (r'\bviolence\b',                        1.5, 'A'),
    (r'\bwitness(?:ed|ing)?\s+(?:death|violence|shooting|accident)', 2.5, 'A'),
    (r'\bwar\s+zone\b',                     2.5, 'A'),
    (r'\bmilitary\s+sexual\s+trauma\b',     3.0, 'A'),
    (r'\bmst\b',                             2.5, 'A'),

    # Criterion B — Intrusion symptoms
    (r'\bflashback[s]?\b',                   3.0, 'B'),
    (r'\bnightmare[s]?\b',                   2.0, 'B'),
    (r'\bre-?experienc(?:e|ing)\b',          3.0, 'B'),
    (r'\bintrusive\s+(?:thought|memor|image|recollection)', 3.0, 'B'),
    (r'\bdistressing\s+(?:dream|memor|recollection)', 2.0, 'B'),
    (r'\brecurrent\s+(?:dream|nightmare|thought|image)', 2.0, 'B'),
    (r'\bdissociative\s+(?:reaction|episode|flashback)', 3.0, 'B'),

    # Criterion C — Avoidance
    (r'\bavoidance\b',                       1.5, 'C'),
    (r'\bavoid(?:s|ing|ed)?\s+(?:trigger|reminder|thought|feeling|place)', 2.5, 'C'),
    (r'\bemotional(?:ly)?\s+numb(?:ing|ness)?\b', 2.0, 'C'),
    (r'\bdetach(?:ed|ment)\b',              1.5, 'C'),

    # Criterion D — Negative cognition and mood
    (r'\bguilt\b',                           0.5, 'D'),
    (r'\bself[- ]?blame\b',                 1.0, 'D'),
    (r'\bdiminished\s+interest\b',          1.0, 'D'),
    (r'\bestrange(?:d|ment)\b',             1.5, 'D'),
    (r'\bpersistent\s+negative\b',          1.0, 'D'),
    (r'\bunable\s+to\s+(?:feel|experience)\s+positive', 1.5, 'D'),

    # Criterion E — Arousal and reactivity
    (r'\bhypervigilance\b',                  3.0, 'E'),
    (r'\bhypervigilant\b',                   3.0, 'E'),
    (r'\bexaggerated\s+startle\b',          3.0, 'E'),
    (r'\bstartle\s+(?:response|reflex|reaction)', 3.0, 'E'),
    (r'\bhyperarous(?:al|ed)\b',            2.5, 'E'),
    (r'\binsomnia\b',                        1.0, 'E'),
    (r'\bsleep\s+disturbance\b',            1.0, 'E'),
    (r'\birritab(?:le|ility)\b',            0.5, 'E'),
    (r'\banger\s+outburst\b',              1.5, 'E'),
    (r'\breckless\s+behavior\b',            1.0, 'E'),
    (r'\bconcentration\s+difficult\b',      0.5, 'E'),

    # Treatment-related (indirect signal)
    (r'\bprazosin\b',                        2.0, 'tx'),
    (r'\bprolonged\s+exposure\s+therapy\b', 3.0, 'tx'),
    (r'\bcpe\b',                             1.5, 'tx'),
    (r'\bemdr\b',                            3.0, 'tx'),
    (r'\btrauma[- ]?focused\b',             2.5, 'tx'),
    (r'\bpcl[- ]?5\b',                      3.0, 'tx'),
    (r'\bcaps[- ]?5\b',                     3.0, 'tx'),
]

# Pre-compile all regexes
COMPILED_LEXICON = [
    (re.compile(pat, re.IGNORECASE), weight, criterion)
    for pat, weight, criterion in PHRASE_LEXICON
]


def score_note_raw(text):
    """Weighted phrase count for a single note."""
    total = 0.0
    matches = {}
    for regex, weight, criterion in COMPILED_LEXICON:
        hits = len(regex.findall(text))
        if hits > 0:
            total += weight * hits
            matches[regex.pattern] = hits
    return total, matches


def score_note_normalized(text, raw_score):
    """TF-normalized: raw score / word count."""
    n_words = len(text.split())
    if n_words == 0:
        return 0.0
    return raw_score / n_words


def score_notes(texts):
    """Score all notes. Returns (raw_scores, norm_scores, per_criterion_scores)."""
    raw_scores = np.zeros(len(texts))
    norm_scores = np.zeros(len(texts))
    criterion_scores = {c: np.zeros(len(texts)) for c in ['A', 'B', 'C', 'D', 'E', 'tx']}

    for i, text in enumerate(texts):
        raw, _ = score_note_raw(text)
        raw_scores[i] = raw
        norm_scores[i] = score_note_normalized(text, raw)

        for regex, weight, criterion in COMPILED_LEXICON:
            hits = len(regex.findall(text))
            if hits > 0:
                criterion_scores[criterion][i] += weight * hits

    return raw_scores, norm_scores, criterion_scores


def threshold_at_recall(scores, labels, target_recall=0.85):
    """Find lowest threshold achieving recall >= target_recall."""
    sorted_scores = np.sort(np.unique(scores))[::-1]
    for t in sorted_scores:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if recall >= target_recall:
            return t
    return sorted_scores[-1] if len(sorted_scores) > 0 else 0.0


def compute_metrics(scores, labels, name):
    """Compute AUPRC, AUROC, and threshold-based metrics."""
    auprc = average_precision_score(labels, scores)
    auroc = roc_auc_score(labels, scores)

    thresh = threshold_at_recall(scores, labels, 0.85)
    preds = (scores >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return {
        'model': name,
        'AUPRC': round(auprc, 4),
        'AUROC': round(auroc, 4),
        'threshold_recall_85': round(float(thresh), 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'precision': round(prec, 4),
        'F1': round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def main():
    print('=' * 65)
    print('PTSD NLP — Keyword/Phrase-Lookup Baseline')
    print('=' * 65, flush=True)

    bench = BenchmarkLogger()

    # ── Load data ────────────────────────────────────────────────────────
    print('\n[1/5] Loading data ...')
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    print(f'  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')

    # ── Score all splits ─────────────────────────────────────────────────
    print('\n[2/5] Scoring notes with DSM-5/PCL-5 phrase lexicon ...')
    with bench.track('keyword_baseline', stage='scoring_all', device='cpu',
                     n_samples=len(train_df) + len(val_df) + len(test_df)):
        train_raw, train_norm, train_crit = score_notes(train_df['note_text'].tolist())
        val_raw, val_norm, val_crit = score_notes(val_df['note_text'].tolist())
        test_raw, test_norm, test_crit = score_notes(test_df['note_text'].tolist())

    val_labels = val_df['ptsd_label'].values.astype(np.int64)
    test_labels = test_df['ptsd_label'].values.astype(np.int64)

    # ── Evaluate on validation — pick best variant ───────────────────────
    print('\n[3/5] Evaluating on validation set ...')
    val_metrics_raw = compute_metrics(val_raw, val_labels, 'Keyword (raw weighted)')
    val_metrics_norm = compute_metrics(val_norm, val_labels, 'Keyword (TF-normalized)')

    print(f'  Raw weighted:  AUPRC={val_metrics_raw["AUPRC"]:.4f}  '
          f'AUROC={val_metrics_raw["AUROC"]:.4f}')
    print(f'  TF-normalized: AUPRC={val_metrics_norm["AUPRC"]:.4f}  '
          f'AUROC={val_metrics_norm["AUROC"]:.4f}')

    # Pick the variant with higher val AUPRC
    if val_metrics_norm['AUPRC'] > val_metrics_raw['AUPRC']:
        best_variant = 'normalized'
        best_val_scores = val_norm
        best_test_scores = test_norm
        best_val_metrics = val_metrics_norm
    else:
        best_variant = 'raw'
        best_val_scores = val_raw
        best_test_scores = test_raw
        best_val_metrics = val_metrics_raw

    print(f'\n  Best variant: {best_variant}')

    # Compute val-derived threshold for test evaluation (Fix 4 compliance)
    val_thresh = threshold_at_recall(best_val_scores, val_labels, 0.85)
    print(f'  Val-derived threshold (recall>=0.85): {val_thresh:.4f}')

    # ── Evaluate on test set using val-derived threshold ─────────────────
    print('\n[4/5] Evaluating on test set (val-derived threshold) ...')
    with bench.track('keyword_baseline', stage='test_eval', device='cpu',
                     n_samples=len(test_df)):
        test_metrics = compute_metrics(best_test_scores, test_labels,
                                       f'Keyword ({best_variant})')

        # Also report at val-derived threshold
        preds_val_thresh = (best_test_scores >= val_thresh).astype(int)
        tp_vt = int(((preds_val_thresh == 1) & (test_labels == 1)).sum())
        fp_vt = int(((preds_val_thresh == 1) & (test_labels == 0)).sum())
        tn_vt = int(((preds_val_thresh == 0) & (test_labels == 0)).sum())
        fn_vt = int(((preds_val_thresh == 0) & (test_labels == 1)).sum())

        sens_vt = tp_vt / (tp_vt + fn_vt) if (tp_vt + fn_vt) > 0 else 0
        spec_vt = tn_vt / (tn_vt + fp_vt) if (tn_vt + fp_vt) > 0 else 0
        prec_vt = tp_vt / (tp_vt + fp_vt) if (tp_vt + fp_vt) > 0 else 0
        f1_vt = 2 * tp_vt / (2 * tp_vt + fp_vt + fn_vt) if (2 * tp_vt + fp_vt + fn_vt) > 0 else 0

    print(f'  Test AUPRC={test_metrics["AUPRC"]:.4f}  AUROC={test_metrics["AUROC"]:.4f}')
    print(f'  At val threshold {val_thresh:.4f}: '
          f'sens={sens_vt:.4f} spec={spec_vt:.4f} prec={prec_vt:.4f} F1={f1_vt:.4f}')

    # Criterion-level breakdown on test positives
    print('\n  Criterion-level signal (test set, PTSD+ notes):')
    pos_mask = test_labels == 1
    for crit in ['A', 'B', 'C', 'D', 'E', 'tx']:
        crit_pos_mean = test_crit[crit][pos_mask].mean()
        crit_neg_mean = test_crit[crit][~pos_mask].mean()
        ratio = crit_pos_mean / crit_neg_mean if crit_neg_mean > 0 else float('inf')
        print(f'    Criterion {crit}: pos_mean={crit_pos_mean:.2f}  '
              f'neg_mean={crit_neg_mean:.2f}  ratio={ratio:.2f}')

    # ── Save outputs ─────────────────────────────────────────────────────
    print('\n[5/5] Saving outputs ...')

    # Phrase weights
    lexicon_export = [
        {'pattern': pat, 'weight': w, 'criterion': c}
        for pat, w, c in PHRASE_LEXICON
    ]
    weights_out = {
        'best_variant': best_variant,
        'val_threshold_recall_85': round(float(val_thresh), 4),
        'n_phrases': len(PHRASE_LEXICON),
        'phrases': lexicon_export,
    }
    os.makedirs(os.path.dirname(WEIGHTS_JSON), exist_ok=True)
    with open(WEIGHTS_JSON, 'w') as f:
        json.dump(weights_out, f, indent=2)
    print(f'  Phrase weights → {WEIGHTS_JSON}')

    # Val results
    os.makedirs(os.path.dirname(VAL_RESULTS), exist_ok=True)
    with open(VAL_RESULTS, 'w') as f:
        f.write('Keyword Baseline — Validation Results\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Lexicon: {len(PHRASE_LEXICON)} phrases (DSM-5/PCL-5 derived)\n')
        f.write(f'Best variant: {best_variant}\n\n')
        for label, m in [('Raw weighted', val_metrics_raw),
                         ('TF-normalized', val_metrics_norm)]:
            f.write(f'{label}:\n')
            for k, v in m.items():
                f.write(f'  {k}: {v}\n')
            f.write('\n')
    print(f'  Val results    → {VAL_RESULTS}')

    # Test predictions
    os.makedirs(os.path.dirname(TEST_PREDS), exist_ok=True)
    pred_df = pd.DataFrame({
        'subject_id': test_df['subject_id'].values,
        'hadm_id': test_df['hadm_id'].values,
        'ptsd_label': test_labels,
        'keyword_score_raw': test_raw,
        'keyword_score_norm': test_norm,
        'keyword_pred_at_val_thresh': preds_val_thresh,
    })
    pred_df.to_csv(TEST_PREDS, index=False)
    print(f'  Test preds     → {TEST_PREDS}')

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('KEYWORD BASELINE SUMMARY')
    print('=' * 65)
    print(f'\n  Lexicon size: {len(PHRASE_LEXICON)} phrases')
    print(f'  Best variant: {best_variant}')
    print(f'  Training time: 0 seconds (no training required)')
    print(f'\n  {"Metric":<20} {"Val":>8} {"Test":>8}')
    print(f'  {"-"*20} {"-"*8} {"-"*8}')
    print(f'  {"AUPRC":<20} {best_val_metrics["AUPRC"]:>8.4f} {test_metrics["AUPRC"]:>8.4f}')
    print(f'  {"AUROC":<20} {best_val_metrics["AUROC"]:>8.4f} {test_metrics["AUROC"]:>8.4f}')
    print(f'  {"Sensitivity":<20} {best_val_metrics["sensitivity"]:>8.4f} {sens_vt:>8.4f}')
    print(f'  {"Specificity":<20} {best_val_metrics["specificity"]:>8.4f} {spec_vt:>8.4f}')
    print(f'  {"Precision":<20} {best_val_metrics["precision"]:>8.4f} {prec_vt:>8.4f}')
    print(f'  {"F1":<20} {best_val_metrics["F1"]:>8.4f} {f1_vt:>8.4f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
