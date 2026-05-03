"""
ewang163_ptsd_cross_model.py
============================
Cross-model comparison analysis. Reads existing per-patient prediction CSVs
and computes:

  1. All-pairs McNemar's test (continuity-corrected) over the 5 deployed
     models — each at its own val-derived threshold. Outputs a square p-value
     matrix and a TP/TN/FP/FN agreement matrix.

  2. Pairwise prediction-agreement table (% identical binary predictions,
     Cohen's kappa, Pearson correlation of predicted probabilities).

  3. Top-quintile rank-overlap matrix (how often the top 20% of patients by
     score overlap between models).

  4. Per-model NNS by subgroup (sex, age group, race binary, emergency)
     at deployment prevalences {1, 2, 5, 10%}.

  5. Best-of-K ensemble: simple max-pool over all 5 models' predicted
     probabilities, plus mean-pool. Reports whether either ensemble beats
     the best individual model on test AUPRC.

The 5 models compared:
    - PULSNAR Clinical Longformer (primary)
    - BioClinicalBERT (truncated)
    - BioClinicalBERT (chunk-pool)
    - Structured + LogReg
    - Keyword (DSM-5/PCL-5)

Outputs:
    results/metrics/ewang163_cross_model_mcnemar.csv
    results/metrics/ewang163_cross_model_agreement.csv
    results/metrics/ewang163_cross_model_topquintile_overlap.csv
    results/metrics/ewang163_subgroup_nns_by_model.csv
    results/metrics/ewang163_ensemble_results.json

Submit via SLURM:
    sbatch scripts/04_evaluation/ewang163_ptsd_cross_model.sh
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import (average_precision_score, roc_auc_score, cohen_kappa_score)
from scipy.stats import chi2 as chi2_dist, pearsonr

STUDENT_DIR     = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_SPLITS     = f'{STUDENT_DIR}/data/splits'
DATA_COHORT     = f'{STUDENT_DIR}/data/cohort'
RESULTS_PRED    = f'{STUDENT_DIR}/results/predictions'
RESULTS_METRICS = f'{STUDENT_DIR}/results/metrics'

TEST_PARQUET    = f'{DATA_SPLITS}/ewang163_split_test.parquet'
ADM_PARQUET     = f'{DATA_COHORT}/ewang163_ptsd_adm_extract.parquet'

# Model name → (test_predictions_csv, prob_column, val_threshold).
# Thresholds are val-derived (recall>=0.85). Longformer comes from
# evaluation_results_pulsnar.json; BERT thresholds from full_eval_summary;
# structured/keyword from evaluation_results.json::val_thresholds.
MODELS = {
    'longformer_pulsnar': {
        'csv': f'{RESULTS_PRED}/ewang163_longformer_test_predictions_pulsnar.csv',
        'prob_col': 'predicted_prob', 'threshold_key': 'longformer_pulsnar',
    },
    'bert_trunc': {
        'csv': f'{RESULTS_PRED}/ewang163_bioclinbert_trunc_test_predictions.csv',
        'prob_col': 'predicted_prob', 'threshold_key': 'bert_trunc',
    },
    'bert_chunkpool': {
        'csv': f'{RESULTS_PRED}/ewang163_bioclinbert_chunkpool_test_predictions.csv',
        'prob_col': 'predicted_prob', 'threshold_key': 'bert_chunkpool',
    },
    'structured': {
        'csv': None,  # no per-patient CSV — score column reconstructed below
        'prob_col': None, 'threshold_key': 'structured',
    },
    'keyword': {
        'csv': f'{RESULTS_PRED}/ewang163_keyword_test_predictions.csv',
        'prob_col': 'keyword_score_norm',  # score, not prob — but rank-only ok
        'threshold_key': 'keyword',
    },
}

PREVS = (0.01, 0.02, 0.05, 0.10)


# ── Threshold loading ─────────────────────────────────────────────────────
def load_thresholds():
    """Pull val-derived thresholds from each model's eval JSON."""
    out = {}

    # PULSNAR Longformer
    p = f'{RESULTS_METRICS}/ewang163_evaluation_results_pulsnar.json'
    with open(p) as f:
        out['longformer_pulsnar'] = float(json.load(f)['val_threshold'])

    # BERT thresholds — both modes — from the full_eval summary written by
    # ewang163_ptsd_bert_full_eval.py
    p = f'{RESULTS_METRICS}/ewang163_bioclinbert_full_eval_summary.json'
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        out['bert_trunc'] = float(s['trunc']['val_threshold'])
        out['bert_chunkpool'] = float(s['chunkpool']['val_threshold'])
    else:
        # Fallback to legacy thresholds shared between both modes
        with open(f'{RESULTS_METRICS}/ewang163_evaluation_results.json') as f:
            r = json.load(f)
        out['bert_trunc'] = float(r['val_thresholds']['bioclinbert'])
        out['bert_chunkpool'] = float(r['val_thresholds']['bioclinbert'])

    # Structured / Keyword from main eval
    with open(f'{RESULTS_METRICS}/ewang163_evaluation_results.json') as f:
        r = json.load(f)
    out['structured'] = float(r['val_thresholds']['structured'])
    out['keyword']    = float(r['val_thresholds']['keyword'])
    return out


# ── Score loading ─────────────────────────────────────────────────────────
def load_structured_scores(test_df):
    """Build / load per-patient structured-model probabilities for the test set.

    Cached at results/predictions/ewang163_structured_test_predictions.csv.
    If absent, regenerates by importing build_structured_features from
    ewang163_ptsd_evaluate (which streams diagnoses_icd.csv + prescriptions.csv)
    and applying the saved logreg pickle.
    """
    p = f'{RESULTS_PRED}/ewang163_structured_test_predictions.csv'
    if os.path.exists(p):
        return pd.read_csv(p)

    print('  generating structured per-patient predictions (~1 min)...')
    import pickle
    sys.path.insert(0,
        '/oscar/data/class/biol1595_2595/students/ewang163/scripts/04_evaluation')
    from ewang163_ptsd_evaluate import build_structured_features  # noqa: E402

    adm = pd.read_parquet(ADM_PARQUET)
    X, _ = build_structured_features(test_df, adm)
    with open(f'{STUDENT_DIR}/models/ewang163_structured_logreg.pkl', 'rb') as f:
        struct_lr = pickle.load(f)
    probs = struct_lr.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        'subject_id': test_df['subject_id'].values,
        'hadm_id': test_df['hadm_id'].values,
        'ptsd_label': test_df['ptsd_label'].values,
        'predicted_prob': probs,
    })
    out.to_csv(p, index=False)
    print(f'  → {p}')
    return out


def load_predictions(test_df):
    """For each model in MODELS, return aligned probs / preds / labels.

    The test split has 1,551 rows / 1,207 unique subjects — some patients
    contribute multiple admissions. Two cases:
      (a) Prediction CSV has both `subject_id` AND `hadm_id` (BERT,
          structured, keyword) — these are 1:1 with admissions, merge on both.
      (b) Prediction CSV has only `subject_id` (PULSNAR Longformer's
          pulsnar_reeval output) — the rows were generated by iterating
          `test_df` in original order so they align positionally with
          test_df. We trust positional alignment but verify subject_id
          matches as a sanity check.
    """
    prob_arrays = {}
    for name, cfg in MODELS.items():
        if name == 'structured':
            d = load_structured_scores(test_df)
            if d is None:
                print(f'  WARNING: structured per-patient predictions not found, skipping')
                continue
            prob_col = 'predicted_prob'
        else:
            d = pd.read_csv(cfg['csv'])
            prob_col = cfg['prob_col']

        if 'hadm_id' in d.columns:
            merged = test_df[['subject_id', 'hadm_id']].merge(
                d[['subject_id', 'hadm_id', prob_col]],
                on=['subject_id', 'hadm_id'], how='left')
            probs = merged[prob_col].values.astype(float)
        else:
            # Positional alignment with test_df row order — verify subject_id
            if len(d) != len(test_df):
                raise ValueError(
                    f'{name} predictions row count {len(d)} != test_df {len(test_df)}'
                )
            if not (d['subject_id'].values == test_df['subject_id'].values).all():
                raise ValueError(f'{name} subject_id order does not match test_df')
            probs = d[prob_col].values.astype(float)

        prob_arrays[name] = np.nan_to_num(probs, nan=0.0)
    return prob_arrays


def attach_demographics(df):
    adm = pd.read_parquet(ADM_PARQUET)
    hadms = set(df['hadm_id'].tolist())
    adm = adm[adm['hadm_id'].isin(hadms)].copy()
    adm = adm.sort_values('admittime').drop_duplicates('subject_id', keep='first')
    adm['race_cat'] = adm['race'].astype(str).str.upper().apply(
        lambda r: 'White' if 'WHITE' in r else
                  ('Black' if ('BLACK' in r or 'AFRICAN' in r) else
                   ('Hispanic' if ('HISPANIC' in r or 'LATINO' in r) else
                    ('Asian' if 'ASIAN' in r else 'Other/Unknown'))))

    def _age_decade(a):
        if pd.isna(a): return 'Other'
        a = int(a)
        if 20 <= a <= 29: return '20s'
        if 30 <= a <= 39: return '30s'
        if 40 <= a <= 49: return '40s'
        if 50 <= a <= 59: return '50s'
        return 'Other'
    adm['age_group'] = adm['age_at_admission'].apply(_age_decade)
    adm['is_emergency'] = adm['admission_type'].str.upper().str.contains('EMER', na=False)
    demo = adm.set_index('subject_id')[
        ['gender', 'race_cat', 'age_group', 'is_emergency']
    ].to_dict('index')
    df = df.copy()
    df['gender'] = df['subject_id'].map(lambda s: demo.get(s, {}).get('gender', 'U'))
    df['race_cat'] = df['subject_id'].map(lambda s: demo.get(s, {}).get('race_cat', 'O'))
    df['age_group'] = df['subject_id'].map(lambda s: demo.get(s, {}).get('age_group', 'Other'))
    df['is_emergency'] = df['subject_id'].map(lambda s: demo.get(s, {}).get('is_emergency', False))
    df['race_binary'] = df['race_cat'].apply(lambda r: 'White' if r == 'White' else 'Non-White')
    return df


# ── McNemar ───────────────────────────────────────────────────────────────
def mcnemar(a, b, labels):
    """McNemar's test comparing two binary prediction vectors."""
    correct_a = (a == labels).astype(int)
    correct_b = (b == labels).astype(int)
    bb = int(((correct_a == 1) & (correct_b == 0)).sum())
    cc = int(((correct_a == 0) & (correct_b == 1)).sum())
    if bb + cc == 0:
        return 1.0, bb, cc
    chi2 = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    return float(1 - chi2_dist.cdf(chi2, df=1)), bb, cc


def kappa(a, b):
    return float(cohen_kappa_score(a, b))


# ── Per-prevalence NNS ────────────────────────────────────────────────────
def nns_at_prev(sens, spec, prev):
    eps = 1e-9
    ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev) + eps)
    return 1 / ppv if ppv > 0 else float('inf')


def main():
    print('=' * 70)
    print('PTSD NLP — Cross-Model Comparison')
    print('=' * 70, flush=True)

    test_df = pd.read_parquet(TEST_PARQUET)
    labels = test_df['ptsd_label'].values.astype(int)

    print('\n[1/6] Loading thresholds')
    thresholds = load_thresholds()
    for k, v in thresholds.items():
        print(f'  {k:>22}  threshold = {v:.4f}')

    print('\n[2/6] Loading per-patient predictions')
    probs = load_predictions(test_df)
    models = list(probs.keys())
    print(f'  Models loaded: {models}')

    # Binary predictions
    binary = {m: (probs[m] >= thresholds[m]).astype(int) for m in models}

    # Per-model AUPRC (sanity)
    print('\n  Test AUPRC sanity check:')
    for m in models:
        ap = average_precision_score(labels, probs[m])
        ar = roc_auc_score(labels, probs[m])
        print(f'    {m:<22}  AUPRC={ap:.4f}  AUROC={ar:.4f}')

    # ── Step 3: All-pairs McNemar + agreement ──────────────────────────────
    print('\n[3/6] All-pairs McNemar + agreement')
    rows = []
    for a, b in combinations(models, 2):
        p, bb, cc = mcnemar(binary[a], binary[b], labels)
        agree = float((binary[a] == binary[b]).mean())
        kap = kappa(binary[a], binary[b])
        # Pearson correlation between continuous probs
        try:
            rho, _ = pearsonr(probs[a], probs[b])
        except ValueError:
            rho = float('nan')
        rows.append({
            'model_a': a, 'model_b': b,
            'mcnemar_p': p, 'b_only_A_correct': bb, 'c_only_B_correct': cc,
            'agreement_pct': round(agree * 100, 2),
            'cohens_kappa': round(kap, 4),
            'pearson_r_on_probs': round(float(rho), 4),
        })
    pd.DataFrame(rows).to_csv(
        f'{RESULTS_METRICS}/ewang163_cross_model_mcnemar.csv', index=False)
    print(f'  → ewang163_cross_model_mcnemar.csv ({len(rows)} pairs)')

    # ── Step 4: Top quintile overlap ───────────────────────────────────────
    print('\n[4/6] Top-quintile rank overlap')
    n_test = len(labels)
    top_n = max(1, n_test // 5)
    top_idx = {m: set(np.argsort(probs[m])[::-1][:top_n].tolist()) for m in models}
    rows = []
    for a, b in combinations(models, 2):
        overlap = len(top_idx[a] & top_idx[b]) / top_n
        rows.append({
            'model_a': a, 'model_b': b,
            'top_quintile_size': top_n,
            'overlap_jaccard': round(len(top_idx[a] & top_idx[b]) /
                                     len(top_idx[a] | top_idx[b]), 4),
            'overlap_fraction': round(overlap, 4),
        })
    pd.DataFrame(rows).to_csv(
        f'{RESULTS_METRICS}/ewang163_cross_model_topquintile_overlap.csv',
        index=False)

    # ── Step 5: Subgroup NNS by model ──────────────────────────────────────
    print('\n[5/6] Subgroup NNS (per model x deployment prevalence)')
    test_with_demo = attach_demographics(test_df)
    rows = []
    groupings = [('sex', 'gender'), ('age_group', 'age_group'),
                 ('race_binary', 'race_binary'),
                 ('emergency', 'is_emergency')]
    for m in models:
        thr = thresholds[m]
        preds = (probs[m] >= thr).astype(int)
        for gname, col in groupings:
            for val in sorted(test_with_demo[col].unique()):
                mask = (test_with_demo[col] == val).values
                n = int(mask.sum())
                n_pos = int(labels[mask].sum())
                if n_pos == 0 or n_pos == n:
                    continue
                tp = int(((preds[mask] == 1) & (labels[mask] == 1)).sum())
                fp = int(((preds[mask] == 1) & (labels[mask] == 0)).sum())
                tn = int(((preds[mask] == 0) & (labels[mask] == 0)).sum())
                fn = int(((preds[mask] == 0) & (labels[mask] == 1)).sum())
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                row = {'model': m, 'group': gname, 'value': str(val),
                       'n': n, 'n_pos': n_pos,
                       'sens': round(sens, 4), 'spec': round(spec, 4)}
                for prev in PREVS:
                    row[f'NNS_{int(prev * 100)}pct'] = round(
                        nns_at_prev(sens, spec, prev), 2)
                rows.append(row)
    pd.DataFrame(rows).to_csv(
        f'{RESULTS_METRICS}/ewang163_subgroup_nns_by_model.csv', index=False)

    # ── Step 6: Ensemble (max-pool / mean-pool over normalised probs) ──────
    print('\n[6/6] Ensemble experiments')
    # Normalise each model's probs to [0,1] via empirical CDF (so different
    # threshold scales don't dominate the max).
    normed = {}
    for m in models:
        ranks = pd.Series(probs[m]).rank(method='average', pct=True).values
        normed[m] = ranks
    M = np.stack([normed[m] for m in models], axis=1)
    max_ens = M.max(axis=1)
    mean_ens = M.mean(axis=1)
    auprc_max = float(average_precision_score(labels, max_ens))
    auprc_mean = float(average_precision_score(labels, mean_ens))
    auroc_max = float(roc_auc_score(labels, max_ens))
    auroc_mean = float(roc_auc_score(labels, mean_ens))
    individual = {m: {'AUPRC': float(average_precision_score(labels, probs[m])),
                      'AUROC': float(roc_auc_score(labels, probs[m]))}
                  for m in models}
    best_indiv = max(individual.values(), key=lambda r: r['AUPRC'])

    summary = {
        'n_test': len(labels),
        'n_pos': int(labels.sum()),
        'individual': individual,
        'ensemble_max_pool_rank': {'AUPRC': round(auprc_max, 4),
                                   'AUROC': round(auroc_max, 4)},
        'ensemble_mean_pool_rank': {'AUPRC': round(auprc_mean, 4),
                                    'AUROC': round(auroc_mean, 4)},
        'best_individual_AUPRC': round(best_indiv['AUPRC'], 4),
        'ensemble_max_beats_best': auprc_max > best_indiv['AUPRC'],
        'ensemble_mean_beats_best': auprc_mean > best_indiv['AUPRC'],
    }
    with open(f'{RESULTS_METRICS}/ewang163_ensemble_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Best individual AUPRC: {summary["best_individual_AUPRC"]}')
    print(f'  Max-pool ensemble AUPRC: {summary["ensemble_max_pool_rank"]["AUPRC"]}')
    print(f'  Mean-pool ensemble AUPRC: {summary["ensemble_mean_pool_rank"]["AUPRC"]}')

    print('\nDone.')


if __name__ == '__main__':
    main()
