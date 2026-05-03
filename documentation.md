# PTSD Underdiagnosis Detection — Documentation Index

**Author:** Eric Wang (ewang163), Brown University, AIH 2025 (Spring 2026)
**Source data:** MIMIC-IV v3.1 (BIDMC, 2008–2019) on Brown's Oscar HPC cluster
**Final deployment model:** Clinical Longformer fine-tuned with PULSNAR
propensity-weighted nnPU loss (α = 0.196), trained on the section-filtered, PTSD-string-masked corpus.

---

## What lives where

The project documentation has been consolidated into a small set of canonical
files. This file is a redirect index — it used to be a near-duplicate of the
project writeup, and the duplicated content is now maintained in one place.

| Looking for... | Read this |
|---|---|
| Project overview, results headlines, repo layout, pipeline order | `README.md` |
| Full methods (cohort, models, evaluation), all results, discussion, limitations | `ewang163_project_writeup.md` |
| Multi-model comparison (discrimination + calibration + DCA + fairness + IG + cross-model agreement) | `ewang163_model_comparison.md` |
| Why PULSNAR Longformer was chosen as primary | `ewang163_model_selection_memo.md` |
| Authoritative project specification (cohort definitions, MIMIC quirks, hard rules) | `CLAUDE.md` |
| Original methodology fix plans (literature-grounded) | `methodology_fix_plans.md` |
| Methodology fix execution results | `ewang163_methodology_fixes_results.md` |

---

## Project at a glance

**Clinical motivation.** PTSD is systematically undercoded in inpatient settings
(~1 % ICD coding rate vs. ≥ 20 % estimated true prevalence in trauma-exposed
populations). Treating non-coded patients as confirmed negatives is wrong by
assumption — the entire study design is built around this problem.

**Approach.** Clinical Longformer (4,096 tokens) fine-tuned with PULSNAR
(Kumar & Lambert 2024) — a SAR-aware variant of nnPU that up-weights
under-coded positives via propensity reweighting. Compared head-to-head on
identical patient-level splits with BioClinicalBERT (truncated 512 +
chunk-and-pool 512 × 256), structured features + logistic regression, and a
zero-training DSM-5/PCL-5 keyword baseline.

---

## Key results

All on the held-out test set (n = 1,551 patients, 660 PTSD+) under val-derived
thresholds.

### Discrimination

| Model | AUPRC | AUROC | F1 | NNS @ 2 % | McNemar p vs PULSNAR |
|---|---:|---:|---:|---:|---:|
| **PULSNAR Clinical Longformer** | **0.8848** | **0.8904** | 0.772 | 15.8 | — |
| BioClinicalBERT (chunk-pool) | 0.8775 | 0.8853 | **0.785** | **14.2** | 0.107 (n.s.) |
| BioClinicalBERT (truncated) | 0.8576 | 0.8656 | 0.751 | 17.2 | 0.043 |
| Structured + LogReg | 0.6833 | 0.7310 | 0.610 | 36.7 | < 1e-300 |
| Keyword (DSM-5/PCL-5) | 0.5096 | 0.6190 | 0.597 | 50.0 | < 1e-300 |

PULSNAR Longformer wins discrimination. BERT chunk-pool is statistically
indistinguishable from Longformer on McNemar (p = 0.107) and wins F1 / NNS at
the operating point — but requires post-hoc Platt calibration before
deployment because raw ECE is 0.51 vs. Longformer's 0.088.

### Non-circular validation (proxy)

Pharmacological proxy patients (n = 102, prazosin + SSRI/SNRI history, no ICD
PTSD code) score substantially higher than 500 random unlabeled controls
across all three text models:

| Model | Proxy MW AUC | MW p |
|---|---:|---:|
| PULSNAR Longformer | **0.7701** | 3.8e-18 |
| BioClinicalBERT (truncated) | 0.7442 | 3.7e-15 |
| BioClinicalBERT (chunk-pool) | 0.7333 | 5.3e-14 |

The model — never shown a single proxy patient during training — assigns
~6× the median probability to patients whose pharmacotherapy is consistent
with PTSD treatment. This is the project's strongest single piece of
non-circular validity evidence.

### Specificity vs. psychiatric controls

A separate Longformer trained PTSD+ vs. age/sex-matched MDD/anxiety controls
(standard cross-entropy) reaches AUPRC = 0.91 — PTSD-specific signal is
recoverable above-and-beyond generic "psychiatric admission" language.

---

## Pipeline order (reproducing from raw MIMIC-IV)

The full ordering is in `README.md` §5. Headlines:

1. Cohort + Table 1 — `01_cohort/`
2. Notes extraction + corpus build + splits — `02_corpus/`
3. **Train PULSNAR Longformer** (primary) — `03_training/ewang163_ptsd_train_pulsnar.py`
4. Train BioClinicalBERT, structured, keyword, specificity baselines — `03_training/`
5. **Evaluate PULSNAR Longformer** — `04_evaluation/ewang163_ptsd_pulsnar_reeval.py`
6. **Evaluate BioClinicalBERT (both modes)** — `04_evaluation/ewang163_ptsd_bert_full_eval.py`
7. Run all attribution + cross-model + ancillary scripts — `04_evaluation/`

---

## Reproducibility

Random seeds pinned to 42 throughout (matching, sampling, splits, model training, bootstrap). All design decisions, validated cohort definitions, ICD code lists, drug lists, MIMIC-IV data quirks, and bug fixes are documented in `CLAUDE.md`. SLURM job logs for every run are preserved under `logs/`.
