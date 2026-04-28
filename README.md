# PTSD Underdiagnosis Detection in Clinical Notes

A Clinical Longformer model trained to detect undercoded PTSD in inpatient discharge
notes from MIMIC-IV, using positive-unlabeled (PU) learning to handle the absence of
confirmed negatives.

---

## Background

PTSD is systematically undercoded in inpatient settings (~1% ICD coding rate vs. an
estimated 20%+ true prevalence in trauma-exposed populations). Treating non-coded
patients as confirmed negatives is methodologically incorrect. This project applies
Kiryo et al. (2017) non-negative PU risk estimation so the model is trained without
assuming unlabeled patients are PTSD-free.

---

## Installation

### Requirements

- Python 3.13+
- Packages: `pandas`, `numpy`, `transformers`, `torch`, `scikit-learn`, `captum`

### Setup (Oscar HPC)

```bash
# Activate the existing virtual environment
source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate

# Install additional packages if needed
pip install transformers torch scikit-learn captum
```

### Paths

| Variable      | Path                                                  |
|---------------|-------------------------------------------------------|
| `STUDENT_DIR` | `/oscar/data/class/biol1595_2595/students/ewang163/`  |
| `MIMIC_ROOT`  | `/oscar/data/shared/ursa/mimic-iv` (read-only)        |
| `HOSP`        | `/oscar/data/shared/ursa/mimic-iv/hosp/3.1/`          |
| `NOTES`       | `/oscar/data/shared/ursa/mimic-iv/note/2.2/`          |

---

## Usage

All scripts must be submitted via SLURM, not run interactively.

### CPU jobs (data processing)

```bash
sbatch --partition=batch --mem=16G --time=2:00:00 \
       --output=ewang163_%j.out --wrap="python ewang163_<script>.py"
```

### GPU jobs (model training)

```bash
sbatch ewang163_ptsd_train_longformer.py   # SLURM header is embedded in the script
```

### Pipeline order

Run scripts in the order listed in [Programs](#programs). Each stage saves its
output to `STUDENT_DIR` before the next stage begins, so any stage can be re-run
independently.

---

## Programs

### Cohort Construction & Data Extraction

| Script | Description |
|--------|-------------|
| `ewang163_ptsd_table1.py` | Builds three study cohorts from MIMIC-IV ICD codes and prescriptions, computes all Table 1 demographic and clinical characteristics, and writes results to CSV and human-readable summary. |
| `ewang163_ptsd_cohort_sets.py` | Reconstructs the three cohort subject ID sets (PTSD+, pharmacological proxy, matched unlabeled) via streaming reads of MIMIC-IV source files. |
| `ewang163_ptsd_admissions_extract.py` | Extracts admission rows for all cohort groups, attaches patient demographics, computes age at admission, and flags the index admission for each patient. |
| `ewang163_ptsd_notes_extract.py` | Streams the 3.3 GB MIMIC-IV discharge note file and writes section-filtered notes (HPI, social history, PMH, brief hospital course) for all three cohort groups. |

### Corpus Assembly & Splits

| Script | Description |
|--------|-------------|
| `ewang163_ptsd_corpus_build.py` | Assembles the training corpus from extracted notes, applies Ablation 1 PTSD-string masking to fallback notes, and holds the proxy group out as a separate validation file. |
| `ewang163_ptsd_splits.py` | Creates patient-level 80/10/10 train/validation/test splits stratified by PTSD label, ensuring all admissions for a given patient land in the same split. |

### Model Training

| Script | Description |
|--------|-------------|
| `ewang163_ptsd_train_longformer.py` | Fine-tunes Clinical Longformer (`yikuan8/Clinical-Longformer`, 4 096 tokens) with the Kiryo et al. PU non-negative risk loss; saves the best checkpoint by validation AUPRC. |
| `ewang163_ptsd_train_bioclinbert.py` | Fine-tunes BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`, 512 tokens) with the same PU loss; runs McNemar's test against Longformer on the test set. |
| `ewang163_ptsd_train_tfidf.py` | TF-IDF + logistic regression text baseline with PU-friendly `class_weight='balanced'`; tunes regularization C on validation AUPRC. |
| `ewang163_ptsd_train_structured.py` | Logistic regression baseline on structured features only (demographics, prior comorbidities, prior medications); tunes C on validation AUPRC and reports feature coefficients. |
| `ewang163_ptsd_train_keyword.py` | Zero-training naive keyword/phrase-lookup baseline. Scores notes using a hand-curated DSM-5/PCL-5 phrase lexicon (62 weighted patterns across criteria A–E + treatment signals). Two scoring variants (raw weighted count, TF-normalized); best variant selected on validation AUPRC. Compares speed and accuracy against NLP methods — runs in seconds with no GPU. |
| `ewang163_ptsd_train_pulsnar.py` | **Fix 3**: PULSNAR SAR-PU integration. Estimates propensity scores P(coded\|features) via logistic regression, reweights the nnPU positive loss term by 1/e(x), and optionally uses PULSNAR library for class-prior estimation. Corrects the SCAR violation in vanilla nnPU. |
| `ewang163_ptsd_pip_sweep.sh` | **Fix 2**: Shell script that submits 7 parallel SLURM GPU jobs sweeping π_p ∈ {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25}. Each job trains a full Longformer with a different class prior. |
| `ewang163_ptsd_pip_sweep_eval.py` | Aggregates the π_p sweep results: runs proxy validation on each checkpoint, picks the best π_p by proxy-vs-unlabeled Mann-Whitney AUC, and produces a sweep plot. |

### Shared Utilities

| Script | Description |
|--------|-------------|
| `scripts/common/ewang163_bench_utils.py` | Runtime benchmarking context manager. Captures wall-clock time, CPU time, and peak memory per pipeline stage; appends rows to `results/metrics/ewang163_runtime_benchmarks.csv`. Used by all training and evaluation scripts for compute-vs-accuracy comparison. |

### Evaluation & Analysis

| Script | Description |
|--------|-------------|
| `ewang163_ptsd_evaluate.py` | Evaluates all five models (Longformer, BioClinicalBERT, TF-IDF, structured, keyword) on the held-out test set. **Fix 4 applied**: thresholds are derived from the validation set to eliminate selection-on-test bias. Reports AUPRC, AUROC, sensitivity/specificity/precision/F1, plus clinical utility metrics (LR+, LR-, DOR, alert rate, workup reduction, NNS/NNE at deployment prevalences), and subgroup analysis by sex, age, race, and admission type. |
| `ewang163_ptsd_calibration.py` | Fits Platt scaling on validation, then applies **Fix 5** Elkan-Noto PU correction (divides by c = P(s=1\|y=1)) so predicted probabilities approximate P(PTSD=1) not P(coded=1). Reports ECE for raw, Platt-scaled, and Elkan-Noto variants. |
| `ewang163_ptsd_decision_curves.py` | Performs Decision Curve Analysis at 2% and 5% deployment prevalences; recalibrates probabilities and plots net benefit curves over threshold range [0.01, 0.40]. |
| `ewang163_ptsd_proxy_validation.py` | External validation on the held-out pharmacological proxy group; compares Longformer scores between proxy patients and a random unlabeled sample using Mann-Whitney U and AUC. |
| `ewang163_ptsd_ablations.py` | Runs two ablation experiments — Ablation 1 (PTSD string masking) and Ablation 2 (PMH section removal) — to verify the model is not exploiting label-leaking text. |
| `ewang163_ptsd_attribution.py` | Computes Integrated Gradients (Captum `LayerIntegratedGradients`) on 50 high-confidence true positives; aggregates token-level attribution by note section. |
| `ewang163_ptsd_attribution_v2.py` | Updated version of the attribution analysis with revised aggregation and visualization logic. |
| `ewang163_ptsd_error_analysis.py` | Samples 25 false positives and 25 false negatives, writes annotated notes to text files, and computes aggregate statistics (mean predicted probability, note length, top TF-IDF terms). |
| `ewang163_ptsd_specificity.py` | Specificity check: retrains Clinical Longformer on PTSD+ vs. age/sex-matched MDD/anxiety controls using standard cross-entropy to assess whether the model learned PTSD-specific signal. |
| `ewang163_ptsd_fairness.py` | **Fix 9**: Statistically defensible fairness reporting. Calibration-in-the-large per subgroup, equal opportunity difference, and bootstrap 95% CI on AUPRC (only reported if CI width < 0.15). White vs. non-White primary contrast. |
| `ewang163_ptsd_chart_review_packet.py` | **Fix 11** (partial): Prepares a chart-review packet for the top-50 model-flagged unlabeled patients with de-identified notes and a clinician rating form. Does not perform the review — requires clinician/advisor. |

---

## Methodology Fixes Applied

All fixes are derived from `methodology_fix_plans.md` and cross-referenced against published literature.

| Fix | Description | Status |
|-----|-------------|--------|
| **Fix 1** | Mask ALL PTSD+ notes (pre-dx + fallback), not just fallback | Code applied in `corpus_build.py` |
| **Fix 2** | π_p sweep over 7 values, pick best by proxy AUC | Scripts created: `pip_sweep.sh`, `pip_sweep_eval.py` |
| **Fix 3** | PULSNAR SAR-PU with propensity-weighted nnPU loss | Script created: `train_pulsnar.py` |
| **Fix 4** | Move threshold selection from test to validation | Applied in `evaluate.py`, `proxy_validation.py` |
| **Fix 5** | Elkan-Noto calibration correction (P(PTSD=1) not P(coded=1)) | Applied in `calibration.py` |
| **Fix 6** | Ramola PU-corrected metrics as lower bounds | Applied in `evaluate.py` |
| **Fix 7** | Temporal split (pre/post-2015) | Added `--temporal` flag to `splits.py` |
| **Fix 8** | BioClinicalBERT chunk-and-pool inference | Added to `evaluate.py` |
| **Fix 9** | Calibration-in-the-large + bootstrap fairness | New script: `fairness.py` |
| **Fix 10** | IG at full 4096 context (was 1024) | Updated `attribution_v2.py` |
| **Fix 11** | Chart review packet for top-50 flagged patients | New script: `chart_review_packet.py` |

---

## Cohort Summary

| Group | Definition | N |
|-------|-----------|---|
| PTSD+ (labeled positives) | Any admission with ICD-10 F43.1x or ICD-9 309.81 | 5,711 |
| Pharmacological proxy (external validation only) | Prazosin + SSRI/SNRI overlap ≤ 180 days, no exclusion ICD | 163 |
| Matched unlabeled pool | All remaining patients, 3:1 matched on age decade × sex | 17,133 |

---

## Key Design Decisions

- **PU Learning:** Kiryo et al. (2017) non-negative risk estimator as primary;
  PULSNAR (Kumar & Lambert 2024) propensity-weighted variant to address the SCAR
  violation (PTSD coding is biased toward younger women with prior psychiatric contact).
- **Label leakage prevention (Fix 1):** PTSD-string masking applied to ALL positive
  notes (pre-dx + fallback), not just fallback. Section filtering removes diagnostic sections.
- **Threshold selection (Fix 4):** Operating thresholds derived from validation, not test.
- **Primary metric:** AUPRC, labelled as PU lower bound (Fix 6, Ramola et al. 2019).
  Proxy Mann-Whitney AUC is co-headline (only PU-uncontaminated metric).
- **Clinical utility:** LR+, LR-, DOR, alert rate, workup reduction, NNS/NNE at
  deployment prevalences; Elkan-Noto-corrected probabilities (Fix 5).
- **Naive baseline comparison:** DSM-5/PCL-5 keyword lookup establishes the floor
  performance achievable without ML. Speed-vs-accuracy comparison across all models.
- **Explainability:** Integrated Gradients at full 4096 context (Fix 10), not
  attention weights (Jain & Wallace 2019).
- **Patient-level splits:** A patient's data never spans train/val/test boundaries.
  Temporal split available (Fix 7) for generalization testing.
- **Fairness (Fix 9):** Calibration-in-the-large + equal opportunity difference +
  bootstrap CI, not unreliable per-subgroup AUPRC at small n.

---

## Project Status

All 11 methodology fixes from `methodology_fix_plans.md` are implemented. Pipeline
stages are ready for execution via SLURM. Recommended execution order:

1. `corpus_build.py` (Fix 1 masking)
2. `splits.py` (random) + `splits.py --temporal` (Fix 7)
3. `train_keyword.py` (CPU, seconds)
4. `train_longformer.py` (GPU, ~7.5h) or `pip_sweep.sh` (Fix 2, ~2 GPU-days)
5. `train_pulsnar.py` (Fix 3, GPU, ~9h)
6. `evaluate.py` (all fixes: 4, 6, 8 applied)
7. `calibration.py` (Fix 5)
8. `fairness.py` (Fix 9)
9. `attribution_v2.py` (Fix 10, GPU)
10. `chart_review_packet.py` (Fix 11, CPU)
