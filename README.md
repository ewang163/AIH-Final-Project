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

### Evaluation & Analysis

| Script | Description |
|--------|-------------|
| `ewang163_ptsd_evaluate.py` | Evaluates all four models on the held-out test set; reports AUPRC, AUROC, sensitivity/specificity/precision/F1 at recall ≥ 0.85, and subgroup metrics by sex, age, race, and admission type. |
| `ewang163_ptsd_calibration.py` | Fits Platt scaling on the validation set to calibrate raw PU-loss probabilities; plots calibration curves with 95% CI and computes Expected Calibration Error before and after scaling. |
| `ewang163_ptsd_decision_curves.py` | Performs Decision Curve Analysis at 2% and 5% deployment prevalences; recalibrates probabilities and plots net benefit curves over threshold range [0.01, 0.40]. |
| `ewang163_ptsd_proxy_validation.py` | External validation on the held-out pharmacological proxy group; compares Longformer scores between proxy patients and a random unlabeled sample using Mann-Whitney U and AUC. |
| `ewang163_ptsd_ablations.py` | Runs two ablation experiments — Ablation 1 (PTSD string masking) and Ablation 2 (PMH section removal) — to verify the model is not exploiting label-leaking text. |
| `ewang163_ptsd_attribution.py` | Computes Integrated Gradients (Captum `LayerIntegratedGradients`) on 50 high-confidence true positives; aggregates token-level attribution by note section. |
| `ewang163_ptsd_attribution_v2.py` | Updated version of the attribution analysis with revised aggregation and visualization logic. |
| `ewang163_ptsd_error_analysis.py` | Samples 25 false positives and 25 false negatives, writes annotated notes to text files, and computes aggregate statistics (mean predicted probability, note length, top TF-IDF terms). |
| `ewang163_ptsd_specificity.py` | Specificity check: retrains Clinical Longformer on PTSD+ vs. age/sex-matched MDD/anxiety controls using standard cross-entropy to assess whether the model learned PTSD-specific signal. |

---

## Cohort Summary

| Group | Definition | N |
|-------|-----------|---|
| PTSD+ (labeled positives) | Any admission with ICD-10 F43.1x or ICD-9 309.81 | 5,711 |
| Pharmacological proxy (external validation only) | Prazosin + SSRI/SNRI overlap ≤ 180 days, no exclusion ICD | 163 |
| Matched unlabeled pool | All remaining patients, 3:1 matched on age decade × sex | 17,133 |

---

## Key Design Decisions

- **PU Learning:** Kiryo et al. (2017) non-negative risk estimator. The SCAR
  assumption (Elkan & Noto 2008) is violated here because PTSD coding is biased
  toward younger women with prior psychiatric contact.
- **Label leakage prevention:** Pre-diagnosis notes used as the primary signal for
  multi-admission patients; section-filtered index notes with PTSD-string masking
  used as a fallback.
- **Primary metric:** AUPRC (AUROC is misleading under class imbalance).
- **Explainability:** Integrated Gradients, not attention weights (Jain & Wallace
  2019 show attention ≠ explanation).
- **Patient-level splits:** A patient's data never spans train/val/test boundaries.

---

## Project Status

Active development — all pipeline stages (Steps 1–11) are implemented.
