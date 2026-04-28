# PTSD Underdiagnosis Detection in MIMIC-IV Discharge Notes

**Author:** Eric Wang (ewang163), Brown University, AIH 2025 (Spring 2026)
**Source data:** MIMIC-IV v3.1 (BIDMC, 2008–2019)
**Final deployment model:** Clinical Longformer fine-tuned with Kiryo non-negative PU loss (π_p = 0.25) on a section-filtered, label-masked corpus.

---

## 1. Project Overview

### Clinical motivation

PTSD is systematically undercoded in inpatient settings. Inpatient teams aren't trained to elicit DSM-5 PTSD criteria, and PTSD has no laboratory or imaging marker — it has to be asked about, with knowledge of the patient's psychological and social history. Estimated true prevalence in trauma-exposed inpatients is ≥ 20%, but the ICD coding rate in MIMIC-IV is closer to 1%. Patients who are missed lose access to targeted treatment (trauma-focused psychotherapy, prazosin for nightmares, SSRI/SNRI), generate avoidable readmissions, and lose continuity of trauma history. A screening tool that flags patients whose discharge notes contain language suggestive of PTSD would let clinicians target a more thorough evaluation at otherwise-overlooked patients.

The methodological challenge is that the obvious approach — train on ICD-coded patients vs. non-coded patients — is wrong by assumption. If undercoding is the central feature of the problem, the "negative" group is contaminated with real positives, and any model trained against that gold standard is measuring agreement with a flawed reference standard rather than actual clinical validity.

### Approach

A Clinical Longformer (`yikuan8/Clinical-Longformer`, 4,096 tokens) is fine-tuned with the Kiryo et al. (2017) non-negative positive-unlabeled (PU) risk estimator. PU learning treats unlabeled patients as a mixture of true negatives and hidden positives, weighted by an estimated class prior π_p, instead of assuming every uncoded patient is PTSD-negative. PULSNAR (Kumar & Lambert 2024) propensity-weighted nnPU is run alongside as a SAR-aware sensitivity model.

Five models are trained head-to-head — Clinical Longformer (primary), BioClinicalBERT (truncated and chunk-and-pool variants), TF-IDF + logistic regression, structured-features-only logistic regression, and a zero-training DSM-5/PCL-5 keyword baseline — to isolate the contribution of the long-context transformer architecture, the PU loss, and the narrative content itself.

### Cohort

Three subject groups are assembled from MIMIC-IV via streaming I/O (no full source CSV is ever loaded into RAM):

| Group | Definition | N | Role |
|---|---|---:|---|
| ICD-coded PTSD+ | ICD-10 `F431` or ICD-9 `30981` at any admission | 5,711 | Training positives |
| Pharmacological proxy | Prazosin × SSRI/SNRI overlap ≤ 180 days; excludes I10/N40/I7300/S06 and Group 1 | 163 | External validation only — never in training |
| Matched unlabeled pool | All remaining subjects, 3:1 matched on age decade × sex | 17,133 | PU pool |

Of Group 1, 2,492 (43.6%) have ≥ 1 admission **before** their first PTSD code — these become the primary pre-diagnosis training subsample. The remaining 3,219 use masking-based section-filtered index-admission notes as fallback. A separate psych-control cohort (5,711 patients, 3,148 with notes) is built for a specificity sanity check (PTSD+ vs. age/sex-matched MDD/anxiety patients).

### Label-leakage prevention

A tiered defense is applied so the model learns clinical signal rather than label surface form:

1. **Section filtering.** Only narrative low-leakage sections are kept: HPI, Social History, PMH, Brief Hospital Course. Diagnostic and plan sections are excluded.
2. **Pre-diagnosis notes** as primary signal — for multi-admission PTSD+ patients, training text comes from admissions strictly before the first PTSD code.
3. **Universal PTSD-string masking.** A regex (`ptsd`, `posttraumatic`, `post-traumatic stress`, `f43.1`, `309.81`) replaces matches with `[PTSD_MASKED]` across **all** PTSD+ notes (pre-dx + fallback). An audit found 8.6% of pre-dx notes still contained explicit PTSD strings carried forward from outside records — confirming the original "pre-dx is automatically clean" assumption was wrong for ~1 in 12 patients.
4. **Two ablations** quantify residual leakage: post-hoc string masking (Ablation 1) and full PMH-section removal (Ablation 2).

### Evaluation strategy

- **Primary metric:** AUPRC (with AUROC alongside). All raw metrics labeled as *PU lower bounds* with Ramola et al. (2019) corrections reported alongside.
- **Operating threshold derived on validation** at sensitivity ≥ 0.85, then frozen before any test-set metrics — eliminates selection-on-test bias.
- **Class prior π_p swept** over {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25} in 7 parallel SLURM training jobs; winner selected by proxy Mann-Whitney AUC, the only PU-uncontaminated criterion.
- **Pharmacological proxy external validation** — Mann-Whitney U of model scores between proxy patients and 500 random unlabeled controls. Proxy patients are identified by an entirely independent (medication-based) criterion the model cannot see.
- **Specificity check** — separate Longformer trained PTSD+ vs. age/sex-matched MDD/anxiety controls with standard cross-entropy, isolating PTSD-specific signal from generic psychiatric language.
- **Calibration** — raw + Platt + Elkan-Noto-corrected probabilities; ECE on 10 equal-frequency bins with Wilson 95% CIs.
- **Decision Curve Analysis** (Vickers) at 2% and 5% deployment prevalence.
- **Clinical utility** — LR+, LR−, DOR, alert rate, workup reduction, NNS at deployment prevalences {1, 2, 5, 10, 20%}.
- **Fairness** — calibration-in-the-large + equal-opportunity differences + bootstrap 95% CI on AUPRC (only reported when CI width < 0.15).
- **Explainability** — Integrated Gradients (Captum) at full 4,096-token context, aggregated by section and by whole word (BPE subwords merged with summed attribution). Attention weights deliberately not used (Jain & Wallace 2019).
- **McNemar's test** with continuity correction for paired model comparison.
- **Temporal generalization** — pre-2015 train / 2017–2019 test split using each patient's `anchor_year_group` (raw `admittime` is unusable due to per-patient random date shifts of ~100–200 years).

### Key results

All numbers on the held-out test set (n = 1,551 patients, 660 PTSD+) under val-derived thresholds.

**Discrimination:**

| Model | AUPRC | AUROC | Sens | Spec | F1 | NNS @ 2% prev |
|---|---:|---:|---:|---:|---:|---:|
| **Clinical Longformer (π_p=0.25)** | **0.8939** | **0.9002** | 0.852 | 0.782 | **0.794** | **13.5** |
| Clinical Longformer (PULSNAR) | 0.8848 | 0.8904 | 0.846 | 0.745 | 0.772 | 15.8 |
| BioClinicalBERT (chunk-pool) | 0.8775 | 0.8853 | 0.902 | 0.626 | 0.749 | 21.0 |
| BioClinicalBERT (truncated) | 0.8576 | 0.8656 | 0.820 | 0.728 | 0.750 | 16.5 |
| TF-IDF + LogReg | 0.8380 | 0.8567 | 0.817 | 0.721 | 0.745 | 17.0 |
| Structured + LogReg | 0.6833 | 0.7310 | 0.909 | 0.207 | 0.610 | 36.7 |
| Keyword (DSM-5/PCL-5) | 0.5373 | 0.6086 | 1.000 | 0.000 | 0.597 | 47.0 |

Longformer beats every comparator on McNemar's test (p < 1e-5). Chunk-and-pool BERT closes most of the gap, indicating that long-context inference (not architecture per se) drives most of Longformer's lift; the residual ~0.016 AUPRC gain comes from long-range pre-training. Structured-only is well below text-based models — narrative content carries the bulk of predictive signal. Keyword is essentially random.

**Pharmacological proxy validation (the headline non-circular result):** Proxy patients (n = 102) score median 0.383 vs. 0.059 for 500 random unlabeled controls — Mann-Whitney AUC = 0.799, p = 8.3e-22. **58% of proxy patients clear the screening threshold vs. 15% of unlabeled.** The model — never shown a single proxy patient during training — assigns ~6× the median probability to patients whose pharmacotherapy is consistent with PTSD treatment.

**Ablations:** Post-hoc PTSD-string masking costs only −0.008 AUPRC (model is *not* exploiting literal PTSD strings). Removing the entire PMH section costs −0.061 AUPRC, but the model still scores 0.82 — comfortably above TF-IDF — confirming HPI, Social History, and Brief Hospital Course independently encode enough PTSD-associated language.

**Specificity check:** A separate Longformer trained PTSD+ vs. matched MDD/anxiety controls with standard cross-entropy reaches AUPRC 0.91 — PTSD-specific signal is recoverable above-and-beyond generic "psychiatric admission" language.

**Calibration:** Raw ECE 0.064 (best), Platt 0.074, Elkan-Noto 0.080. Elkan-Noto c = 0.78 implies ~22% undercoding rate, consistent with PCL-5 inpatient prevalence findings.

**Subgroup performance (the central deployment caveat):** AUPRC 0.92 (women) vs. 0.83 (men); 0.94 (20s) vs. 0.85 ("Other" age); 0.92 (emergency) vs. 0.83 (elective). Race-binary EO difference is minimal (0.024); sex EO 0.116; age EO 0.209. The model performs best on the demographics most likely to be coded today, weakest where undercoding is most prevalent — exactly the residual selection bias PU learning reduces but does not eliminate.

**Decision Curve Analysis:** Positive net benefit over treat-none across thresholds 0.01–0.30 at 5% prevalence; at 2% prevalence, the model is competitive with treat-all only at moderate thresholds — the cost of missing a case is high enough at very low prevalence that flagging everyone is competitive in the very-low-threshold band.

**Integrated Gradients:** HPI dominates (36.4% of total attribution), then Brief Hospital Course (35.9%), PMH (26.3%, highest per-token density), Social History (0.5%). Top attributed words are clinically appropriate trauma/psychiatric/substance vocabulary (`bipolar`, `personality`, `schizoaffective`, `assault`, `abuse`, `heroin`, `psychiatric`); no label-leakage tokens appear. The PULSNAR sensitivity model attributes more weight to HPI (43.2%) and less to PMH (22.4%), consistent with SAR-aware training discounting the comorbidity-coding signal.

**Compute frontier:** Longformer 80.4 ms/patient on L40S (5.7 GPU-h training); BERT chunk-pool 22.7 ms/patient (~0.22 GPU-h); keyword 0.34 ms/patient on 16 CPUs (zero training). Architecture (4,096 vs. 512 attention) — not GPU generation — drives the gap.

### Limitations

- **Section filtering is not perfect leakage prevention.** Even on a pre-diagnosis admission, PMH may carry forward "history of PTSD" from outside records. The full-PMH-removal ablation bounds residual leakage at 6 AUPRC points.
- **PU learning reduces but does not eliminate selection bias.** The Kiryo formulation removes the assumption that unlabeled patients are confirmed negatives but does not correct for non-random ICD coding — exactly why the older-patient and male-patient AUPRC gaps persist.
- **The pre-diagnosis training subsample is not representative.** Only 43.6% of PTSD+ patients had pre-diagnosis admissions; the remaining 56.4% use index-admission notes with masking applied and carry more residual leakage risk.
- **Proxy validation set is small (n = 102) and has an estimated 15–20% FPR** from off-label prazosin use not covered by the exclusion ICD set.
- **Single-site data.** MIMIC-IV is one academic medical center in Boston; transfer to community hospitals, rural settings, or VA facilities is unknown.
- **TF-IDF baseline has a label leak** (`ptsd_masked` coefficient +37.13 — TF-IDF tokenization strips brackets, leaving the masked token itself as a strong feature). Reported TF-IDF AUPRC of 0.838 is therefore inflated. Transformers do not appear to suffer the analogous leak (the post-hoc string ablation costs only −0.008 AUPRC).

### What this tool is and is not

**It is** a screening prompt — a way to point inpatient clinicians at patients whose narrative notes contain language patterns associated with PTSD. The output is intended to inform a more thorough psychiatric evaluation, not to make a diagnosis. At 2% deployment prevalence and the chosen operating threshold, the tool would surface ~14 patients per true case for clinician review.

**It is not** a diagnostic tool, a replacement for structured PTSD screening (PCL-5, CAPS-5), or a basis for ICD coding.

### Future directions

- External validation at a non-MIMIC site (a VA medical center, where PTSD prevalence is much higher and the tool would arguably be most valuable).
- Re-evaluation under a true reference standard — a small prospective cohort screened with PCL-5 or CAPS-5 would let absolute model performance be measured against a defensible gold standard rather than against contaminated ICD labels.
- Bias mitigation for demographic subgroups — explicit subgroup-aware loss reweighting or label-noise modelling could narrow the inherited age/sex gap.
- Re-trained TF-IDF baseline with stricter masking to determine its honest leak-free performance.
- Calibration / fairness / attribution computed on chunk-pool BERT to close the methodology comparison.
- Concatenating structured features (excluding the artifactual `n_prior_admissions`) into the Longformer head for complementary signal where text features are weakest.

---

## 2. Programs

### Cohort construction & data extraction (`scripts/01_cohort/`)

| Script | Description |
|---|---|
| `ewang163_ptsd_table1.py` | Builds the three study cohorts from MIMIC-IV ICD codes and prescriptions, computes all Table 1 demographic and clinical characteristics, and writes results to CSV + human-readable summary. Single source of truth for cohort definitions. |
| `ewang163_ptsd_cohort_sets.py` | Reconstructs the three cohort subject ID sets (PTSD+, pharmacological proxy, matched unlabeled) via streaming reads of MIMIC-IV source files; verifies counts against the canonical 5,711 / 163 / 17,133. |
| `ewang163_ptsd_admissions_extract.py` | Extracts admission rows for all cohort groups, attaches patient demographics, computes age at admission, and tags the index admission per patient (PTSD+ index = first PTSD-coded admission; proxy / unlabeled index = first MIMIC-IV admission). |
| `ewang163_ptsd_notes_extract.py` | Streams the 3.3 GB MIMIC-IV `discharge.csv`, regex-parses section headers, and writes section-filtered notes (HPI, Social History, PMH, Brief Hospital Course) for all three cohort groups. Resolves PTSD+ patients to pre-diagnosis vs. fallback. |

### Corpus assembly & splits (`scripts/02_corpus/`)

| Script | Description |
|---|---|
| `ewang163_ptsd_corpus_build.py` | Assembles the training corpus from extracted notes, audits pre-diagnosis PTSD-string leakage rate, applies universal `[PTSD_MASKED]` substitution to all PTSD+ notes, and holds the proxy group out as a separate validation file. |
| `ewang163_ptsd_splits.py` | Creates patient-level 80/10/10 train/validation/test splits stratified by PTSD label. `--temporal` flag uses each patient's `anchor_year_group` (un-shifted) for a pre-2015 / 2017–2019 generalization test. |

### Model training (`scripts/03_training/`)

| Script | Description |
|---|---|
| `ewang163_ptsd_train_longformer.py` (+ `.sbatch`) | Fine-tunes Clinical Longformer (4,096 tokens) with Kiryo nnPU loss; AdamW lr 2e-5, effective batch 32, 5 epochs, mixed precision, gradient checkpointing. Best-epoch checkpointing on val AUPRC. Accepts `--pi_p` and `--output_suffix` for the class-prior sweep. |
| `ewang163_ptsd_train_bioclinbert.py` (+ `.sh`) | Fine-tunes BioClinicalBERT (512 tokens) with the same nnPU loss; runs McNemar's test against Longformer on the test set. |
| `ewang163_ptsd_train_tfidf.py` | TF-IDF (50k features, word 1+2-grams, sublinear TF) + logistic regression baseline with `class_weight='balanced'`; tunes regularization C on validation AUPRC. |
| `ewang163_ptsd_train_structured.py` (+ `.sh`) | Logistic regression baseline on 20 structured features (demographics, prior comorbidities, prior medications); tunes C on val AUPRC and reports feature coefficients. |
| `ewang163_ptsd_train_keyword.py` | Zero-training keyword/phrase-lookup baseline. Scores notes using a hand-curated DSM-5/PCL-5 phrase lexicon (62 weighted regex patterns across criteria A–E + treatment signals). Two scoring variants compared (raw weighted count, TF-normalized); best variant selected on validation AUPRC. |
| `ewang163_ptsd_train_pulsnar.py` | PULSNAR (SAR-aware) variant. Estimates propensity scores P(coded \| features) via logistic regression on prior-admission demographics/comorbidities/medications, reweights the nnPU positive loss term by 1/e(x), and uses PULSNAR for class-prior estimation. |
| `ewang163_ptsd_pip_sweep.sh` | Submits 7 parallel SLURM GPU jobs sweeping π_p ∈ {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25}. Each job trains a full Longformer with a different class prior. |
| `ewang163_ptsd_pip_sweep_eval.py` | Aggregates the π_p sweep results: runs proxy validation on each checkpoint, picks the best π_p by proxy-vs-unlabeled Mann-Whitney AUC, and produces a sweep plot. |
| `ewang163_ptsd_specificity.py` (+ `.sh`) | Specificity check: retrains Clinical Longformer on PTSD+ vs. age/sex-matched MDD/anxiety controls using standard cross-entropy to assess whether the model learned PTSD-specific signal vs. generic psychiatric language. |

### Evaluation & analysis (`scripts/04_evaluation/`)

| Script | Description |
|---|---|
| `ewang163_ptsd_evaluate.py` (+ `.sh`) | Master evaluation. Runs all five models on the held-out test set with **validation-derived thresholds** (sensitivity ≥ 0.85). Reports AUPRC, AUROC, sens/spec/precision/F1, plus clinical utility (LR+, LR−, DOR, alert rate, workup reduction, NNS/NNE at deployment prevalences), Ramola PU corrections, McNemar's tests, and subgroup analysis. Implements chunk-and-pool inference for BioClinicalBERT (overlapping 512-token windows, stride 256, max-pool). |
| `ewang163_ptsd_calibration.py` (+ `.sh`) | Fits Platt scaling on validation, then applies the Elkan-Noto correction (divides by c = mean(raw model prob on val positives)) so probabilities approximate P(PTSD=1) rather than P(coded=1). Reports ECE for raw, Platt-scaled, and Elkan-Noto variants on 10 equal-frequency bins with Wilson 95% CIs. |
| `ewang163_ptsd_decision_curves.py` (+ `.sh`) | Decision Curve Analysis at 2% and 5% deployment prevalences. Recalibrates probabilities to deployment prevalence via Bayes' rule and plots net-benefit curves over thresholds 0.01–0.40. |
| `ewang163_ptsd_proxy_validation.py` (+ `.sh`) | External validation on the held-out pharmacological proxy group; compares Longformer scores between proxy patients and a 500-patient random sample drawn from the training-pool unlabeled patients using Mann-Whitney U and AUC. |
| `ewang163_ptsd_ablations.py` (+ `.sh`) | Two ablation experiments — Ablation 1 (post-hoc PTSD-string masking) and Ablation 2 (PMH section removal at re-extraction time) — to verify the model is not exploiting label-leaking text. |
| `ewang163_ptsd_attribution.py` (+ `.sh`) | First-pass Integrated Gradients via Captum's `LayerIntegratedGradients` (largely deprecated due to Longformer global-attention conflicts). |
| `ewang163_ptsd_attribution_v2.py` (+ `.sh`) | Updated IG using `IntegratedGradients` on the embedding tensor via a custom wrapper that handles Longformer's hybrid local/global attention. Full 4,096-token context, n_steps=20, word-level aggregation merges contiguous BPE subwords with summed attributions. Top words and per-section attribution. |
| `ewang163_ptsd_error_analysis.py` (+ `.sh`) | Samples 25 false positives and 25 false negatives at the operating threshold, writes annotated notes to text files, computes aggregate statistics (mean predicted probability, note length, top TF-IDF terms, demographic skew, trauma-term scan). |
| `ewang163_ptsd_fairness.py` | Calibration-in-the-large per subgroup, equal opportunity difference, and bootstrap 95% CI on AUPRC (1,000 resamples; only reported when CI width < 0.15). White vs. non-White primary contrast plus sex / age / race / emergency. |
| `ewang163_ptsd_temporal_eval.py` | Compares temporal-trained model on temporal test, random-trained model on temporal test, and random-trained model on random test — measures generalization across the ICD-9→ICD-10 transition and post-DSM-5 reclassification. |
| `ewang163_ptsd_pulsnar_reeval.py` | Re-runs calibration / clinical utility / fairness on the PULSNAR-trained checkpoint so the manuscript can present pi_p=0.25 and PULSNAR results side-by-side (dual-winner reporting). |
| `ewang163_unified_inference_bench.py` (+ `.sh`) | Apples-to-apples inference timing for all five models in a single L40S GPU allocation. |
| `ewang163_cpu_inference_bench.py` | Re-measures inference for CPU-only baselines (keyword, structured) under realistic 16-CPU allocation, with joblib-parallel keyword scoring. |

### Shared utilities (`scripts/common/`)

| Script | Description |
|---|---|
| `ewang163_bench_utils.py` | `BenchmarkLogger` context manager. Captures wall-clock time, CPU time, peak memory, and GPU-hours per pipeline stage; appends rows to `results/metrics/ewang163_runtime_benchmarks.csv`. Used by all training and evaluation scripts. |

### Top-level utilities

| Script | Description |
|---|---|
| `ewang163_runtime_audit.py` | Captures missing keyword + structured inference timings as benchmark rows. |
| `ewang163_bench_baselines.py` | Captures TF-IDF training time as benchmark rows (vectorize + C sweep). |

---

## 3. Repository Layout

```
ewang163/
├── CLAUDE.md                                  Authoritative project specification
├── README.md                                  This file
├── ewang163_project_writeup.md                Full project write-up
├── ewang163_model_selection_memo.md           Final model selection memo
├── ewang163_methodology_fixes_results.md      Methodology audit execution results
├── ewang163_model_comparison.md               Multi-model comparison (discrimination + runtime + explainability)
├── methodology_fix_plans.md                   Pre-execution methodology fix plans (literature-grounded)
│
├── scripts/                                   Pipeline code, ordered by stage
│   ├── common/ewang163_bench_utils.py
│   ├── 01_cohort/                             Cohort construction from MIMIC-IV
│   │   ├── ewang163_ptsd_table1.py
│   │   ├── ewang163_ptsd_cohort_sets.py
│   │   ├── ewang163_ptsd_admissions_extract.py
│   │   └── ewang163_ptsd_notes_extract.py
│   ├── 02_corpus/                             Corpus assembly + patient-level splits
│   │   ├── ewang163_ptsd_corpus_build.py
│   │   └── ewang163_ptsd_splits.py
│   ├── 03_training/                           Model training
│   │   ├── ewang163_ptsd_train_longformer.py        (+ .sbatch)
│   │   ├── ewang163_ptsd_train_bioclinbert.py       (+ .sh)
│   │   ├── ewang163_ptsd_train_tfidf.py
│   │   ├── ewang163_ptsd_train_structured.py        (+ .sh)
│   │   ├── ewang163_ptsd_train_keyword.py
│   │   ├── ewang163_ptsd_train_pulsnar.py
│   │   ├── ewang163_ptsd_pip_sweep.sh
│   │   ├── ewang163_ptsd_pip_sweep_eval.py
│   │   └── ewang163_ptsd_specificity.py             (+ .sh)
│   └── 04_evaluation/                         Evaluation, calibration, ablations, attribution, etc.
│       ├── ewang163_ptsd_evaluate.py                (+ .sh)
│       ├── ewang163_ptsd_calibration.py             (+ .sh)
│       ├── ewang163_ptsd_decision_curves.py         (+ .sh)
│       ├── ewang163_ptsd_proxy_validation.py        (+ .sh)
│       ├── ewang163_ptsd_ablations.py               (+ .sh)
│       ├── ewang163_ptsd_attribution.py             (+ .sh)
│       ├── ewang163_ptsd_attribution_v2.py          (+ .sh)
│       ├── ewang163_ptsd_error_analysis.py          (+ .sh)
│       ├── ewang163_ptsd_fairness.py
│       ├── ewang163_ptsd_temporal_eval.py
│       ├── ewang163_ptsd_pulsnar_reeval.py
│       ├── ewang163_unified_inference_bench.py      (+ .sh)
│       └── ewang163_cpu_inference_bench.py
│
├── data/                                      [git-ignored — local only]
│   ├── cohort/                                Subject ID lists, admission extracts
│   ├── notes/                                 Section-filtered note parquets
│   └── splits/                                Patient-level train/val/test splits (random + temporal)
│
├── models/                                    Trained model artefacts
│   ├── ewang163_longformer_best/              → symlinked to ewang163_longformer_best_pip025/
│   ├── ewang163_longformer_best_pip{005..025}/   π_p sweep checkpoints
│   ├── ewang163_longformer_best_retrain_empirical/  π_p = 0.398 retrain
│   ├── ewang163_longformer_pulsnar/           PULSNAR (α = 0.196) variant
│   ├── ewang163_longformer_best_temporal/     Pre-2015 train (not recommended for deployment)
│   ├── ewang163_bioclinbert_best/             Comparison BERT model
│   ├── ewang163_specificity_longformer_best/  Specificity check vs. psych controls
│   ├── ewang163_tfidf_vectorizer.pkl + ewang163_tfidf_logreg.pkl
│   ├── ewang163_structured_logreg.pkl + ewang163_structured_features.json
│   ├── ewang163_keyword_weights.json          DSM-5/PCL-5 phrase lexicon (62 patterns)
│   └── ewang163_platt_calibrator.pkl          Post-hoc probability calibrator
│
├── results/
│   ├── table1/                                Cohort characterization (CSV + plain text)
│   ├── predictions/                           Per-patient predicted probabilities (val + test)
│   ├── metrics/                               JSON + CSV evaluation outputs (eval, ablations, calibration, DCA, proxy, training logs, runtime benchmarks)
│   ├── figures/                               PNG plots (calibration, DCA, proxy histogram, π_p sweep)
│   ├── attribution/                           IG section + token + word attribution
│   └── error_analysis/                        Sampled FP/FN summary stats
│
├── logs/                                      SLURM .out files from every job submitted
├── PULSNAR/                                   Vendored Kumar & Lambert 2024 PULSNAR library [git-ignored]
└── ptsd_env/                                  Python virtual environment
```

---

## 4. Installation

### Requirements

- Python 3.13+
- Packages: `pandas`, `numpy`, `transformers`, `torch`, `scikit-learn`, `captum`, `scipy`, `matplotlib`, `joblib`, `xgboost` (for PULSNAR), `pyarrow` (parquet)
- For PULSNAR class-prior estimation: see `PULSNAR/requirements.txt` (xgboost, catboost, threadpoolctl). The `rpy2`/R dependency is lazy and only triggered by non-default bandwidth methods.

### Setup (Oscar HPC)

```bash
# Activate the existing virtual environment
source /oscar/data/class/biol1595_2595/students/ewang163/ptsd_env/bin/activate

# Install additional packages if needed
pip install transformers torch scikit-learn captum xgboost
```

### Paths

| Variable | Path |
|---|---|
| `STUDENT_DIR` | `/oscar/data/class/biol1595_2595/students/ewang163/` |
| `MIMIC_ROOT` | `/oscar/data/shared/ursa/mimic-iv` (read-only) |
| `HOSP` | `/oscar/data/shared/ursa/mimic-iv/hosp/3.1/` |
| `NOTES` | `/oscar/data/shared/ursa/mimic-iv/note/2.2/` |

---

## 5. Usage

**All scripts must be submitted via SLURM, never run interactively on the login node.** Login-node usage is monitored — exceeding limits disrupts the user account.

### CPU jobs (data processing, baselines)

```bash
sbatch --partition=batch --mem=16G --time=2:00:00 \
       --output=ewang163_%j.out --wrap="python <script>.py"
```

### GPU jobs (transformer training and inference)

Most GPU scripts ship a paired `.sh` / `.sbatch` file with the SLURM header embedded:

```bash
sbatch scripts/03_training/ewang163_ptsd_train_longformer.sbatch
sbatch scripts/04_evaluation/ewang163_ptsd_evaluate.sh
```

Generic GPU template:

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o ewang163_%j.out
```

### Full pipeline order (re-run from raw MIMIC-IV)

Each stage saves its output to `STUDENT_DIR` before the next stage begins, so any stage can be re-run independently.

1. `scripts/01_cohort/ewang163_ptsd_table1.py` → `results/table1/`
2. `scripts/01_cohort/ewang163_ptsd_cohort_sets.py` → `data/cohort/*_subjects.txt`
3. `scripts/01_cohort/ewang163_ptsd_admissions_extract.py` → `data/cohort/ewang163_ptsd_adm_extract.parquet`
4. `scripts/01_cohort/ewang163_ptsd_notes_extract.py` → `data/notes/ewang163_ptsd_notes_raw.parquet`
5. `scripts/02_corpus/ewang163_ptsd_corpus_build.py` → `data/notes/ewang163_ptsd_corpus.parquet`
6. `scripts/02_corpus/ewang163_ptsd_splits.py` (and `--temporal`) → `data/splits/`
7. `scripts/03_training/ewang163_ptsd_train_keyword.py` (CPU, seconds)
8. `scripts/03_training/ewang163_ptsd_train_tfidf.py` (CPU, ~16 s)
9. `scripts/03_training/ewang163_ptsd_train_structured.py` (CPU, ~70 s)
10. `scripts/03_training/ewang163_ptsd_train_bioclinbert.py` (GPU, ~14 min)
11. `scripts/03_training/ewang163_ptsd_pip_sweep.sh` (7 parallel GPU jobs, ~5.7 GPU-h each)
12. `scripts/03_training/ewang163_ptsd_pip_sweep_eval.py` (selects π_p winner)
13. `scripts/03_training/ewang163_ptsd_train_pulsnar.py` (GPU, ~3.5 GPU-h)
14. `scripts/03_training/ewang163_ptsd_specificity.py` (GPU, ~7.5 GPU-h, secondary)
15. `scripts/04_evaluation/ewang163_ptsd_evaluate.py` (val-derived thresholds frozen here)
16. `scripts/04_evaluation/ewang163_ptsd_calibration.py`
17. `scripts/04_evaluation/ewang163_ptsd_decision_curves.py`
18. `scripts/04_evaluation/ewang163_ptsd_proxy_validation.py`
19. `scripts/04_evaluation/ewang163_ptsd_ablations.py`
20. `scripts/04_evaluation/ewang163_ptsd_attribution_v2.py`
21. `scripts/04_evaluation/ewang163_ptsd_error_analysis.py`
22. `scripts/04_evaluation/ewang163_ptsd_fairness.py`
23. `scripts/04_evaluation/ewang163_ptsd_temporal_eval.py`
24. `scripts/04_evaluation/ewang163_ptsd_pulsnar_reeval.py`
25. `scripts/04_evaluation/ewang163_unified_inference_bench.py`

### Reproducibility

Random seeds are pinned to 42 throughout (matching, sampling, splits, model training, bootstrap resampling). All design decisions, ICD code lists, drug lists, MIMIC-IV data quirks, and bug fixes are documented in `CLAUDE.md`. SLURM job logs for every run are preserved under `logs/`. The full project write-up with detailed results discussion is in `ewang163_project_writeup.md`.
