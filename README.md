# PTSD Underdiagnosis Detection in MIMIC-IV Discharge Notes

**Author:** Eric Wang (ewang163), Brown University, AIH 2025 (Spring 2026)
**Source data:** MIMIC-IV v3.1 (BIDMC, 2008–2019)
**Final deployment model:** Clinical Longformer fine-tuned with PULSNAR
(Positive Unlabeled Learning Selected Not At Random; Kumar & Lambert 2024)
propensity-weighted nnPU loss on a section-filtered, label-masked corpus.

---

## 1. Project Overview

### Clinical motivation

PTSD is systematically undercoded in inpatient settings. Inpatient teams aren't
trained to elicit DSM-5 PTSD criteria, and PTSD has no laboratory or imaging
marker — it has to be asked about, with knowledge of the patient's psychological
and social history. Estimated true prevalence in trauma-exposed inpatients is
≥ 20%, but the ICD coding rate in MIMIC-IV is closer to 1%. Patients who are
missed lose access to targeted treatment (trauma-focused psychotherapy, prazosin
for nightmares, SSRI/SNRI), generate avoidable readmissions, and lose continuity
of trauma history. A screening tool that flags patients whose discharge notes
contain language suggestive of PTSD would let clinicians target a more thorough
evaluation at otherwise-overlooked patients.

The methodological challenge is that the obvious approach — train on ICD-coded
patients vs. non-coded patients — is wrong by assumption. If undercoding is the
central feature of the problem, the "negative" group is contaminated with real
positives, and any model trained against that gold standard is measuring agreement
with a flawed reference rather than actual clinical validity.

### Approach

A Clinical Longformer (`yikuan8/Clinical-Longformer`, 4,096 tokens) is fine-tuned
with the **PULSNAR** propensity-weighted non-negative PU risk estimator
(Kumar & Lambert 2024, building on Kiryo et al. 2017). PU learning treats
unlabeled patients as a mixture of true negatives and hidden positives;
PULSNAR additionally relaxes the "selected completely at random" (SCAR)
assumption that standard nnPU requires — important here because PTSD coding
is biased by demographics and prior psychiatric contact, not random.

Four models are trained and compared head-to-head on identical patient-level
splits to isolate the contribution of long-context inference, the PU loss, and
the narrative content itself:

  1. **Clinical Longformer (PULSNAR)** — primary deployment model
  2. **BioClinicalBERT** — comparison transformer, two inference modes:
     truncated 512 tokens *and* chunk-and-pool (overlapping 512-token windows,
     stride 256, max-pool)
  3. **Structured features + logistic regression** — 20 hand-engineered
     features (demographics + prior comorbidities + prior medications)
  4. **Keyword (DSM-5/PCL-5)** — 62 weighted regex patterns; zero training

### Cohort

Three subject groups are assembled from MIMIC-IV via streaming I/O (no full
source CSV is ever loaded into RAM):

| Group | Definition | N | Role |
|---|---|---:|---|
| ICD-coded PTSD+ | ICD-10 `F431` or ICD-9 `30981` at any admission | 5,711 | Training positives |
| Pharmacological proxy | Prazosin × SSRI/SNRI overlap ≤ 180 days; excludes I10/N40/I7300/S06 and Group 1 | 163 | External validation only — never in training |
| Matched unlabeled pool | All remaining subjects, 3:1 matched on age decade × sex | 17,133 | PU pool |

Of Group 1, 2,492 (43.6%) have ≥ 1 admission **before** their first PTSD code —
these become the primary pre-diagnosis training subsample. The remaining 3,219
use masking-based section-filtered index-admission notes as fallback. A separate
psych-control cohort (5,711 patients, 3,148 with notes) is built for a specificity
sanity check (PTSD+ vs. age/sex-matched MDD/anxiety patients).

### Label-leakage prevention

A tiered defense is applied so the model learns clinical signal rather than
label surface form:

1. **Section filtering.** Only narrative low-leakage sections are kept: HPI,
   Social History, PMH, Brief Hospital Course. Diagnostic and plan sections
   are excluded.
2. **Pre-diagnosis notes** as primary signal — for multi-admission PTSD+
   patients, training text comes from admissions strictly before the first
   PTSD code.
3. **Universal PTSD-string masking.** A regex (`ptsd`, `posttraumatic`,
   `post-traumatic stress`, `f43.1`, `309.81`) replaces matches with
   `[PTSD_MASKED]` across **all** PTSD+ notes (pre-dx + fallback). An audit
   found 8.6% of pre-dx notes still contained explicit PTSD strings carried
   forward from outside records — confirming the original "pre-dx is
   automatically clean" assumption was wrong for ~1 in 12 patients.
4. **Two ablations** quantify residual leakage: post-hoc string masking
   (Ablation 1) and full PMH-section removal (Ablation 2).

### Evaluation strategy

- **Primary metric:** AUPRC (with AUROC alongside). All raw metrics are
  labeled as *PU lower bounds* with Ramola et al. (2019) corrections reported
  alongside.
- **Operating threshold derived on validation** at sensitivity ≥ 0.85, then
  frozen before any test-set metrics — eliminates selection-on-test bias.
- **Pharmacological proxy external validation.** Mann-Whitney U of model
  scores between proxy patients and 500 random unlabeled controls. Proxy
  patients are identified by an entirely independent (medication-based)
  criterion the model cannot see — the strongest non-circular validity signal.
- **Specificity check.** Separate Longformer trained PTSD+ vs. age/sex-matched
  MDD/anxiety controls with standard cross-entropy, isolating PTSD-specific
  signal from generic psychiatric language.
- **Calibration.** Raw + Platt + Elkan-Noto-corrected probabilities; ECE on
  10 equal-frequency bins with Wilson 95% CIs.
- **Decision Curve Analysis** (Vickers) at 2% and 5% deployment prevalence.
- **Clinical utility.** LR+, LR−, DOR, alert rate, workup reduction, NNS at
  deployment prevalences {1, 2, 5, 10, 20%}.
- **Fairness.** Calibration-in-the-large + equal-opportunity differences +
  bootstrap 95% CI on AUPRC (only reported when CI width < 0.15).
- **Explainability.** Integrated Gradients (Captum) at full 4,096-token context
  for Longformer, and at both truncated-512 and per-window chunk-pool slices
  for BioClinicalBERT, aggregated by section and by whole word (BPE subwords
  merged with summed attribution). Attention weights are deliberately not used
  (Jain & Wallace 2019).
- **McNemar's test** with continuity correction for paired model comparison —
  pairwise across all 5 deployed models.
- **Cross-model agreement matrix.** Pairwise McNemar p, Cohen's kappa on
  binary predictions, Pearson correlation on probabilities, top-quintile
  rank overlap, plus a max-pool / mean-pool ensemble probe to test whether
  any combination beats the best individual model.
- **Temporal generalization.** Pre-2015 train / 2017–2019 test split using
  each patient's `anchor_year_group` (raw `admittime` is unusable due to
  per-patient random date shifts of ~100–200 years).

The full set of analyses (calibration, DCA, fairness, subgroup AUPRC, error
analysis, ablations, IG attribution, proxy validation) is computed for **all**
text-based models — Clinical Longformer (PULSNAR), BioClinicalBERT
(truncated), and BioClinicalBERT (chunk-pool) — so the multi-model comparison
is symmetric.

### Key results

All numbers on the held-out test set (n = 1,551 patients, 660 PTSD+) under
val-derived thresholds.

**Discrimination** (see `ewang163_model_comparison.md` for full numbers including
NNS and DCA):

| Model | AUPRC | AUROC | Sens | Spec | F1 |
|---|---:|---:|---:|---:|---:|
| **Clinical Longformer (PULSNAR, primary)** | **0.8848** | **0.8904** | 0.846 | 0.745 | **0.772** |
| BioClinicalBERT (chunk-pool 512×256)        | 0.8775 | 0.8853 | 0.902 | 0.626 | 0.749 |
| BioClinicalBERT (truncated 512)             | 0.8576 | 0.8656 | 0.820 | 0.728 | 0.750 |
| Structured + LogReg                         | 0.6833 | 0.7310 | 0.909 | 0.207 | 0.610 |
| Keyword (DSM-5/PCL-5)                       | 0.5373 | 0.6086 | 1.000 | 0.000 | 0.597 |

The Longformer wins discrimination on the held-out set. BioClinicalBERT
chunk-pool closes most of the gap (Δ AUPRC = 0.007), indicating that
long-context inference — not architecture per se — drives most of Longformer's
lift; the residual ~0.007 AUPRC gain is attributable to long-range pre-training.
Structured-only is well below text-based models — narrative content carries
the bulk of predictive signal. Keyword is essentially random.

**Pharmacological proxy validation (the headline non-circular result).** All
three text models separate proxy patients (n = 102) from 500 random unlabeled
controls at p ≪ 1e-10:

| Model | Proxy MW AUC | MW p |
|---|---:|---:|
| Clinical Longformer (PULSNAR) | **0.7701** | 3.8e-18 |
| BioClinicalBERT (truncated)   | 0.7442 | 3.7e-15 |
| BioClinicalBERT (chunk-pool)  | 0.7333 | 5.3e-14 |

The PULSNAR Longformer gives the strongest proxy separation, but every text
model assigns substantially higher scores to medication-pattern-positive
patients than to demographically-matched unlabeled patients — non-circular
evidence that they recover a real PTSD-associated narrative signal rather
than just re-deriving the ICD coding rule.

**Ablations** (PULSNAR Longformer; see `ewang163_model_comparison.md` for full
per-model deltas including BERT). Post-hoc PTSD-string masking costs less than
0.01 AUPRC across all models — they are *not* exploiting literal PTSD strings.
Removing the entire PMH section costs ~0.06 AUPRC for Longformer and ~0.06 for
BERT chunk-pool — comparable PMH dependence — but every model still scores
above the structured baseline, confirming HPI, Social History, and Brief
Hospital Course independently encode enough PTSD-associated language.

**Specificity check.** A separate Longformer trained PTSD+ vs. matched
MDD/anxiety controls with standard cross-entropy reaches AUPRC 0.91 —
PTSD-specific signal is recoverable above-and-beyond generic "psychiatric
admission" language.

**Calibration** (Longformer PULSNAR). Raw ECE 0.088, Platt-scaled 0.077,
Elkan-Noto 0.097. Elkan-Noto c = 0.728 implies ~27% undercoding rate,
consistent with PCL-5 inpatient prevalence findings. BioClinicalBERT is
substantially **over-confident** with most predictions concentrated near 1.0
(raw ECE truncated 0.48 / chunk-pool 0.51); Platt scaling cuts ECE by ~3×
but it remains well above Longformer's. Calibration is the single largest
operational gap between BERT and Longformer.

**Subgroup performance (the central deployment caveat).** AUPRC roughly
0.92 for women vs. 0.83 for men; ~0.94 for patients in their 20s vs. ~0.85
for the "Other" age bucket; ~0.92 for emergency vs. ~0.83 for elective. Race
binary EO difference is minimal (~0.024); sex EO ~0.114; age EO ~0.211.
The pattern is shared across Longformer and BioClinicalBERT — the model
performs best on the demographics most likely to be coded today, weakest
where undercoding is most prevalent — exactly the residual selection bias
PU learning reduces but does not eliminate.

**Decision Curve Analysis.** Positive net benefit over treat-none across
thresholds 0.01–0.30 at 5% prevalence; at 2% prevalence, the model is
competitive with treat-all only at moderate thresholds — the cost of missing
a case is high enough at very low prevalence that flagging everyone is
competitive in the very-low-threshold band.

**Integrated Gradients.** *Longformer:* HPI dominates (43.2%), then Brief
Hospital Course (32.0%), PMH (22.4%), Social History (1.3%). Top attributed
words are clinically appropriate trauma/psychiatric/substance vocabulary
(`bipolar`, `narcotic`, `arrested`, `assault`, `psychosis`, `pancreatitis`,
`schizoaffective`); no label-leakage tokens appear.
*BioClinicalBERT (truncated):* HPI 58.2%, PMH 27.4%, BHC 13.0%, Social History
0.7% — the truncation skews attribution toward early sections (HPI/PMH).
*BioClinicalBERT (chunk-pool, top-window):* HPI 53.5%, PMH 27.3%, BHC 17.3%,
Social History 1.2% — chunk-pool widens the BHC share by allowing windows
deeper in the note to drive the prediction. Top BERT words across both modes
are heavily psychiatric-comorbid (`psych`, `anxiety`, `bipolar`, `psychiatric`,
`disorder`, `dilaudid`, `overdose`, `suicide`, `abuse`, `methadone`); BERT is
more comorbidity-anchored and less trauma-narrative-anchored than Longformer.

**Cross-model agreement** (full table in `results/metrics/ewang163_cross_model_mcnemar.csv`):

| Pair | Agreement | Cohen's κ | Pearson r (probs) | McNemar p | Top-quintile overlap |
|---|---:|---:|---:|---:|---:|
| Longformer-PULSNAR vs BERT chunk-pool | 86.85 % | 0.737 | 0.527 | **0.107** (n.s.) | 83.5 % |
| Longformer-PULSNAR vs BERT truncated  | 85.88 % | 0.718 | 0.575 | 0.043 | 75.8 % |
| BERT truncated vs BERT chunk-pool     | 88.72 % | 0.774 | 0.863 | 4.5e-5 | 84.8 % |
| Longformer-PULSNAR vs Structured      | 56.03 % | 0.113 | 0.353 | < 1e-300 | 33.2 % |
| Longformer-PULSNAR vs Keyword         | 50.61 % | 0.000 | 0.223 | < 1e-300 | 38.4 % |

The non-significant McNemar p between PULSNAR Longformer and BERT chunk-pool
(p = 0.11) is the most notable cross-model finding — chunk-pool BERT actually
classifies *more patients correctly* than Longformer on McNemar disagreement
counts (b = 90 / c = 114), and the AUPRC gap of 0.007 is within the
McNemar-detectable noise band. The two transformers agree on ~83 % of the
top-quintile-by-score, suggesting the high-confidence flags are largely
shared. Structured + Keyword are essentially independent rankings (κ < 0.15,
overlap ~30–40 %). A max-pool ensemble over rank-normalised probabilities
*hurts* (AUPRC 0.751) because including the noisier rankings pulls the
ensemble down; mean-pool 0.874 is competitive but does not beat the best
individual model. **No ensemble lift was found across the deployed lineup.**

**Compute frontier.** Longformer 80.4 ms/patient on L40S (3.5 GPU-h training);
BERT chunk-pool 22.7 ms/patient (~0.22 GPU-h training); keyword 0.34 ms/patient
on 16 CPUs (zero training). Architecture (4,096 vs. 512 attention) — not GPU
generation — drives the gap.

### Limitations

- **Section filtering is not perfect leakage prevention.** Even on a
  pre-diagnosis admission, PMH may carry forward "history of PTSD" from outside
  records. The full-PMH-removal ablation bounds residual leakage at ~6 AUPRC
  points across both Longformer and BERT.
- **PU learning reduces but does not eliminate selection bias.** PULSNAR's
  propensity weighting up-weights underrepresented PTSD positives in the loss
  but cannot create labels for them — exactly why the older-patient and
  male-patient AUPRC gaps persist.
- **The pre-diagnosis training subsample is not representative.** Only 43.6% of
  PTSD+ patients had pre-diagnosis admissions; the remaining 56.4% use
  index-admission notes with masking applied and carry more residual leakage
  risk.
- **Proxy validation set is small (n = 102) and has an estimated 15–20% FPR**
  from off-label prazosin use not covered by the exclusion ICD set.
- **BioClinicalBERT is poorly calibrated out of the box** (raw ECE 0.48–0.51).
  Platt scaling on validation cuts ECE to ~0.17–0.21 but absolute probabilities
  still need post-hoc calibration before use in any threshold-sensitive
  workflow.
- **Single-site data.** MIMIC-IV is one academic medical center in Boston;
  transfer to community hospitals, rural settings, or VA facilities is unknown.

### What this tool is and is not

**It is** a screening prompt — a way to point inpatient clinicians at patients
whose narrative notes contain language patterns associated with PTSD. The
output is intended to inform a more thorough psychiatric evaluation, not to
make a diagnosis. At 2% deployment prevalence and the chosen operating
threshold, the tool would surface ~16 patients per true case for clinician
review under PULSNAR (NNS = 15.8).

**It is not** a diagnostic tool, a replacement for structured PTSD screening
(PCL-5, CAPS-5), or a basis for ICD coding.

### Future directions

- External validation at a non-MIMIC site (a VA medical center, where PTSD
  prevalence is much higher and the tool would arguably be most valuable).
- Re-evaluation under a true reference standard — a small prospective cohort
  screened with PCL-5 or CAPS-5 would let absolute model performance be
  measured against a defensible gold standard rather than against contaminated
  ICD labels.
- Bias mitigation for demographic subgroups — a richer PULSNAR propensity
  feature set (without the `n_prior_admissions` artifact discovered in the
  v2 propensity model) might widen the SAR-aware fairness benefit.
- Concatenating structured features (excluding the artifactual
  `n_prior_admissions`) into the Longformer head for complementary signal
  where text features are weakest.
- Post-hoc Platt or isotonic calibration on BioClinicalBERT to bring its
  probabilities into a deployable range, since BERT's ~3.5× faster inference
  is operationally meaningful.

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
| `ewang163_ptsd_train_pulsnar.py` | **Primary training script.** PULSNAR (SAR-aware) propensity-weighted nnPU. Estimates propensity scores P(coded \| features) via logistic regression on prior-admission demographics/comorbidities/medications, reweights the nnPU positive loss by 1/e(x), and uses PULSNAR for class-prior estimation (α = 0.196 with the 4-feature propensity model). Fine-tunes Clinical Longformer (4,096 tokens) at lr 1e-5, 3 epochs. |
| `ewang163_ptsd_train_longformer.py` (+ `.sbatch`) | Plain Kiryo nnPU baseline for Clinical Longformer (no propensity reweighting). Kept for sensitivity comparison; can be invoked via `--pi_p` to override the empirical class prior. |
| `ewang163_ptsd_train_bioclinbert.py` (+ `.sh`) | Fine-tunes BioClinicalBERT (512 tokens) with the same Kiryo nnPU loss; runs McNemar's test against Longformer on the test set. Inference modes (truncated vs. chunk-pool) are switched at evaluation time, not training time. |
| `ewang163_ptsd_train_structured.py` (+ `.sh`) | Logistic regression baseline on 20 structured features (demographics, prior comorbidities, prior medications); tunes C on val AUPRC and reports feature coefficients. |
| `ewang163_ptsd_train_keyword.py` | Zero-training keyword/phrase-lookup baseline. Scores notes using a hand-curated DSM-5/PCL-5 phrase lexicon (62 weighted regex patterns across criteria A–E + treatment signals). Two scoring variants compared (raw weighted count, TF-normalized); best variant selected on validation AUPRC. |
| `ewang163_ptsd_specificity.py` (+ `.sh`) | Specificity check: retrains Clinical Longformer on PTSD+ vs. age/sex-matched MDD/anxiety controls using standard cross-entropy to assess whether the model learned PTSD-specific signal vs. generic psychiatric language. |

### Evaluation & analysis (`scripts/04_evaluation/`)

| Script | Description |
|---|---|
| `ewang163_ptsd_pulsnar_reeval.py` | **Canonical Longformer evaluation.** Runs PULSNAR Longformer inference on val + test, derives the val threshold at recall ≥ 0.85, computes calibration (raw + Platt + Elkan-Noto), clinical utility, and fairness. Outputs land in `results/predictions/*_pulsnar.csv` and `results/metrics/*_pulsnar.{json,csv}`. |
| `ewang163_ptsd_bert_full_eval.py` (+ `.sh`) | **BioClinicalBERT full evaluation suite.** Runs both inference modes (truncated 512 + chunk-pool 512×256) and produces, per mode: val + test predictions, calibration (raw + Platt + Elkan-Noto + ECE plot), DCA at 2% / 5% deployment prevalence, fairness (cal-in-large + EO + bootstrap CI AUPRC), per-subgroup AUPRC, proxy validation, error analysis (FP/FN demographics + trauma terms), and ablations (PTSD-string masking + PMH removal). One job, two modes — to mirror the full Longformer analysis suite for symmetric comparison. |
| `ewang163_ptsd_bert_attribution.py` (+ `.sh`) | **BioClinicalBERT Integrated Gradients.** Runs Captum IG on the BERT embedding layer for two slices: (a) the truncated 512-token input and (b) the highest-scoring chunk-pool window per note. Same 50-patient high-confidence true-positive sample as the Longformer attribution; section + whole-word aggregation matches `attribution_v2.py`. |
| `ewang163_ptsd_cross_model.py` (+ `.sh`) | **Cross-model comparison.** All-pairs McNemar (continuity-corrected p), pairwise agreement % + Cohen's kappa + Pearson r on probabilities, top-quintile rank overlap, per-subgroup NNS at deployment prevalences, and a max/mean-pool rank-normalised ensemble probe across the 5 deployed models. |
| `ewang163_ptsd_evaluate.py` (+ `.sh`) | Legacy multi-model evaluator. Now only used to produce val-derived thresholds for the structured + keyword baselines (read by cross_model.py). The Longformer / BERT numbers it computes have been superseded by the dedicated scripts above. |
| `ewang163_ptsd_calibration.py` (+ `.sh`) | Original Longformer-only calibration script (Platt + Elkan-Noto + ECE plot). Kept for reproducibility; the canonical PULSNAR calibration is computed in-line by `ewang163_ptsd_pulsnar_reeval.py`. |
| `ewang163_ptsd_decision_curves.py` (+ `.sh`) | Decision Curve Analysis at 2% and 5% deployment prevalences. Prevalence-recalibrates probabilities via Bayes' rule and plots net-benefit curves over thresholds 0.01–0.40. |
| `ewang163_ptsd_proxy_validation.py` (+ `.sh`) | External validation on the held-out pharmacological proxy group; compares Longformer scores between proxy patients and a 500-patient random sample drawn from the training-pool unlabeled patients using Mann-Whitney U and AUC. |
| `ewang163_ptsd_ablations.py` (+ `.sh`) | Original Longformer-only ablations (PTSD-string masking + PMH removal). Per-model ablations for both BERT modes are now produced inline by `ewang163_ptsd_bert_full_eval.py`. |
| `ewang163_ptsd_attribution.py` (+ `.sh`) | First-pass Longformer Integrated Gradients via Captum's `LayerIntegratedGradients` (largely deprecated due to Longformer global-attention conflicts). |
| `ewang163_ptsd_attribution_v2.py` (+ `.sh`) | Updated Longformer IG using `IntegratedGradients` on the embedding tensor via a custom wrapper that handles Longformer's hybrid local/global attention. Full 4,096-token context, n_steps=20, word-level aggregation merges contiguous BPE subwords with summed attributions. Top words and per-section attribution. Now defaults to the PULSNAR symlinked checkpoint. |
| `ewang163_ptsd_error_analysis.py` (+ `.sh`) | Samples 25 false positives and 25 false negatives at the operating threshold for the **PULSNAR** Longformer, writes annotated notes to text files, computes aggregate statistics. |
| `ewang163_ptsd_fairness.py` | Calibration-in-the-large per subgroup, equal opportunity difference, and bootstrap 95% CI on AUPRC (1,000 resamples). |
| `ewang163_ptsd_temporal_eval.py` | Compares temporal-trained model on temporal test, random-trained model on temporal test, and random-trained model on random test — measures generalization across the ICD-9→ICD-10 transition and post-DSM-5 reclassification. |
| `ewang163_unified_inference_bench.py` (+ `.sh`) | Apples-to-apples inference timing for all five models in a single L40S GPU allocation. |
| `ewang163_cpu_inference_bench.py` | Re-measures inference for CPU-only baselines (keyword, structured) under realistic 16-CPU allocation. |

### Shared utilities (`scripts/common/`)

| Script | Description |
|---|---|
| `ewang163_bench_utils.py` | `BenchmarkLogger` context manager. Captures wall-clock time, CPU time, peak memory, and GPU-hours per pipeline stage; appends rows to `results/metrics/ewang163_runtime_benchmarks.csv`. Used by all training and evaluation scripts. |

### Top-level utilities

| Script | Description |
|---|---|
| `ewang163_runtime_audit.py` | Captures keyword + structured inference timings as benchmark rows. |

---

## 3. Repository Layout

```
ewang163/
├── CLAUDE.md                                  Authoritative project specification
├── README.md                                  This file
├── ewang163_project_writeup.md                Full project write-up
├── ewang163_model_selection_memo.md           Final model selection memo
├── ewang163_model_comparison.md               Multi-model comparison (discrimination + runtime + explainability)
├── ewang163_methodology_fixes_results.md      Methodology audit execution results
├── methodology_fix_plans.md                   Pre-execution methodology fix plans (literature-grounded)
│
├── scripts/                                   Pipeline code, ordered by stage
│   ├── common/ewang163_bench_utils.py
│   ├── 01_cohort/                             Cohort construction from MIMIC-IV
│   ├── 02_corpus/                             Corpus assembly + patient-level splits
│   ├── 03_training/                           Model training (PULSNAR primary, BERT, structured, keyword, specificity)
│   └── 04_evaluation/                         All downstream analyses (per-model + cross-model)
│
├── data/                                      [git-ignored — local only]
│   ├── cohort/                                Subject ID lists, admission extracts
│   ├── notes/                                 Section-filtered note parquets
│   └── splits/                                Patient-level train/val/test splits (random + temporal)
│
├── models/                                    Trained model artefacts
│   ├── ewang163_longformer_best/              → symlinked to ewang163_longformer_pulsnar/
│   ├── ewang163_longformer_pulsnar/           Primary deployment checkpoint
│   ├── ewang163_longformer_best_temporal/     Pre-2015 train (not recommended for deployment)
│   ├── ewang163_bioclinbert_best/             Comparison BERT model
│   ├── ewang163_specificity_longformer_best/  Specificity check vs. psych controls
│   ├── ewang163_structured_logreg.pkl + ewang163_structured_features.json
│   ├── ewang163_keyword_weights.json          DSM-5/PCL-5 phrase lexicon (62 patterns)
│   └── ewang163_platt_calibrator.pkl          Post-hoc probability calibrator
│
├── results/
│   ├── table1/                                Cohort characterization (CSV + plain text)
│   ├── predictions/                           Per-patient predicted probabilities (val + test, all models)
│   ├── metrics/                               JSON + CSV evaluation outputs (eval, ablations, calibration, DCA, proxy, fairness, cross-model)
│   ├── figures/                               PNG plots (calibration, DCA, proxy histogram)
│   ├── attribution/                           IG section + token + word attribution (Longformer + BERT)
│   └── error_analysis/                        Sampled FP/FN summary stats (Longformer + BERT)
│
├── logs/                                      SLURM .out files from every job submitted
├── PULSNAR/                                   Vendored Kumar & Lambert 2024 PULSNAR library [git-ignored]
└── ptsd_env/                                  Python virtual environment
```

---

## 4. Installation

### Requirements

- Python 3.13+
- Packages: `pandas`, `numpy`, `transformers`, `torch`, `scikit-learn`,
  `captum`, `scipy`, `matplotlib`, `joblib`, `xgboost` (for PULSNAR),
  `pyarrow` (parquet)
- For PULSNAR class-prior estimation: see `PULSNAR/requirements.txt`
  (xgboost, catboost, threadpoolctl). The `rpy2`/R dependency is lazy
  and only triggered by non-default bandwidth methods.

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

**All scripts must be submitted via SLURM, never run interactively on the
login node.** Login-node usage is monitored — exceeding limits disrupts the
user account.

### CPU jobs (data processing, baselines)

```bash
sbatch --partition=batch --mem=16G --time=2:00:00 \
       --output=ewang163_%j.out --wrap="python <script>.py"
```

### GPU jobs (transformer training and inference)

Most GPU scripts ship a paired `.sh` / `.sbatch` file with the SLURM header
embedded:

```bash
sbatch scripts/03_training/ewang163_ptsd_train_pulsnar.py
sbatch scripts/04_evaluation/ewang163_ptsd_bert_full_eval.sh
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

Each stage saves its output to `STUDENT_DIR` before the next stage begins, so
any stage can be re-run independently.

1. `scripts/01_cohort/ewang163_ptsd_table1.py` → `results/table1/`
2. `scripts/01_cohort/ewang163_ptsd_cohort_sets.py` → `data/cohort/*_subjects.txt`
3. `scripts/01_cohort/ewang163_ptsd_admissions_extract.py` → `data/cohort/ewang163_ptsd_adm_extract.parquet`
4. `scripts/01_cohort/ewang163_ptsd_notes_extract.py` → `data/notes/ewang163_ptsd_notes_raw.parquet`
5. `scripts/02_corpus/ewang163_ptsd_corpus_build.py` → `data/notes/ewang163_ptsd_corpus.parquet`
6. `scripts/02_corpus/ewang163_ptsd_splits.py` (and `--temporal`) → `data/splits/`
7. `scripts/03_training/ewang163_ptsd_train_keyword.py` (CPU, seconds)
8. `scripts/03_training/ewang163_ptsd_train_structured.py` (CPU, ~70 s)
9. `scripts/03_training/ewang163_ptsd_train_bioclinbert.py` (GPU, ~14 min)
10. `scripts/03_training/ewang163_ptsd_train_pulsnar.py` (GPU, ~3.5 GPU-h) — primary model
11. `scripts/03_training/ewang163_ptsd_specificity.py` (GPU, ~7.5 GPU-h, secondary)
12. `scripts/04_evaluation/ewang163_ptsd_pulsnar_reeval.py` (Longformer val + test + calibration + fairness + utility)
13. `scripts/04_evaluation/ewang163_ptsd_bert_full_eval.py` (BERT both modes — predictions + calibration + DCA + fairness + ablations + proxy + error analysis)
14. `scripts/04_evaluation/ewang163_ptsd_bert_attribution.py` (BERT IG, both slices)
15. `scripts/04_evaluation/ewang163_ptsd_attribution_v2.py` (Longformer IG)
16. `scripts/04_evaluation/ewang163_ptsd_proxy_validation.py` (Longformer proxy plot)
17. `scripts/04_evaluation/ewang163_ptsd_decision_curves.py` (Longformer DCA)
18. `scripts/04_evaluation/ewang163_ptsd_error_analysis.py` (Longformer FP/FN sampling)
19. `scripts/04_evaluation/ewang163_ptsd_fairness.py` (Longformer fairness re-run; redundant with pulsnar_reeval but produces the standalone CSV)
20. `scripts/04_evaluation/ewang163_ptsd_temporal_eval.py`
21. `scripts/04_evaluation/ewang163_ptsd_evaluate.py` (legacy — produces structured + keyword val thresholds in `results/metrics/ewang163_evaluation_results.json`)
22. `scripts/04_evaluation/ewang163_ptsd_cross_model.py` (all-pairs McNemar, agreement, ensemble — runs after the per-model scripts)
23. `scripts/04_evaluation/ewang163_unified_inference_bench.py` (apples-to-apples timing)

### Reproducibility

Random seeds are pinned to 42 throughout (matching, sampling, splits, model
training, bootstrap resampling). All design decisions, ICD code lists, drug
lists, MIMIC-IV data quirks, and bug fixes are documented in `CLAUDE.md`.
SLURM job logs for every run are preserved under `logs/`. The full project
write-up with detailed results discussion is in `ewang163_project_writeup.md`.
