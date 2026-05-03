# Methodology — Execution Results and Interpretation

**Author:** Eric Wang (ewang163)
**Date:** 2026-05-03

This document records the execution outcomes for each methodology decision
listed in `methodology_fix_plans.md`. The deployment-model-selection
narrative is in `ewang163_model_selection_memo.md`. The full multi-model
comparison is in `ewang163_model_comparison.md`.

---

## Final deployment model

**Clinical Longformer fine-tuned with PULSNAR propensity-weighted nnPU loss
(α = 0.1957)** on the section-filtered, PTSD-string-masked MIMIC-IV corpus.

- **Test AUPRC = 0.8848**
- **Test AUROC = 0.8904**
- **Proxy Mann-Whitney AUC = 0.7701** (p = 3.8e-18) — only non-circular validity metric
- **Sensitivity = 0.846, Specificity = 0.745, F1 = 0.772** at val-derived threshold 0.188
- **NNS at 2% deployment prevalence = 15.8**

PULSNAR (Kumar & Lambert 2024) extends Kiryo et al.'s nnPU to the SAR
(Selected At Random conditional on features) regime that matches the
data-generating process — PTSD coding is biased toward younger women with
prior psychiatric contact, not random within the unlabeled pool.

---

## 1. Execution Summary

All scripts run on Brown's Oscar HPC cluster via SLURM. CPU jobs on the
`batch` partition; GPU jobs on `gpu` (NVIDIA L40S primary; some baselines
on RTX 3090).

### Cohort + corpus + splits

| Script | Status |
|--------|--------|
| `01_cohort/ewang163_ptsd_table1.py` | ✅ |
| `01_cohort/ewang163_ptsd_cohort_sets.py` | ✅ |
| `01_cohort/ewang163_ptsd_admissions_extract.py` | ✅ |
| `01_cohort/ewang163_ptsd_notes_extract.py` | ✅ |
| `02_corpus/ewang163_ptsd_corpus_build.py` (universal PTSD masking) | ✅ |
| `02_corpus/ewang163_ptsd_splits.py` (random + `--temporal`) | ✅ |

### Training

| Script | Role | Status |
|--------|------|--------|
| `03_training/ewang163_ptsd_train_pulsnar.py` | **PRIMARY model** | ✅ |
| `03_training/ewang163_ptsd_train_longformer.py` | Plain Kiryo nnPU sensitivity | ✅ |
| `03_training/ewang163_ptsd_train_bioclinbert.py` | Comparison BERT | ✅ |
| `03_training/ewang163_ptsd_train_structured.py` | Structured baseline | ✅ |
| `03_training/ewang163_ptsd_train_keyword.py` | Keyword baseline | ✅ |
| `03_training/ewang163_ptsd_specificity.py` | PTSD vs MDD/anxiety control | ✅ |

### Evaluation

| Script | Role | Status |
|--------|------|--------|
| `04_evaluation/ewang163_ptsd_pulsnar_reeval.py` | Canonical Longformer eval (val + test + cal + utility + fairness) | ✅ |
| `04_evaluation/ewang163_ptsd_bert_full_eval.py` | Canonical BERT eval — both inference modes, full downstream suite | ✅ |
| `04_evaluation/ewang163_ptsd_attribution_v2.py` | Longformer Integrated Gradients (4,096-context) | ✅ |
| `04_evaluation/ewang163_ptsd_bert_attribution.py` | BERT Integrated Gradients (truncated + chunk-pool top window) | ✅ |
| `04_evaluation/ewang163_ptsd_cross_model.py` | All-pairs McNemar + κ + Pearson r + ensemble probe | ✅ |
| `04_evaluation/ewang163_ptsd_proxy_validation.py` | Longformer proxy plot | ✅ |
| `04_evaluation/ewang163_ptsd_decision_curves.py` | Longformer DCA (BERT DCA inline in `bert_full_eval.py`) | ✅ |
| `04_evaluation/ewang163_ptsd_calibration.py` | Longformer calibration (canonical version inline in `pulsnar_reeval.py`) | ✅ |
| `04_evaluation/ewang163_ptsd_ablations.py` | Longformer ablations (BERT inline in `bert_full_eval.py`) | ✅ |
| `04_evaluation/ewang163_ptsd_error_analysis.py` | Longformer FP/FN sampling + lexicon overrepresentation | ✅ |
| `04_evaluation/ewang163_ptsd_fairness.py` | Longformer fairness (canonical inline in `pulsnar_reeval.py`) | ✅ |
| `04_evaluation/ewang163_ptsd_temporal_eval.py` | Pre-2015-train / 2017–2019-test generalization | ✅ |
| `04_evaluation/ewang163_ptsd_evaluate.py` | Legacy evaluator — produces structured + keyword val thresholds for `cross_model.py` | ✅ |
| `04_evaluation/ewang163_unified_inference_bench.py` | Apples-to-apples L40S inference timing | ✅ |
| `04_evaluation/ewang163_cpu_inference_bench.py` | 16-CPU baseline inference timing | ✅ |

---

## 2. Bugs Found and Fixed During Testing

### 2a. Temporal split date shift

**Problem:** MIMIC-IV uses per-patient random date shifts (~100–200 years),
so each patient has a different offset. The original temporal split code
compared raw `admittime` to a calendar date `2015-01-01` — landed in zero
patients because all shifted dates are in the 2100s–2200s range.

**Fix:** Rewrote `temporal_split()` to join `patients.csv` and use the
`anchor_year_group` column (e.g., "2008 - 2010", "2014 - 2016"), which
contains the real 3-year window. Patients with `anchor_year_group` starting
before 2015 → train; 2017–2019 → test.

### 2b. Cross-model PULSNAR prediction-row alignment

**Problem:** `cross_model.py` initially merged the PULSNAR Longformer test
predictions on `subject_id` only (the predictions CSV from `pulsnar_reeval.py`
lacks `hadm_id`), but the test split has 1,551 rows / 1,207 unique subjects
(some patients contribute multiple admissions). The naïve subject-only merge
re-shuffled row alignment for multi-admission patients, dropping the
Longformer AUPRC from 0.885 to 0.816 — a misleading sanity-check failure.

**Fix:** Distinguish predictions with vs. without `hadm_id` in
`load_predictions()`. For predictions with `hadm_id` (BERT, structured,
keyword), merge on (subject_id, hadm_id). For predictions without `hadm_id`
(PULSNAR Longformer), trust positional alignment with `test_df` row order
and verify subject_id matches as a sanity check. Post-fix Longformer
AUPRC = 0.8848 ✓.

### 2c. Per-mode BERT thresholds

**Problem:** Earlier evaluation reused a single BERT val threshold (0.976,
derived in truncated mode) for both the truncated and chunk-pool inference.
Chunk-pool's max-pool aggregation pushes the score distribution upward, so
the chunk-pool model needs its own (higher) val threshold.

**Fix:** `bert_full_eval.py` derives the threshold *separately* per inference
mode. Truncated 0.976; chunk-pool 0.993.

### 2d. Keyword baseline threshold saturation (expected behaviour, not a bug)

The keyword baseline's val-derived threshold is 0.0, meaning it flags
everyone as positive (recall = 1.0, specificity = 0.0). This happens
because ~50% of PTSD+ notes score 0 on the keyword lexicon. Expected
behaviour for a zero-training baseline; correctly demonstrates why NLP
methods are needed.

---

## 3. Key Results

### 3a. Universal PTSD-string masking — pre-diagnosis leakage audit

| Metric | Value |
|--------|-------|
| Pre-diagnosis PTSD+ notes audited | 4,169 |
| Notes containing PTSD-related strings | 360 (**8.6 %**) |
| Total PTSD+ notes masked (pre-dx + fallback) | 5,950 |
| Notes with `[PTSD_MASKED]` after masking | 1,655 (27.8 %) |

**Interpretation:** 8.6 % of the primary training set contained explicit
PTSD-related strings that could leak the label into the model. The original
assumption that "the PTSD label cannot appear in notes from before the
patient was ever coded" was wrong for ~1 in 12 patients. Masking closes
this leakage path.

### 3b. Validation-derived thresholds (per model)

| Model | Threshold | Source |
|-------|-----------|--------|
| Clinical Longformer (PULSNAR, primary) | **0.188** | val-derived |
| BioClinicalBERT (truncated) | 0.976 | val-derived (own) |
| BioClinicalBERT (chunk-pool) | 0.993 | val-derived (own) |
| Structured + LogReg | 0.338 | val-derived |
| Keyword (DSM-5/PCL-5) | 0.000 | val-derived |

### 3c. Model comparison (test set, val-derived thresholds)

| Model | AUPRC | AUROC | Sens | Spec | Prec | F1 | NNS @ 2 % | McNemar p vs. PULSNAR |
|-------|------:|------:|-----:|-----:|-----:|---:|---------:|---:|
| **Clinical Longformer (PULSNAR)** | **0.8848** | **0.8904** | 0.846 | 0.745 | 0.711 | 0.772 | 15.8 | — |
| BioClinicalBERT (chunk-pool) | 0.8775 | 0.8853 | 0.846 | 0.772 | 0.733 | **0.785** | **14.2** | **0.107 (n.s.)** |
| BioClinicalBERT (truncated) | 0.8576 | 0.8656 | 0.821 | 0.728 | 0.691 | 0.751 | 17.2 | 0.043 |
| Structured + LogReg | 0.6833 | 0.7310 | 0.909 | 0.207 | 0.459 | 0.610 | 36.7 | < 1e-300 |
| Keyword (DSM-5/PCL-5) | 0.5096 | 0.6190 | 1.000 | 0.000 | 0.426 | 0.597 | 50.0 | < 1e-300 |

**Headlines:**

1. **PULSNAR Longformer wins discrimination** (AUPRC, AUROC). Smallest sex
   and race fairness gaps. Best-calibrated raw probabilities.
2. **BioClinicalBERT chunk-pool is statistically indistinguishable from
   Longformer on McNemar (p = 0.107).** Wins F1, LR+, DOR, and NNS at the
   operating point because of higher specificity at threshold — but raw ECE
   is 5–6× Longformer's, requiring post-hoc Platt calibration before deployment.
3. **Chunk-and-pool beats truncation** by +0.020 AUPRC (McNemar p = 4.5e-5)
   because chunk-pool can see the BHC section which truncation cuts off.
4. **The keyword baseline establishes a floor** (AUPRC 0.510 vs random
   baseline 0.426).

### 3d. Clinical utility (PULSNAR Longformer, val-derived threshold 0.188)

| Metric | Value |
|--------|-------|
| Alert rate | 50.6 % |
| LR+ | 3.32 |
| LR− | 0.207 |
| DOR | 16.0 |
| Workup reduction vs. treat-all | 49.4 % |

Prevalence recalibration:

| Deployment prevalence | PPV | NPV | NNS |
|-----------------------|----:|----:|----:|
| 1 % | 0.032 | 0.998 | 30.8 |
| 2 % | 0.063 | 0.996 | 15.8 |
| 5 % | 0.149 | 0.989 | 6.7 |
| 10 % | 0.269 | 0.978 | 3.7 |
| 20 % | 0.453 | 0.951 | 2.2 |

NPV > 0.99 means a negative screen is highly reliable. LR+ = 3.32 is
"moderate" — appropriate for a screening prompt, not for diagnosis.

### 3e. Calibration (all three text models)

| Calibration variant | PULSNAR Longformer | BERT trunc | BERT chunk-pool |
|---------------------|--------------------:|-----------:|---------------:|
| ECE raw | **0.088** | 0.482 | 0.507 |
| ECE Platt-scaled | 0.077 | 0.174 | 0.208 |
| ECE Elkan-Noto | 0.097 | 0.173 | 0.208 |
| Elkan-Noto c | 0.728 | 0.983 | 0.991 |

PULSNAR Longformer's raw probabilities are the only deployable ones.
BioClinicalBERT is severely over-confident; raw probabilities concentrate
near 1.0 (c estimates are no longer informative as labelling-frequency
estimates). Platt scaling cuts BERT ECE by ~3× but it remains 2–3× higher
than Longformer's raw — BERT requires post-hoc calibration before any
threshold-sensitive deployment.

The Longformer Elkan-Noto c estimate (0.728) implies ~27 % undercoding rate,
consistent with PCL-5 inpatient prevalence findings (Stanley et al. 2020).

### 3f. PU-corrected metrics (Ramola)

The Ramola correction with α = 0.196 (from PULSNAR estimation):

| Metric | Raw (PU lower bound) | Ramola-corrected (α = 0.196) |
|--------|---------------------:|-----------------------------:|
| AUPRC | 0.8848 | 1.0 (ceiling-clipped) |
| AUROC | 0.8904 | **0.987** |

The corrected AUROC of 0.987 suggests the model's true discrimination
(against actual PTSD status, not just ICD codes) is substantially higher
than the raw 0.890. The raw metrics remain the primary conservative
report, labelled as "PU lower bounds," with the corrections as the
optimistic bound.

### 3g. Temporal split — temporal training hurt generalization

| Scenario | Test AUPRC |
|----------|----------:|
| Random-split model on random test | 0.888 |
| Random-split model on temporal test (2017–2019) | 0.886 |
| **Temporal-trained model on temporal test** | 0.842 |

Random-split model loses only 0.002 AUPRC on temporal test (random distribution
is already representative of late MIMIC-IV); temporal-trained loses 0.044 because
of less training data and missed post-2013 DSM-5-era patterns. **Random split
recommended for deployment.** Temporal model archived but not recommended.

### 3h. Fairness (all three text models)

Equal opportunity differences (recall gap at val-derived threshold):

| Subgroup | PULSNAR Longformer | BERT trunc | BERT chunk-pool |
|----------|-------------------:|-----------:|---------------:|
| Sex (F vs M) | **0.114** | 0.151 | 0.127 |
| Age | 0.211 | 0.237 | **0.181** |
| Race binary (W vs Non-W) | **0.024** | 0.064 | 0.047 |
| Emergency | **0.046** | 0.038 | 0.067 |

PULSNAR Longformer has the smallest sex EO and race EO across the three
text models, and competitive age EO. All three share the same residual bias
pattern (best on younger women in emergency admissions, weakest on older
men in elective admissions) — inherited from non-SCAR PTSD coding bias.
PULSNAR's propensity reweighting reduces but does not eliminate the gap.

**AUPRC reliability:** Asian subgroup AUPRC has CI width 0.6+ across all
models — correctly flagged as unreliable (n_pos = 5). Hispanic and
Other/Unknown borderline. Black, White, and the binary Non-White / White
contrast have CI widths < 0.15 and are reported as reliable.

### 3i. Proxy validation (all three text models)

| Model | Proxy median | Unlabeled median | MW AUC | MW p |
|---|---:|---:|---:|---:|
| PULSNAR Longformer | ~0.38 | ~0.06 | **0.7701** | 3.8e-18 |
| BioClinicalBERT (truncated) | — | — | 0.7442 | 3.7e-15 |
| BioClinicalBERT (chunk-pool) | — | — | 0.7333 | 5.3e-14 |

PULSNAR Longformer's separation is strongest. All three text models clear
the validity bar (p ≪ 1e-10). Because proxy patients are identified by an
entirely independent (medication-based) criterion the model cannot see
(discharge medications are filtered out), this is the project's strongest
single piece of validity evidence.

### 3j. PULSNAR α estimation

PULSNAR (Kumar & Lambert 2024) was installed from
`github.com/unmtransinfo/PULSNAR` with a small patch making the rpy2
dependency lazy.

| Estimator | Estimated α | Method |
|-----------|------------:|--------|
| **PULSNAR (SAR), 4-feature propensity** | **0.1957** | xgboost classifier, divide-and-conquer KDE, 5 clusters found |
| Empirical labeled fraction | 0.2614 | n_pos / (n_pos + n_unl) |
| ~~PULSNAR with `n_prior_admissions` propensity feature~~ | ~~0.0006~~ | **REJECTED** — propensity model perfectly separates coded from uncoded by `n_prior_admissions` (coef +5.63), leaving PULSNAR no signal |

PULSNAR estimates that 19.6 % of the combined (labeled + unlabeled) training
pool is truly PTSD-positive, vs. the 26.1 % that are ICD-coded. The
difference (6.5 percentage points) represents the model's estimate of the
*overcounting* from the 3:1 matched design.

The richer-features alternative was rejected because adding `n_prior_admissions`
makes the propensity model "too good": it perfectly separates coded from
uncoded patients, leaving PULSNAR no residual signal to detect hidden
positives. The same `n_prior_admissions` confound surfaces in the structured
baseline (coef +6.51), and is a cohort-design artifact (Group 3 unlabeled has
index = first admission, so prior count is 0; Group 1 PTSD+ index is later
by construction). The 4-feature propensity is the principled estimate.

### 3k. BERT both inference modes (chunk-and-pool)

`ewang163_ptsd_bert_full_eval.py` runs BERT in **both** modes from the same
trained checkpoint:

- **Truncated** — first 512 tokens. AUPRC 0.858, ECE_raw 0.482.
- **Chunk-pool** — overlapping 512-token windows with stride 256, max-pool.
  AUPRC 0.878, ECE_raw 0.507.

Both modes are evaluated symmetrically (own val threshold, own calibration,
own DCA, own subgroup, own IG slice, etc.). The two modes are highly
correlated (Pearson r on probabilities = 0.86) but chunk-pool is
statistically better on McNemar (p = 4.5e-5). For deployment, chunk-pool is
the recommended BERT inference mode.

### 3l. Integrated Gradients (Longformer + BERT)

Per-section attribution share:

| Section | PULSNAR Longformer | BERT truncated | BERT chunk-pool top window |
|---|---:|---:|---:|
| HPI | 43.2 % | 58.2 % | 53.5 % |
| PMH | 22.4 % | 27.4 % | 27.3 % |
| BHC | **32.0 %** | 13.0 % | 17.3 % |
| Social History | 1.3 % | 0.7 % | 1.2 % |

**Architectural finding.** Longformer puts ~32 % of attribution on Brief
Hospital Course; BERT-truncated puts only 13 % because BHC literally falls
past the 512-token window. Chunk-pool partially recovers (17 %).

Top attributed words:
- **Longformer:** bipolar, narcotic, illness, arrested, delayed, pancreatitis, schizoaffective, psychosis, anemia, assault. Notably trauma-anchored.
- **BERT (both modes):** psych, anxiety, bipolar, psychiatric, disorder, dilaudid, overdose, suicide, abuse, methadone. More comorbidity-anchored.

**No label-leakage tokens** in any model's top attributions, confirming
the universal masking worked across both architectures.

### 3m. Cross-model agreement matrix

`ewang163_ptsd_cross_model.py` produces all-pairs McNemar, Cohen's κ,
Pearson r on probabilities, top-quintile rank overlap, per-subgroup NNS,
and a max/mean-pool ensemble probe. Key findings:

- **PULSNAR Longformer ⇄ BERT chunk-pool: McNemar p = 0.107 (n.s.).**
  Cohen's κ = 0.737, top-quintile overlap 83.5 %. The two best models are
  statistically tied.
- **BERT trunc ⇄ chunk-pool:** Cohen's κ = 0.774, Pearson r = 0.86 — same
  model, different aggregation, mostly correlated.
- **Anything ⇄ Structured/Keyword:** Cohen's κ < 0.15 — effectively
  independent rankings.
- **Ensemble:** max-pool 0.751, mean-pool 0.874 — **neither beats the best
  individual model (0.885).** No ensemble lift.

---

## 4. Runtime Benchmarks

| Model | Hardware | Wall time (s) | Peak mem | GPU-h | Test AUPRC |
|-------|----------|--------------:|---------:|------:|----------:|
| PULSNAR Longformer training | L40S (gpu2708) | 12,617 | 2.25 GB GPU | 3.50 | 0.8848 |
| BioClinicalBERT training | RTX 3090 (gpu2105) | 791 | ~3.7 GB MaxRSS | 0.22 | 0.858 / 0.878 (trunc / chunk-pool) |
| Structured + LogReg pipeline | CPU node | 68 | 0.30 GB | 0 | 0.6833 |
| Keyword scoring (training set) | 16-CPU | 127 | 0.31 GB | 0 | 0.5096 |
| Longformer val inference (1,471) | L40S | 119.8 | 2.03 GB | 0.033 | — |
| Longformer test inference (1,551) | L40S | 124.6 | 2.03 GB | 0.035 | — |
| BERT truncated test inference (1,551) | L40S | 4.6 | 0.83 GB | 0.001 | — |
| BERT chunk-pool test inference (1,551) | L40S | 35.1 | 0.65 GB | 0.010 | — |
| BERT chunk-pool val inference (1,471) | L40S | 33.2 | 0.65 GB | 0.011 | — |
| Cross-model script | CPU | 70 | 1 GB | 0 | — |
| BERT IG attribution (both slices, 50 patients) | L40S | ~50 | 0.85 GB | 0.014 | — |
| Longformer IG attribution (50 patients @ 4096) | L40S | 442–612 | 1.5 GB | 0.12–0.17 | — |

**Key takeaways:**

- BERT chunk-pool inference on identical L40S is **3.55× faster** than
  Longformer (22.7 vs 80.4 ms/patient).
- BERT chunk-pool training is **~16× cheaper** than Longformer training
  (after hardware normalization).
- Both BERT IG attribution slices together cost less than one Longformer
  IG run.

---

## 5. Interpretation for the Paper

**Strongest findings:**

1. **PULSNAR is the principled choice for SAR-violated PU learning.** The
   PULSNAR-vs-plain-nnPU comparison is **not** a horse race — it's a choice of
   which loss matches the data-generating process. Empirically, PULSNAR's
   gain over plain nnPU is in attribution profile (more trauma-narrative,
   less coding-comorbidity), better fairness (smaller sex EO), and a
   defensible non-tuned hyperparameter.

2. **BERT chunk-pool is statistically tied with Longformer on McNemar.**
   The architecture comparison is much closer than the original Li et al.
   (2022) framing. Long-context inference (not architecture per se) drives
   most of Longformer's lift over BERT-truncated, and chunk-pool BERT is
   McNemar-indistinguishable from Longformer on the test set. **Longformer's
   residual advantage is in calibration, attribution, and fairness — not raw
   discrimination.**

3. **BERT requires post-hoc calibration before deployment.** Raw ECE
   0.48–0.51 vs Longformer's 0.088. Platt scaling cuts BERT ECE by ~3× but
   still leaves it 2–3× higher than Longformer's raw. For threshold-sensitive
   deployments this is the most important practical difference between the
   two models.

4. **The same FN profile across all three transformers.** False negatives
   skew male, older, and have shorter notes — invariant to architecture. The
   under-detection pattern is a property of how those patients are
   documented, not of any one model.

5. **No ensemble lift across the deployed lineup.** Max-pool drops AUPRC
   because the structured + keyword noise dominates; mean-pool is
   competitive (0.874) but does not beat the best individual model (0.885).
   The text models largely identify the same patients.
