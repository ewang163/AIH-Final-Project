# Model Selection Memo — PULSNAR Clinical Longformer for PTSD Undercoding Detection

**Author:** Eric Wang (ewang163)
**Date:** 2026-05-03
**Purpose:** Document the final model choice and the empirical basis for that choice.

---

## Decision

**Primary deployment model:** Clinical Longformer fine-tuned with **PULSNAR**
propensity-weighted nnPU (α = 0.1957)

- Checkpoint: `models/ewang163_longformer_pulsnar/` (also accessible via the symlink `models/ewang163_longformer_best`)
- Operating threshold (val-derived at sensitivity ≥ 0.85): 0.188
- Test AUPRC = 0.8848, AUROC = 0.8904, F1 = 0.772
- Proxy Mann-Whitney AUC = 0.7701, p = 3.8e-18 (only PU-uncontaminated validity criterion)
- NNS at 2 % deployment prevalence = 15.8

**Comparison models** (all evaluated symmetrically — calibration, DCA, fairness, ablations, attribution, proxy validation):

- BioClinicalBERT (truncated 512 tokens) — `models/ewang163_bioclinbert_best/`
- BioClinicalBERT (chunk-and-pool 512 × 256) — same checkpoint, different inference mode
- Structured features + L2 logistic regression — `models/ewang163_structured_logreg.pkl`
- Keyword DSM-5/PCL-5 lexicon — `models/ewang163_keyword_weights.json`

**Specificity sanity-check model:** Standard cross-entropy Longformer trained on PTSD+ vs. age/sex-matched MDD/anxiety controls — `models/ewang163_specificity_longformer_best/`. Used for one downstream check only (§"Specificity check" below).

**Plain Kiryo nnPU Longformer** (`scripts/03_training/ewang163_ptsd_train_longformer.py`) is retained as a sensitivity-analysis variant of the primary loss but not deployed.

**Temporal sensitivity model:** `models/ewang163_longformer_best_temporal/` is archived for the temporal-generalization check; it is **not** recommended for deployment (random-split training generalizes better — see "Temporal generalization" below).

---

## Why PULSNAR over plain Kiryo nnPU — the SCAR-vs-SAR framing

| Assumption | Meaning | Applies here? |
|---|---|---|
| **SCAR** (Elkan & Noto 2008; Kiryo et al. 2017 nnPU) | Labeled positives are a uniform random sample of all true positives | **Violated.** PTSD coding is biased toward younger women, White patients, and patients with prior psychiatric contact — the fairness analysis (below) confirms exactly this pattern. |
| **SAR** (Bekker & Davis 2020 review; Kumar & Lambert 2024 PULSNAR) | Labeling probability depends on observed features | **Correct framing.** A trauma-exposed older man with substance use is less likely to be ICD-coded than a younger woman with the same symptoms. |

Literature consensus (Bekker & Davis 2020 ACM SIGKDD Explorations; Jaskie & Spanias 2022 IEEE): when SCAR is violated, the principled choice is a SAR-aware loss whose propensity model is *data-driven*, not tuned on a downstream proxy. Tuning the class prior on a SCAR-coded validation signal — including on proxy AUC, since the proxy criterion itself selects a "prazosin-adjacent" phenotype — risks reproducing the coding bias rather than correcting for it.

PULSNAR (`scripts/03_training/ewang163_ptsd_train_pulsnar.py`) implements:

1. Propensity model `e(x) = P(coded | features)` via logistic regression on 4 prior-admission demographic features (sex, age, emergency, medicaid). Clipped to [0.05, 0.95].
2. Class-prior estimation via xgboost-based PULSNAR routine (`bin_method='rice', bw_method='hist'`); fall back to PULSCAR (SCAR) then to empirical fraction. **α = 0.1957.**
3. Modified nnPU loss with positives reweighted by `1/e(x)`, which up-weights labeled positives whose coding propensity is low — exactly the under-detected population the screening tool is meant to surface.

**Why a 4-feature propensity model (not richer features):** A propensity model that adds `n_prior_admissions` perfectly separates coded from uncoded patients (coef +5.63), leaving PULSNAR no signal to detect hidden positives — α collapses to 0.0006. This is a cohort-design artifact: by construction, Group 3 (unlabeled) has index = first MIMIC-IV admission, so prior-admission count is always 0; Group 1 (PTSD+) has index = first PTSD-coded admission, which tends to be later. The same artifact dominates the structured baseline (coef +6.51). The 4-feature propensity (demographics only, no severity/admission-count proxies) is the principled choice.

---

## Key results on the held-out test set (n = 1,551, 660 PTSD+)

### Discrimination + clinical utility

| Model | Threshold | AUPRC | AUROC | Sens | Spec | F1 | LR+ | DOR | NNS @ 2 % | Calibration ECE_raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PULSNAR Longformer (primary)** | 0.188 | **0.8848** | **0.8904** | 0.846 | 0.745 | 0.772 | 3.32 | 16.0 | 15.8 | **0.088** |
| BioClinicalBERT (chunk-pool) | 0.993 | 0.8775 | 0.8853 | 0.846 | 0.772 | **0.785** | **3.71** | **18.6** | **14.2** | 0.507 |
| BioClinicalBERT (truncated) | 0.976 | 0.8576 | 0.8656 | 0.821 | 0.728 | 0.751 | 3.02 | 12.3 | 17.2 | 0.482 |
| Structured + LogReg | 0.338 | 0.6833 | 0.7310 | 0.909 | 0.207 | 0.610 | 1.15 | 2.6 | 36.7 | not computed |
| Keyword (DSM-5/PCL-5) | 0.000 | 0.5096 | 0.6190 | 1.000 | 0.000 | 0.597 | 1.00 | 1.00 | 50.0 | not computed |

**PULSNAR Longformer wins discrimination.** BERT chunk-pool wins F1, LR+, DOR, and NNS @ 2% at the operating point — but only because BERT's *uncalibrated* threshold lands at higher specificity. After Platt scaling (post-hoc), BERT's clinical utility is similar to Longformer's. PULSNAR Longformer's 3–6× lower ECE means it can ship with raw probabilities; BERT requires post-hoc calibration before deployment.

### McNemar all-pairs

| Pair | McNemar p | b (A correct/B wrong) | c (A wrong/B correct) | Cohen's κ |
|---|---:|---:|---:|---:|
| LF-PULSNAR vs BERT chunk-pool | **0.107 (n.s.)** | 90 | 114 | 0.737 |
| LF-PULSNAR vs BERT truncated | 0.043 | 125 | 94 | 0.718 |
| LF-PULSNAR vs Structured | < 1e-300 | 560 | 122 | 0.113 |
| LF-PULSNAR vs Keyword | < 1e-300 | 664 | 102 | 0.000 |
| BERT trunc vs chunk-pool | 4.5e-5 | 60 | 115 | 0.774 |

**PULSNAR Longformer and BERT chunk-pool are statistically indistinguishable on McNemar (p = 0.107)** — chunk-pool is correct on slightly more McNemar disagreements but the AUPRC gap of 0.007 is within the McNemar-detectable noise band. Longformer beats both BERT-truncated (p = 0.043) and the structured/keyword baselines (p ≈ 0). The two BERT modes differ significantly from each other (p = 4.5e-5), with chunk-pool clearly preferred.

### Proxy validation (non-circular)

| Model | Proxy MW AUC | MW p |
|---|---:|---:|
| PULSNAR Longformer | **0.7701** | 3.8e-18 |
| BERT (truncated) | 0.7442 | 3.7e-15 |
| BERT (chunk-pool) | 0.7333 | 5.3e-14 |

PULSNAR Longformer assigns the strongest separation between proxy patients and unlabeled controls — the most direct piece of non-circular validity evidence. All three text models clear the validity bar (p ≪ 1e-10).

### Ablations (label leakage)

| Ablation | PULSNAR LF Δ | BERT trunc Δ | BERT chunk-pool Δ |
|---|---:|---:|---:|
| 1: post-hoc PTSD masking | −0.008 | −0.013 | −0.012 |
| 2: PMH section removed | −0.061 | −0.063 | −0.066 |

All three text models depend comparably on the PMH section (~6 AUPRC points) and almost not at all on the literal PTSD strings (< 1.5 AUPRC points). The PMH dependence is a property of the data + masking design, not of any specific architecture.

### Fairness (equal opportunity differences at val-derived threshold)

| Subgroup | PULSNAR LF | BERT trunc | BERT chunk-pool |
|---|---:|---:|---:|
| Sex (F vs M) | **0.114** | 0.151 | 0.127 |
| Age | 0.211 | 0.237 | **0.181** |
| Race (W vs Non-W binary) | **0.024** | 0.064 | 0.047 |
| Emergency | **0.046** | 0.038 | 0.067 |

**PULSNAR Longformer has the smallest sex EO and race EO, and competitive age EO.** BERT chunk-pool partially recovers on age (EO 0.181). All three models share the same residual bias pattern: best on younger women in emergency admissions, weakest on older men in elective admissions — inherited from non-SCAR coding bias.

### Calibration

| Variant | PULSNAR LF | BERT trunc | BERT chunk-pool |
|---|---:|---:|---:|
| ECE raw | **0.088** | 0.482 | 0.507 |
| ECE Platt-scaled | 0.077 | 0.174 | 0.208 |
| ECE Elkan-Noto | 0.097 | 0.173 | 0.208 |
| Elkan-Noto c | 0.728 | 0.983 | 0.991 |

**PULSNAR Longformer is the only deployable model with raw probabilities.** BioClinicalBERT requires post-hoc Platt scaling before any threshold-sensitive use. BERT's c estimates collapse to ~1.0 because raw probabilities concentrate near 1 — the c estimate is no longer informative as a labelling-frequency estimate. Longformer's c = 0.728 implies ~27% undercoding rate, consistent with PCL-5 inpatient prevalence findings (Stanley et al. 2020).

### Attribution (Integrated Gradients, 50 high-confidence true positives)

Per-section share:

| Section | PULSNAR LF | BERT truncated | BERT chunk-pool top window |
|---|---:|---:|---:|
| HPI | 43.2 % | 58.2 % | 53.5 % |
| PMH | 22.4 % | 27.4 % | 27.3 % |
| BHC | **32.0 %** | 13.0 % | 17.3 % |
| Social History | 1.3 % | 0.7 % | 1.2 % |

**Longformer puts ~32% of attribution on Brief Hospital Course; BERT-truncated puts only 13% because BHC literally falls past the 512-token window.** Chunk-pool partially recovers (17%). This is a clean architectural finding consistent with Li et al. (2022).

Top words (PULSNAR Longformer): bipolar, narcotic, illness, arrested, delayed, pancreatitis, schizoaffective, psychosis, anemia, assault. Notably trauma-anchored.
Top words (BERT, both modes): psych, anxiety, bipolar, psychiatric, disorder, dilaudid, overdose, suicide, abuse, methadone. Notably more comorbidity-anchored, less trauma-anchored.

**No label-leakage tokens** in any model's top attributions — the universal masking worked.

### Specificity check

A separate Longformer trained PTSD+ vs. age/sex 1:1-matched MDD/anxiety controls (standard cross-entropy) reaches:

| Metric | PULSNAR Longformer | Specificity-trained Longformer |
|---|---:|---:|
| Test AUPRC | 0.885 | **0.911** |
| Test AUROC | 0.890 | 0.815 |
| Sensitivity | 0.846 | 0.852 |
| Specificity | 0.745 | 0.581 |
| Mean predicted prob on proxy | 0.46 | 0.34 |

**PTSD-specific signal is recoverable above-and-beyond generic "psychiatric admission" language**, ruling out the worst-case interpretation that the primary model is a psych-vs-non-psych classifier. The AUROC drop reflects the harder task — psychiatric controls have overlapping vocabulary.

### Temporal generalization

| Scenario | Test AUPRC |
|---|---:|
| Random-split model on random test | 0.888 |
| Random-split model on temporal test (2017–2019) | 0.886 |
| **Temporal-trained model on temporal test** | 0.842 |

Temporal training **hurts** generalization. The random-split model loses only 0.002 AUPRC when tested on 2017–2019 (the random distribution is already representative of late MIMIC-IV); the temporal model loses 0.044 because it has 25% less training data. **Random split recommended for deployment.**

### Cross-model agreement + ensemble

- LF-PULSNAR ⇄ BERT chunk-pool: **86.85 % agreement, κ = 0.737, top-quintile overlap 83.5 %**, McNemar p = 0.107 (n.s.).
- LF-PULSNAR ⇄ BERT truncated: 85.88 % agreement, κ = 0.718.
- BERT trunc ⇄ chunk-pool: 88.72 %, κ = 0.774, Pearson r = 0.86 (very correlated — same model, different aggregation).
- LF-PULSNAR ⇄ Structured: 56 %, κ = 0.11. Effectively independent rankings.
- LF-PULSNAR ⇄ Keyword: 51 %, κ = 0.00.
- **Ensemble probe.** Max-pool over rank-normalised probabilities of all 5 models reaches AUPRC 0.751; mean-pool 0.874. **Neither beats the best individual model (0.885)**. The structured + keyword noise dominates max-pool; mean-pool is competitive but does not lift. **No ensemble lift was found across the deployed lineup.**

---

## Compute cost

### Training (measured)

| Model | Wall-clock (s) | GPU-h | Hardware | Train n | Test AUPRC |
|---|---:|---:|---|---:|---:|
| PULSNAR Longformer | 12,617 | 3.50 | L40S (gpu2708) | 11,837 | 0.8848 |
| BioClinicalBERT (PU) | 791 | 0.22 | RTX 3090 (gpu2105) | 11,837 | 0.8576 / 0.8775 (trunc / chunk-pool) |
| Structured + LogReg | 68 | 0 (CPU) | node2302 | 9,649 | 0.6833 |
| Keyword (DSM-5/PCL-5) | 0 (no training) | 0 | n/a | n/a | 0.5096 |

### Inference (1,551-patient test set on identical L40S GPU, except CPU baselines)

| Model | Hardware | Test wall (s) | ms/patient | Peak mem |
|---|---|---:|---:|---:|
| PULSNAR Longformer (4,096 tok) | L40S | 124.6 | **80.4** | 2.03 GB GPU |
| BERT chunk-pool (512 × 256) | L40S | 35.1 | 22.7 | 0.65 GB GPU |
| BERT truncated (512) | L40S | 4.6 | **2.97** | 0.83 GB GPU |
| Structured + LogReg | 16-CPU batch | 27.7 | 17.9 (I/O) | 0.30 GB RAM |
| Keyword | 16-CPU batch | 0.52 | **0.34** | 0.17 GB RAM |

Architecture (4,096 vs. 512 attention) — not GPU generation — drives the gap. PULSNAR Longformer pays ~16× hardware-normalized training cost and 3.55× inference latency vs. BERT chunk-pool, for **+0.007 AUPRC, McNemar p = 0.11 (n.s.), much better calibration, and a small EO improvement**.

---

## What the final report should say

1. **Primary claim:** "The Clinical Longformer model, fine-tuned with PULSNAR (SAR-aware) propensity-weighted nnPU loss at α = 0.196 on a section-filtered, label-masked MIMIC-IV discharge corpus, achieves test AUPRC = 0.885 and AUROC = 0.890 with val-derived thresholding."

2. **Non-circular validation:** "Pharmacological proxy validation (n = 102 patients with prazosin + SSRI/SNRI history but no ICD PTSD code) shows Mann-Whitney AUC = 0.770 (p = 3.8e-18); proxy patients score ~6× the median probability of demographically-matched unlabeled controls. The model — never shown a single proxy patient during training — assigns substantially higher probability to patients whose pharmacotherapy is consistent with PTSD treatment."

3. **Specificity:** "A separate Longformer trained PTSD+ vs. age/sex-matched MDD/anxiety controls reaches AUPRC = 0.91, ruling out the worst-case interpretation that the primary model is a generic psych-vs-non-psych classifier."

4. **Architecture comparison:** "BioClinicalBERT chunk-pool reaches AUPRC = 0.878, statistically indistinguishable from the Longformer on McNemar's test (p = 0.107). Long-context inference, not architecture per se, drives most of the lift over BERT-truncated. The Longformer's advantage is concentrated in calibration (5–6× lower raw ECE), in attribution shift toward trauma narrative (HPI 43% vs BERT 53–58%, BHC 32% vs BERT 13–17%), and in a small fairness improvement (sex EO 0.114 vs BERT 0.127–0.151)."

5. **Clinical utility:** "At 2% inpatient deployment prevalence, PULSNAR Longformer NNS = 15.8 and PPV = 6.3%; LR+ = 3.32 provides moderate positive evidence appropriate for screening (not diagnosis). NPV > 0.99 makes a negative screen highly reliable. BioClinicalBERT chunk-pool's NNS at the same prevalence is 14.2 (better) but requires post-hoc Platt calibration before deployment because raw ECE = 0.51."

6. **Limitations:**
   - Subgroup performance gap: men and older patients under-detected (sex EO 0.114, age EO 0.211), inherited from non-SCAR PTSD coding bias and not fully corrected by PULSNAR's propensity reweighting.
   - Raw PU labels are used for evaluation; true population AUPRC is likely higher than reported.
   - Temporal generalization tested on MIMIC-IV 2017–2019 only; external validation not performed.
   - BioClinicalBERT requires post-hoc calibration before deployment (raw ECE 0.48–0.51).
   - Proxy validation set is small (n = 102) with estimated 15–20% FPR.

---

## Artifacts

| File | Contents |
|------|---------|
| `models/ewang163_longformer_pulsnar/` (also `models/ewang163_longformer_best/` symlink) | Final deployment checkpoint |
| `models/ewang163_longformer_best_temporal/` | Pre-2015 train; not recommended for deployment |
| `models/ewang163_bioclinbert_best/` | Comparison BERT (both inference modes) |
| `models/ewang163_specificity_longformer_best/` | Specificity check vs. MDD/anxiety |
| `models/ewang163_structured_logreg.pkl` + `_features.json` | Structured baseline |
| `models/ewang163_keyword_weights.json` | DSM-5/PCL-5 lexicon (62 patterns) |
| `models/ewang163_platt_calibrator.pkl` | Post-hoc probability calibrator |
| **PULSNAR Longformer outputs:** | |
| `results/predictions/ewang163_longformer_{val,test}_predictions_pulsnar.csv` | Per-patient predicted probs |
| `results/metrics/ewang163_evaluation_results_pulsnar.json` | Full utility + thresholds |
| `results/metrics/ewang163_calibration_results_pulsnar.csv` | Raw / Platt / Elkan-Noto |
| `results/metrics/ewang163_fairness_results_pulsnar.csv` | Per-subgroup fairness |
| `results/figures/ewang163_calibration_curve_pulsnar.png` | 3-panel calibration plot |
| `results/attribution/ewang163_top_attributed_words_v2_pulsnar.csv` | IG word-level |
| `results/attribution/ewang163_attribution_by_section_v2_pulsnar.csv` | IG per section |
| `results/metrics/ewang163_proxy_validation_results.csv` | Proxy MW AUC (Longformer) |
| **BioClinicalBERT outputs (both modes):** | |
| `results/predictions/ewang163_bioclinbert_{trunc,chunkpool}_{val,test}_predictions.csv` | Per-patient probs |
| `results/metrics/ewang163_bioclinbert_full_eval_summary.json` | Top-line per-mode summary |
| `results/metrics/ewang163_calibration_results_bert_{trunc,chunkpool}.csv` | Calibration |
| `results/metrics/ewang163_dca_results_bert_{trunc,chunkpool}.csv` | Decision curves |
| `results/metrics/ewang163_fairness_results_bert_{trunc,chunkpool}.csv` | Fairness |
| `results/metrics/ewang163_subgroup_auprc_bert_{trunc,chunkpool}.csv` | Subgroup AUPRC |
| `results/metrics/ewang163_proxy_validation_bert_{trunc,chunkpool}.csv` | Proxy MW AUC |
| `results/metrics/ewang163_ablation_results_bert_{trunc,chunkpool}.csv` | A1 + A2 |
| `results/error_analysis/ewang163_bert_{trunc,chunkpool}_error_summary.csv` | FP/FN aggregate |
| `results/attribution/ewang163_attribution_by_section_bert_{trunc,chunkpool}.csv` | IG per section |
| `results/attribution/ewang163_top_attributed_words_bert_{trunc,chunkpool}.csv` | IG word-level |
| `results/figures/ewang163_calibration_curve_bert_{trunc,chunkpool}.png` | Calibration plots |
| `results/figures/ewang163_dca_{2,5}pct_bert_{trunc,chunkpool}.png` | DCA plots |
| **Cross-model:** | |
| `results/metrics/ewang163_cross_model_mcnemar.csv` | All-pairs McNemar + κ + Pearson |
| `results/metrics/ewang163_cross_model_topquintile_overlap.csv` | Rank-overlap |
| `results/metrics/ewang163_subgroup_nns_by_model.csv` | Per-subgroup NNS, all models |
| `results/metrics/ewang163_ensemble_results.json` | Max/mean-pool ensemble |
| **Specificity check:** | |
| `results/metrics/ewang163_specificity_eval_results.json` | PTSD vs. MDD/anxiety controls |
| **Temporal generalization:** | |
| `results/metrics/ewang163_temporal_eval_results.json` | Random vs. temporal split eval |

---

## Sign-off

Selected primary deployment model: **Clinical Longformer + PULSNAR (α = 0.1957), section-filtered + PTSD-string-masked corpus.**

All comparison models (BioClinicalBERT trunc + chunk-pool, structured, keyword) have been evaluated symmetrically on the same downstream pipeline (calibration, DCA, fairness, subgroup, proxy, ablations, error analysis, IG attribution where applicable). All-pairs McNemar, agreement matrix, ensemble probe completed.

**No further model selection work is planned for this submission.**
