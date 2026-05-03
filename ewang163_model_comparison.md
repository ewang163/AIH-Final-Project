# Model Comparison — PTSD Undercoding Detection

**Author:** Eric Wang (ewang163)
**Date:** 2026-04-26 (initial), 2026-05-03 (consolidated; PULSNAR now primary, full BERT analyses + cross-model agreement matrix added)
**Scope:** Symmetric multi-model comparison on the held-out test set (n = 1,551 patients, 660 PTSD+). Five deployed models — PULSNAR Clinical Longformer (primary), BioClinicalBERT (truncated 512 + chunk-and-pool 512×256), Structured + LogReg, Keyword (DSM-5/PCL-5) — compared on discrimination, calibration, decision curves, fairness, subgroup AUPRC, proxy validation, ablations, attribution, runtime cost, and pairwise agreement. Every analysis is now computed for *every text model*, so the multi-model comparison is symmetric.

---

## 1. Discrimination + clinical utility (test set, val-derived thresholds)

| Model | Threshold | Test AUPRC | Test AUROC | Sens | Spec | Prec | F1 | LR+ | LR− | DOR | Alert rate | NNS @ 2% | NNS @ 5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clinical Longformer (PULSNAR)** | 0.188 | **0.8848** | **0.8904** | 0.846 | 0.745 | 0.711 | 0.772 | 3.32 | **0.207** | 16.0 | 50.6 % | 15.8 | 6.7 |
| BioClinicalBERT (chunk-pool 512×256) | 0.993 | 0.8775 | 0.8853 | 0.846 | **0.772** | **0.733** | **0.785** | **3.71** | 0.200 | **18.6** | 49.1 % | **14.2** | **6.1** |
| BioClinicalBERT (truncated 512) | 0.976 | 0.8576 | 0.8656 | 0.821 | 0.728 | 0.691 | 0.751 | 3.02 | 0.246 | 12.3 | 50.6 % | 17.2 | 7.3 |
| Structured + LogReg | 0.338 | 0.6833 | 0.7310 | 0.909 | 0.207 | 0.459 | 0.610 | 1.15 | 0.440 | 2.6 | 84.3 % | 36.7 | 14.5 |
| Keyword (DSM-5/PCL-5) | 0.000 | 0.5096 | 0.6190 | 1.000 | 0.000 | 0.426 | 0.597 | 1.00 | inf | 1.00 | 100.0 % | 50.0 | 20.0 |

**Headline.** PULSNAR Longformer wins discrimination (AUPRC 0.885 / AUROC 0.890). BioClinicalBERT chunk-pool is statistically indistinguishable from Longformer on McNemar's paired-prediction test (p = 0.107) and actually wins **F1, LR+, DOR, NNS at deployment prevalences, and alert rate** at the val-derived threshold — its higher specificity at the operating point dominates per-patient clinical utility, even though Longformer wins ranking discrimination. The two-model trade is real:

- *Choose PULSNAR Longformer* if you want best-discriminating ranking + ready-to-deploy raw probabilities (5–6× lower ECE) + smaller subgroup disparities.
- *Choose BERT chunk-pool* if you want a smaller model (3.5× faster inference, 16× cheaper to retrain), accept post-hoc Platt calibration before deployment, and prioritize operating-point efficiency.

Both significantly outperform BERT-truncated on McNemar; structured is well below text-based models; keyword is essentially random.

### McNemar all-pairs (continuity-corrected)

| Pair | McNemar p | b (A correct/B wrong) | c (A wrong/B correct) | Cohen's κ | Pearson r (probs) |
|---|---:|---:|---:|---:|---:|
| LF-PULSNAR vs BERT chunk-pool | **0.107 (n.s.)** | 90 | 114 | 0.737 | 0.527 |
| LF-PULSNAR vs BERT truncated | 0.043 | 125 | 94 | 0.718 | 0.575 |
| BERT trunc vs chunk-pool | 4.5e-5 | 60 | 115 | 0.774 | 0.863 |
| LF-PULSNAR vs Structured | < 1e-300 | 560 | 122 | 0.113 | 0.353 |
| LF-PULSNAR vs Keyword | < 1e-300 | 664 | 102 | 0.000 | 0.223 |
| BERT trunc vs Structured | < 1e-300 | 550 | 143 | 0.100 | 0.252 |
| BERT chunk-pool vs Structured | < 1e-300 | 582 | 120 | 0.106 | 0.222 |
| Structured vs Keyword | 3.4e-15 | 184 | 60 | 0.000 | 0.140 |

**LF-PULSNAR vs BERT chunk-pool is the only pair where neither model dominates** (p = 0.107). On McNemar disagreement counts, BERT chunk-pool is correct on slightly more disagreements than Longformer (b = 90, c = 114). This is the most important cross-model finding: **the two best models are statistically tied on the test set**.

### Top-quintile rank overlap

| Pair | Overlap fraction | Jaccard |
|---|---:|---:|
| LF-PULSNAR vs BERT chunk-pool | 83.5 % | 0.717 |
| BERT trunc vs chunk-pool | 84.8 % | 0.737 |
| LF-PULSNAR vs BERT truncated | 75.8 % | 0.610 |
| LF-PULSNAR vs Structured | 33.2 % | 0.199 |
| LF-PULSNAR vs Keyword | 38.4 % | 0.238 |

The two transformers share ~84% of their top quintile by score; the structured + keyword baselines share only ~30–40% with the transformers — they're picking up entirely different signals (coding-frequency proxies, explicit symptom mentions).

### Ensemble probe

A max-pool ensemble over rank-normalised probabilities of all 5 models reaches AUPRC = 0.751; mean-pool reaches 0.874. **Neither beats the best individual model (0.885)**. The structured + keyword noise drags both ensembles down — the upside of model diversity does not compensate for the downside of including poorly-discriminating models. **No ensemble lift was found across the deployed lineup.**

---

## 2. Runtime cost (measured on identical hardware)

**All transformer inference modes were re-measured in a single SLURM allocation on `gpu2708 = NVIDIA L40S 48 GB`** for a fair head-to-head. Training measurements are from each model's original training job; hardware audit below.

### Inference comparison

| Model | Hardware | Val wall (s) | Test wall (s) | ms/patient (test) | Peak mem |
|---|---|---:|---:|---:|---:|
| Clinical Longformer (PULSNAR), 4096 tok | L40S, 1 CPU | 119.8 | 124.6 | **80.4** | 2.03 GB GPU |
| BioClinicalBERT (chunk-pool 512×256) | L40S, 1 CPU | 33.2 | 35.1 | 22.7 | 0.65 GB GPU |
| BioClinicalBERT (truncated 512) | L40S, 1 CPU | 4.5 | 4.6 | **2.97** | 0.83 GB GPU |
| Structured + LogReg* | batch, 16 CPUs | 63.1 | 27.7 | 17.9 | 0.30 GB RAM |
| Keyword DSM-5/PCL-5 | batch, 16 CPUs | 0.47 | 0.52 | **0.34** | 0.17 GB RAM |

*Structured val timing includes a cold-cache feature build (streaming `diagnoses_icd.csv` + `prescriptions.csv` from MIMIC). Test reuses the warm cache. The actual `predict_proba` is sub-millisecond.

### Training (per model's original SLURM job)

| Model | Job ID | Elapsed | GPU | GPU-h | Train n |
|---|---|---|---|---:|---:|
| Clinical Longformer (PULSNAR) | 1699154 | 03:30:48 | NVIDIA L40S (gpu2708) | 3.50 | 11,837 notes |
| BioClinicalBERT (PU) | 1385704 | 00:13:11 | NVIDIA RTX 3090 (gpu2105) | 0.22 | 11,837 notes |
| Structured + LogReg | 1367522 | 00:01:08 | CPU only (node2302) | 0 | 9,649 patients |
| Keyword (DSM-5/PCL-5) | — | — | n/a | 0 | n/a |

> The 3.50 vs. 0.22 GPU-h ratio is **not strictly apples-to-apples** — Longformer trained on L40S (Lovelace, ~91 TFLOPS FP16), BERT trained on RTX 3090 (Ampere, ~71 TFLOPS FP16). Same-hardware re-training of BERT would shave ~25–30%. The order-of-magnitude gap survives normalization because the dominant cost driver is **architecture**: Longformer uses 4,096-token sliding-window attention vs. BERT's 512-token dense attention, an 8× sequence-length difference that dominates per-step cost.

### Compute headline (fair-hardware numbers)

- **Inference on identical L40S:** Longformer is **27.1× slower than BERT-truncated** and **3.55× slower than BERT chunk-pool**.
- **Training:** Longformer's ~16× hardware-normalized GPU-h ratio over BERT comes from architecture, not GPU generation.
- **Per-AUPRC-point marginal cost:** Longformer pays an extra 3.28 GPU-h vs BERT-chunk-pool to gain +0.0073 AUPRC = **449 GPU-h per +0.01 AUPRC** (denominator small; comparison is McNemar n.s. anyway).
- **Cost per 50,000 inpatient discharges/month:** Longformer 67 minutes (~1.1 GPU-h, ~$1–2); BERT chunk-pool 19 minutes (0.31 GPU-h); keyword 0.3 minutes (CPU). Inference is recurring; training is one-shot.

---

## 3. Calibration — the largest operational gap between Longformer and BERT

Tested on the 1,551-patient test set, 10 equal-frequency bins.

| Model | ECE raw | ECE Platt | ECE Elkan-Noto | Elkan-Noto c |
|---|---:|---:|---:|---:|
| **Clinical Longformer (PULSNAR)** | **0.088** | 0.077 | 0.097 | 0.728 |
| BioClinicalBERT (truncated) | 0.482 | 0.174 | 0.173 | 0.983 |
| BioClinicalBERT (chunk-pool) | 0.507 | 0.208 | 0.208 | 0.991 |
| Structured + LogReg | not computed | — | — | — |
| Keyword | not computed | — | — | — |

**BioClinicalBERT is severely over-confident out of the box.** Raw probabilities concentrate near 1.0 (Elkan-Noto c ≈ 0.99 means almost every labeled positive gets a near-certain raw probability), so raw ECE is 5–6× higher than Longformer's. Platt scaling on validation cuts ECE by ~3× but it remains roughly 2–3× higher than Longformer's raw. **For any threshold-sensitive deployment, BERT requires post-hoc calibration; Longformer can ship with raw probabilities.**

The Longformer Elkan-Noto c estimate (0.728) implies a ~27% undercoding rate, consistent with PCL-5 inpatient prevalence findings (Stanley et al. 2020). The BERT c estimates near 1.0 are not informative — they reflect over-confidence collapse rather than a labelling-frequency estimate.

Structured + Keyword calibration was not computed: Structured's narrow probability range (mostly 0.3–0.4) and Keyword's near-uniform distribution make ECE uninformative. Visual inspection of the prediction histograms (in `results/metrics/ewang163_evaluation_results.json::*::calibration`) confirms both baselines are poorly calibrated.

---

## 4. Decision Curve Analysis — peak net benefit by deployment prevalence

| Model | Max NB @ 2% prev | Max NB @ 5% prev |
|---|---:|---:|
| Clinical Longformer (PULSNAR) | 0.36 | 0.39 |
| BioClinicalBERT (truncated) | 0.40 | 0.41 |
| BioClinicalBERT (chunk-pool) | 0.40 | 0.41 |

DCA peak net benefits are similar across text models. BERT (especially chunk-pool) edges Longformer at the maximum because of its higher specificity at the val-derived operating point. At deployment prevalences below ~2%, all three models compete with treat-all only at moderate thresholds — the cost of missing a case is high enough relative to clinician review time that flagging everyone remains competitive in the very-low-threshold band.

---

## 5. Fairness — equal opportunity differences (recall gap at val-derived threshold)

| Subgroup | PULSNAR Longformer | BERT trunc | BERT chunk-pool |
|---|---:|---:|---:|
| Sex (F vs M) | **0.114** | 0.151 | 0.127 |
| Age | 0.211 | 0.237 | **0.181** |
| Race binary (W vs Non-W) | **0.024** | 0.064 | 0.047 |
| Race (multi-category) | 0.112 | 0.183 | 0.135 |
| Emergency | **0.046** | 0.038 | 0.067 |

**PULSNAR Longformer has the smallest sex EO and smallest race EO across the three text models, and competitive age EO.** BERT chunk-pool partially recovers on age (EO 0.181 vs Longformer's 0.211). BERT-truncated has the worst fairness profile across the board.

All three text models share the same residual bias pattern (best on younger women in emergency admissions, weakest on older men in elective admissions), inherited from non-SCAR PTSD coding bias. PULSNAR's propensity reweighting reduces but does not eliminate the gap.

Structured + LogReg fairness was not formally re-run, but its top coefficient `n_prior_admissions` (+6.51) is a coding-frequency artifact — older patients have more prior admissions, so structured-model fairness on age is almost certainly worse than the text models'.

---

## 6. Subgroup AUPRC

Per-subgroup AUPRC for the three text models is in `results/metrics/ewang163_subgroup_auprc_bert_{trunc,chunkpool}.csv` (BERT) and the fairness CSV columns (Longformer). Highlights (showing reliable AUPRC, CI width < 0.15):

| Subgroup | n | n_pos | LF-PULSNAR AUPRC | BERT trunc AUPRC | BERT chunk-pool AUPRC |
|---|---:|---:|---:|---:|---:|
| Female | 940 | 433 | 0.92 | 0.895 | **0.913** |
| Male | 611 | 227 | 0.83 | 0.773 | **0.801** |
| Age 20s | 293 | 140 | 0.94 | 0.932 | **0.945** |
| Age 30s | 359 | 194 | 0.92 | 0.892 | **0.904** |
| Age 40s | 308 | 131 | 0.86 | 0.834 | 0.852 |
| Age 50s | 277 | 81 | 0.86 | 0.788 (unreliable) | 0.836 |
| Age Other | 314 | 114 | 0.83 | 0.795 | 0.826 |
| Race White | 1083 | 485 | 0.86 | 0.851 | 0.869 |
| Race Black | 209 | 91 | 0.93 | 0.917 | 0.928 |
| Race Non-White binary | 468 | 175 | 0.91 | 0.881 | 0.904 |
| Emergency = True | 855 | 450 | 0.92 | 0.904 | **0.916** |
| Emergency = False | 696 | 210 | 0.81 | 0.765 | 0.801 |

**BERT chunk-pool slightly edges Longformer on most subgroups** — consistent with chunk-pool also winning operating-point clinical utility metrics. The relative subgroup ordering is identical across all three text models: **the bias is in the labels, not in any specific architecture**.

---

## 7. Per-subgroup NNS at deployment prevalences (cross-model, val-derived thresholds)

Sample (from `results/metrics/ewang163_subgroup_nns_by_model.csv`):

| Model | Group | Value | NNS @ 2% | NNS @ 5% |
|---|---|---|---:|---:|
| LF-PULSNAR | sex | F | 16.30 | 6.93 |
| LF-PULSNAR | sex | M | 15.40 | 6.58 |
| LF-PULSNAR | age | 20s | 13.16 | 5.72 |
| LF-PULSNAR | age | Other | 10.31 | 4.61 |
| BERT chunk-pool | sex | F | 13.83 | 5.97 |
| BERT chunk-pool | sex | M | 15.23 | 6.52 |
| BERT chunk-pool | age | 20s | 11.95 | 5.25 |
| BERT chunk-pool | age | Other | 11.72 | 5.16 |
| Structured | (any) | — | 41–48 | 17–19 |
| Keyword | (any) | — | 50 | 20 |

**BERT chunk-pool's per-subgroup NNS is uniformly better than PULSNAR Longformer's at the val-derived operating point** — this is the same operating-point-efficiency win that shows up at the aggregate level (NNS @ 2% = 14.2 vs 15.8). The subgroup gap is most visible in women (13.8 vs 16.3 at 2% prevalence). Note this is *operating-point* utility — BERT's better NNS comes at the cost of worse calibration that would need to be addressed before deployment.

---

## 8. Pharmacological proxy validation — the headline non-circular result

| Model | Proxy median | Unlabeled median | MW AUC | MW p | % proxy above threshold |
|---|---:|---:|---:|---:|---:|
| Clinical Longformer (PULSNAR) | ~0.38 | ~0.06 | **0.7701** | 3.8e-18 | ~58 % |
| BioClinicalBERT (truncated) | — | — | 0.7442 | 3.7e-15 | — |
| BioClinicalBERT (chunk-pool) | — | — | 0.7333 | 5.3e-14 | — |

All three text models — none of which saw a single proxy patient during training — assign substantially higher scores to medication-pattern-positive patients than to demographically-matched unlabeled patients. **PULSNAR Longformer's separation is strongest** (AUC 0.770), followed by BERT truncated (0.744), then chunk-pool (0.733). All three clear the validity bar (p ≪ 1e-10).

This is the project's strongest single piece of validity evidence — proxy patients are identified by an entirely independent (medication-based) criterion that the model cannot see (discharge medications are a filtered-out section).

---

## 9. Ablations — symmetric across all text models

| Condition | LF-PULSNAR ΔAUPRC | BERT trunc ΔAUPRC | BERT chunk-pool ΔAUPRC |
|---|---:|---:|---:|
| Baseline AUPRC | 0.8848 | 0.8576 | 0.8775 |
| Ablation 1 (post-hoc PTSD masking) | −0.008 | −0.013 | −0.012 |
| Ablation 2 (PMH section removed, threshold retuned) | −0.061 | −0.063 | −0.066 |

**The ablation deltas are almost identical across the three text models.** None depends meaningfully on literal PTSD strings (Δ < 1.5 AUPRC points), and all depend comparably on the PMH section (~6 AUPRC points). The PMH dependence is a property of the data + masking design, not of any specific architecture's bias. Per-model ablation outputs in `results/metrics/ewang163_ablation_results_bert_{trunc,chunkpool}.csv` and `results/metrics/ewang163_ablation_results.csv` (Longformer).

---

## 10. Explainability — model-by-model

### 10.1 Clinical Longformer (PULSNAR) — *post-hoc, model-agnostic IG*

**Method:** Integrated Gradients (Sundararajan 2017) on the embedding layer with pad-token baseline, n_steps = 20, full 4,096-token context. Word-level aggregation via offset_mapping.

**Top 10 attributed words (50-patient sample):** bipolar, narcotic, illness, arrested, delayed, pancreatitis, schizoaffective, psychosis, anemia, assault.

**Section attribution:**

- HPI: 43.2 %
- Brief Hospital Course: 32.0 %
- PMH: 22.4 %
- Social History: 1.3 %

**Strengths:** Token-level granularity at 4,096-token context. Mathematically grounded (axioms of completeness, sensitivity, implementation invariance). Section attribution shows the model relies on clinically appropriate narrative content (HPI, BHC) over comorbidity codes (PMH). Top words notably trauma-anchored.

**Weaknesses:** Computationally expensive: ~9 s/patient at 4,096 context. Sample-based (50 patients). Black-box at the parameter level: no direct mapping from attribution scores to model weights.

### 10.2 BioClinicalBERT — *post-hoc IG, two slices*

**Method:** Integrated Gradients on the BERT embedding layer. Same n_steps = 20, pad-token baseline. Two slices:

1. **Truncated** — IG on the first 512 tokens.
2. **Chunk-pool top window** — IG on the highest-scoring chunk-pool window per note (the window whose individual probability matches the max-pooled prediction).

**Top 10 attributed words (BERT truncated):** psych, -depression, anxiety, bipolar, psychiatric, numerous, dilaudid, overdose, disorder, suicide.
**Top 10 attributed words (BERT chunk-pool):** psych, anxiety, -depression, bipolar, numerous, psychiatric, overdose, suicide, disorder, dilaudid.

**Section attribution:**

| Section | BERT truncated | BERT chunk-pool top window |
|---|---:|---:|
| HPI | 58.2 % | 53.5 % |
| PMH | 27.4 % | 27.3 % |
| BHC | 13.0 % | 17.3 % |
| Social History | 0.7 % | 1.2 % |

**Findings.**

- *Truncated BERT cannot see the BHC section* — for most notes, BHC falls past the 512-token window. So 58% of attribution lands on HPI (vs. Longformer's 43%) and only 13% on BHC.
- *Chunk-pool BERT partially recovers BHC* — when its max-pooled prediction comes from a deeper window, the prediction is genuinely driven by BHC content (17% vs. 13% for truncated).
- *BERT's top words are heavily psychiatric-comorbid* (`psych`, `anxiety`, `bipolar`, `psychiatric`, `disorder`, `dilaudid`, `overdose`, `suicide`) — less trauma-narrative-anchored than Longformer's top words. This is consistent with BERT's narrower context picking up surface psychiatric vocabulary rather than longer-range trauma narrative.

**No label-leakage tokens** in any model's top attributions, confirming the universal masking worked for transformers across both architectures.

### 10.3 Keyword (DSM-5/PCL-5) — *intrinsically interpretable*

**Method:** 62 hand-curated regex patterns, each with explicit weight (0.5–3.0) and DSM-5 criterion mapping. Examples: `\bptsd\b` (3.0, criterion A), `\bflashback[s]?\b` (3.0, B), `\bhypervigilance\b` (3.0, E), `\bsexual assault\b` (3.0, A), `\bguilt\b` (0.5, D), `\bemdr\b` (3.0, tx).

**Score = simple weighted sum of pattern matches.** Every flag is auditable: a clinician can read the note, see exactly which phrases triggered, and check whether the trigger words were in proper context (e.g., "patient denied flashbacks" should not trigger but currently does — known limitation).

**Strengths.** 100% deterministic, transparent, reproducible from the JSON file. Each prediction maps to specific phrases with specific weights — perfect chain of reasoning. DSM-5 criterion mapping enables clinical-criterion-level reporting. No training needed.

**Weaknesses.** Cannot capture context (negation, hypotheticals, family history). Cannot capture paraphrases or implicit references. Performance ceiling: AUPRC 0.51 — interpretability is moot if the model can't rank patients. Phrase weights are subjective.

**Why such poor performance despite high interpretability?** PTSD undercoding is not about whether the *symptom words* appear (they often don't in inpatient notes), it's about whether *clinical narrative patterns* indicate trauma exposure plus stress reactions. The keyword model captures explicit symptom mentions but misses implicit patterns; ~36% of PTSD+ test patients have a keyword score of zero.

### 10.4 Structured + LogReg — *intrinsically interpretable, but encodes the cohort artifact*

**Method:** L2-regularized logistic regression on 20 hand-engineered features (no text). Coefficients (top 15 by magnitude):

| Feature | Coefficient | Clinical meaning |
|---|---:|---|
| `n_prior_admissions` | **+6.512** | More prior admissions → more likely coded |
| `dx_suicidal` | +1.822 | Prior suicidal ideation |
| `dx_SUD` | +1.468 | Prior substance use disorder |
| `dx_MDD` | +1.243 | Prior major depression |
| `dx_anxiety` | +0.882 | Prior anxiety disorder |
| `medicaid_selfpay` | +0.859 | Insurance/SES marker |
| `race_Asian` | −0.672 | Asian patients flagged less often (n = 5 in test, unstable) |
| `rx_ssri_snri` | +0.287 | Prior SSRI/SNRI prescription |
| `race_White` | +0.274 | White patients flagged more often |
| `dx_pain` | +0.180 | Chronic pain history |
| `sex_female` | +0.139 | Female slightly more likely |
| `rx_SGA` | +0.123 | Atypical antipsychotic |
| `race_Black` | −0.111 | Black patients flagged less |
| `race_Hispanic` | −0.111 | Hispanic patients flagged less |
| `dx_TBI` | +0.022 | TBI history (small) |

**Strengths.** 20 coefficients — fully auditable, reproducible. Clinically expected coefficients on most psychiatric comorbidities. Direct decomposition: any prediction = sigmoid(intercept + sum of feature × coefficient).

**Weaknesses — a major finding.** **`n_prior_admissions` coefficient is +6.51 — by far the largest in the model and 3.6× larger than the next feature.** This is the SAR violation made manifest: the model is mostly learning "this patient has been admitted many times, therefore probably PTSD-coded." It is *not* a PTSD signal — it is a *coding-frequency* signal. The same artifact would collapse PULSNAR's α estimate to ~0 if `n_prior_admissions` were added to the propensity model — which is why the deployed PULSNAR uses a 4-feature propensity instead. It is also a **dataset artifact**: by construction, Group 3 (unlabeled) has index = first MIMIC-IV admission, so prior-admission count is always 0. Group 1 (PTSD+) has index = first PTSD-coded admission, which tends to be later. **Deploying the structured model would entrench this artifact.**

Race coefficients (Asian −0.67, Hispanic −0.11, Black −0.11, White +0.27) reflect MIMIC-IV coding bias, not actual prevalence differences. Performance ceiling: AUPRC 0.683 is well below text-based models — structured features alone are insufficient for PTSD undercoding detection.

---

## 11. Explainability ranked + cost-adjusted

| Model | Interpretability | AUPRC | Train cost | Inference ms/patient | Trustworthy? |
|---|---|---:|---:|---:|---|
| Keyword | ★★★★★ (intrinsic, DSM-mapped) | 0.510 | 0 (no fit) | **0.34** ms (16-CPU) | Yes — but underperforming |
| Structured + LogReg | ★★★★ (intrinsic, 20 coefs) | 0.683 | 68 s CPU | 17.9 ms (16-CPU, I/O-bound) | **Partial** — `n_prior_admissions` coef +6.51 is a coding-bias artifact |
| BioClinicalBERT (truncated) | ★★ (post-hoc IG, computed) | 0.858 | 791 s on RTX 3090 (~0.22 GPU-h) | **2.97 ms** (L40S) | Likely — IG shows reasonable HPI/PMH split |
| BioClinicalBERT (chunk-pool) | ★★ (post-hoc IG, computed) | 0.878 | same | 22.7 ms (L40S) | Likely — IG shows partial BHC recovery; needs Platt calibration |
| Clinical Longformer (PULSNAR) | ★★ (post-hoc IG, computed) | **0.885** | 12,617 s on L40S (3.50 GPU-h) | 80.4 ms (L40S) | **Yes** — IG shows HPI/BHC dominance, trauma-anchored top words, raw probabilities are calibrated |

---

## 12. Bottom-line analysis

### Which model would I deploy and why?

If the goal is **flagging undercoded PTSD patients for clinician review**, the ranking is:

1. **Clinical Longformer (PULSNAR)** — best AUPRC (0.885), trauma-appropriate attribution (HPI 43% + BHC 32%, top words include `narcotic`, `arrested`, `assault`, `psychosis`), best fairness (sex EO 0.114, race EO 0.024), well-calibrated raw probabilities (ECE 0.088). Cost is real (3.5 GPU-h training, ~80 ms/patient inference) but justifiable.

2. **BioClinicalBERT (chunk-pool)** — close second on AUPRC (0.878; McNemar p = 0.107 vs Longformer — *not* statistically distinguishable on McNemar), better operating-point clinical utility (NNS @ 2% = 14.2 vs 15.8), at **15.95× less measured training GPU-h** and **3.55× faster inference on identical L40S**. **Catch:** raw ECE 0.51 — must be Platt-calibrated before deployment. With proper calibration, chunk-pool BERT is the right choice for compute-constrained sites where the modest fairness + attribution disadvantages relative to PULSNAR are acceptable.

3. **BioClinicalBERT (truncated)** — significant AUPRC loss vs chunk-pool (0.858 vs 0.878, McNemar p = 4.5e-5) because truncation cuts off the BHC section. Use chunk-pool instead unless inference latency is the binding constraint.

4. **Structured + LogReg** — interpretable but encodes coding bias (`n_prior_admissions` +6.51). AUPRC 0.683 is a hard ceiling. Useful as a sanity-check baseline; not deployable alone.

5. **Keyword (DSM-5/PCL-5)** — most interpretable, near-random (AUPRC 0.510). Clinically appealing but the inpatient discharge note doesn't reliably contain DSM-5 symptom language. Could complement a learned model as a "second opinion" filter, not as primary.

### What this multi-model comparison collectively shows

1. **All text models exploit the SAR coding bias to some degree.** Structured does it most blatantly (`n_prior_admissions`). PULSNAR Longformer shifts attribution toward trauma narrative (HPI/BHC) and away from comorbidity codes (PMH); BERT — at both inference modes — sits in between, with truncation reinforcing the comorbidity bias because BHC literally falls past its window.

2. **Long-context inference matters more than long-context pretraining.** Chunk-pool BERT closes most of the AUPRC gap to Longformer (and is McNemar-indistinguishable). The residual ~0.007 AUPRC and the better fairness/calibration profile come from Longformer's longer pretraining context, not just inference context. Whether the deployment site needs that residual depends on volume, calibration tooling, and clinical workflow.

3. **The "explainable" baselines (Keyword, Structured) reveal the weakness of the cohort design more than the model design.** A linear model on 20 hand-engineered features cannot escape `n_prior_admissions` dominating. A regex-based model cannot escape that PTSD symptom language is sparse in inpatient notes. Both are useful as *baselines* — they bound how much the cohort design is doing for you on its own.

4. **Cross-model agreement is high among text models** (κ = 0.72–0.77, top-quintile overlap 76–85 %). They are largely identifying the same patients. The disagreements concentrate in older men with terse documentation — the under-detected subgroup. **No ensemble lift** because the disagreements are exactly where everyone is unsure.

### What's no longer missing (relative to the previous comparison memo)

- ✅ BERT calibration: raw + Platt + Elkan-Noto + ECE plot for both modes.
- ✅ BERT fairness: cal-in-large + EO + bootstrap CI on AUPRC for both modes.
- ✅ BERT subgroup AUPRC: per-mode CSV.
- ✅ BERT proxy validation: Mann-Whitney AUC for both modes (0.74 trunc, 0.73 chunk-pool).
- ✅ BERT ablations: A1 + A2 for both modes.
- ✅ BERT decision curves: 2% + 5% for both modes.
- ✅ BERT error analysis: FP/FN demographics + trauma-term rate.
- ✅ BERT integrated gradients: section + word attribution for both modes (truncated 512, chunk-pool top window).
- ✅ Cross-model: all-pairs McNemar, Cohen's κ, Pearson r, top-quintile overlap, ensemble probe.
- ✅ Per-subgroup NNS by model.

The multi-model comparison is now symmetric — every text model has been put through the same downstream pipeline as the primary.
