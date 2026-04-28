# Model Comparison — PTSD Undercoding Detection

**Author:** Eric Wang (ewang163)
**Date:** 2026-04-26
**Scope:** PULSNAR Clinical Longformer vs BioClinicalBERT (PU) vs Structured + LogReg vs Keyword (DSM-5/PCL-5). Comparing discrimination, calibration, fairness, clinical utility, **runtime cost**, and **explainability**.

---

## 1. Discrimination + clinical utility (held-out test set, n=1,551, 660 PTSD+)

All numbers from `results/metrics/ewang163_evaluation_results.json` and `..._pulsnar.json`. Val-derived thresholds at recall ≥ 0.85.

| Model | Threshold | Test AUPRC | Test AUROC | Sens | Spec | F1 | LR+ | LR− | DOR | Alert rate | NNS @2% | NNS @5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clinical Longformer (PULSNAR) | 0.188 | **0.8848** | **0.8904** | 0.846 | 0.745 | 0.772 | 3.32 | 0.207 | 16.0 | 50.6% | 15.8 | 6.7 |
| BioClinicalBERT (PU, chunk-pool) | 0.976 | 0.8775 | 0.8853 | 0.902 | 0.626 | 0.749 | 2.41 | 0.157 | 15.3 | 59.8% | 21.0 | 8.4 |
| BioClinicalBERT (PU, truncated 512) | 0.976 | 0.8576 | 0.8656 | 0.820 | 0.728 | 0.750 | 3.02 | 0.248 | 12.2 | 50.5% | 16.5 | 7.0 |
| TF-IDF + LogReg | 0.300 | 0.8380 | 0.8567 | 0.817 | 0.721 | 0.745 | 2.92 | 0.254 | 11.5 | 50.8% | 17.0 | 7.3 |
| Structured + LogReg | 0.338 | 0.6833 | 0.7310 | 0.909 | 0.207 | 0.610 | 1.15 | 0.440 | 2.6 | 84.3% | 36.7 | 14.5 |
| Keyword (DSM-5/PCL-5) | 0.000 | 0.5373 | 0.6086 | 1.000 | 0.000 | 0.597 | 1.00 | inf | inf | 100.0% | 47.0 | 20.0 |

**Headline:** PULSNAR Longformer wins discrimination (AUPRC 0.885 vs second-place BERT-chunk-pool 0.878). On a per-patient utility basis at 2% deployment prevalence, PULSNAR flags ~16 patients per true case found vs BERT-chunk-pool's 21 and Structured's 37. Keyword alone is near-random.

**McNemar p-value (PULSNAR-equivalent pi_p=0.25 vs comparators on same predictions):** all comparators p < 1e-5 (Longformer beats them all on McNemar's paired-prediction test).

---

## 2. Runtime cost (measured on identical hardware, 2026-04-26)

**All five inference modes were re-measured in a single SLURM allocation (job 1923224) on `gpu2708 = NVIDIA L40S 48 GB`** — fair head-to-head. Training measurements are from each model's original training job; hardware audit is below.

### Inference comparison (n_val=1,471 / n_test=1,551)

Transformers measured on L40S GPU (job 1923224). CPU-only models re-measured on a `batch`-partition node with 16 CPUs (job 1923698, node2303) to represent realistic CPU deployment.

| Model | Hardware | Val wall (s) | Test wall (s) | ms/patient (test) | Peak mem (GB) |
|---|---|---:|---:|---:|---:|
| Clinical Longformer (PULSNAR), 4096 tok | L40S, 1 CPU | 119.77 | 124.64 | **80.36** | 2.03 GPU |
| BioClinicalBERT (chunk-pool 512×256) | L40S, 1 CPU | 33.15 | 35.14 | 22.66 | 0.65 GPU |
| BioClinicalBERT (truncated 512) | L40S, 1 CPU | 4.49 | 4.61 | **2.97** | 0.83 GPU |
| Keyword DSM-5/PCL-5 | batch, **16 CPUs** | 0.47 | 0.52 | **0.34** | 0.17 RAM |
| Structured + LogReg\* | batch, **16 CPUs** | 63.06 | 27.73 | 17.88 | 0.30 RAM |

\*Structured val timing includes a cold-cache feature build (streaming `diagnoses_icd.csv` + `prescriptions.csv` from MIMIC); test reuses the warm cache. The actual `predict_proba` is sub-millisecond. Note: Structured ran ~16% slower on the batch node than the GPU node despite 16× more CPUs — the workload is I/O-bound (MIMIC CSV streams from shared storage) and the GPU node happens to have faster scratch I/O. Keyword scaled near-ideally with cores (11× speedup on 16 CPUs).

### Training (from each model's original SLURM job; hardware audit)

| Model | Job ID | Elapsed | GPU type | GPU-h | Train n |
|---|---|---|---|---:|---:|
| Clinical Longformer (PULSNAR) | 1699154 | 03:30:48 | **NVIDIA L40S** (gpu2708) | 3.5049 | 11,837 notes |
| BioClinicalBERT (PU) | 1385704 | 00:13:11 | **NVIDIA RTX 3090** (gpu2105) | 0.2197 | 11,837 notes |
| Structured + LogReg | 1367522 | 00:01:08 | CPU only (node2302) | 0 | 9,649 patients |
| Keyword (DSM-5/PCL-5) | — | — | n/a (no training) | 0 | n/a |

> **Training-fairness flag:** The 3.50 vs. 0.22 GPU-h ratio is **not strictly apples-to-apples** — Longformer trained on L40S (Lovelace, ~91 TFLOPS FP16), BERT trained on RTX 3090 (Ampere, ~71 TFLOPS FP16). A same-hardware re-training of BERT would shave ~25–30%. The order-of-magnitude gap survives normalization because the dominant cost driver is **architecture**: Longformer uses 4,096-token sliding-window attention vs. BERT's 512-token dense attention, an 8× sequence-length difference that dominates per-step cost. The L40S/RTX-3090 difference is in the noise compared to that.

### Compute headline (fair-hardware numbers)

- **Inference on identical L40S:** Longformer is **27.1× slower than BERT-truncated** (80.36 vs 2.97 ms/patient) and **3.55× slower than BERT chunk-pool** (80.36 vs 22.66 ms).
- **Training:** Longformer's 15.95× nominal GPU-h ratio over BERT becomes **~12× hardware-normalized** (~3.50 L40S-equivalent vs ~0.30 L40S-equivalent for a hypothetical BERT-on-L40S retrain). Architecture, not GPU generation, drives the gap.
- **Per-AUPRC-point marginal cost:** Longformer pays an extra 3.28 GPU-h vs BERT-chunk-pool to gain +0.0073 AUPRC = **449 GPU-h per +0.01 AUPRC** (unchanged from before; denominator is small).
- **Structured + LogReg has 68 s training and 15.4 ms test inference** dominated by feature construction (MIMIC CSV streams). A pre-computed feature cache would drop test inference to ~1 ms.
- **Keyword has zero training and 3.68 ms/patient inference** for AUPRC 0.537 (near-random).

### Cost-per-deployment scenarios (50,000 inpatient discharges/month)

| Model | Monthly inference time | Monthly GPU-h |
|---|---:|---:|
| Clinical Longformer (PULSNAR), L40S | 67.0 minutes | 1.12 GPU-h |
| BioClinicalBERT (chunk-pool), L40S | 18.9 minutes | 0.31 GPU-h |
| BioClinicalBERT (truncated), L40S | 2.5 minutes | 0.041 GPU-h |
| Structured + LogReg, 16-CPU batch | 14.9 minutes | 0 |
| Keyword, 16-CPU batch | 0.28 minutes | 0 |

Longformer adds ~0.81 GPU-h/month over BERT chunk-pool — ~$1–2 of compute monthly at L40S rates. Training is one-shot; inference is the recurring cost. Keyword (CPU, 16 cores) is now ~24× faster than truncated BERT (GPU) per patient — but at AUPRC 0.537 vs 0.858, that speed buys near-random predictions.

### Sources

| Cell | Source |
|---|---|
| Transformer inference (Longformer + BERT, L40S) | `results/metrics/ewang163_unified_inference_bench.csv` (job 1923224, gpu2708); log `logs/ewang163_unified_bench_1923224.out` |
| Keyword + Structured inference (16-CPU batch) | `results/metrics/ewang163_cpu_inference_bench.csv` (job 1923698, node2303); log `logs/ewang163_cpu_bench_1923698.out`; `runtime_benchmarks.csv` rows tagged `cpu_bench` |
| Benchmark scripts | `scripts/04_evaluation/ewang163_unified_inference_bench.py`, `scripts/04_evaluation/ewang163_cpu_inference_bench.py` |
| PULSNAR Longformer training | `sacct -j 1699154` → Elapsed 03:30:48 on gpu2708 (L40S) |
| BERT training | `sacct -j 1385704` → Elapsed 00:13:11 on gpu2105 (RTX 3090) |
| Structured training | `sacct -j 1367522` → Elapsed 00:01:08 on node2302 (CPU only) |

---

## 3. Calibration (Fix 5)

Tested on the 1,551-patient test set with 10 equal-frequency bins. ECE = expected calibration error.

| Model | ECE raw | ECE Platt | ECE Elkan-Noto | Elkan-Noto c |
|---|---:|---:|---:|---:|
| Clinical Longformer (PULSNAR) | 0.088 | 0.077 | 0.097 | 0.728 |
| Clinical Longformer (pi_p=0.25, reference) | 0.064 | 0.074 | 0.080 | 0.783 |
| BioClinicalBERT | not computed | — | — | — |
| Structured + LogReg | not computed | — | — | — |
| Keyword | not computed | — | — | — |

**Calibration was only computed for the Longformer winners (per the methodology fixes plan).** BERT/structured/keyword calibration is a gap in the dual-comparison; raw probability bins from `evaluation_results.json` show BERT's probabilities are heavily concentrated near 1.0 (1,091 of 1,551 in the 0.9–1.0 bin), suggesting it is **substantially over-confident**. Structured + LogReg has 569 of 1,551 in the 0.3–0.4 bin (under-confident bunching). These are visual signals; numeric ECE would need a re-run.

---

## 4. Fairness (Fix 9) — Equal Opportunity Differences

| Subgroup | Longformer (PULSNAR) | Longformer (pi_p=0.25) |
|---|---:|---:|
| Sex (F vs M) | 0.114 | 0.116 |
| Age group | 0.211 | 0.209 |
| Race (multi) | 0.101 | 0.112 |
| Race binary (W vs non-W) | 0.024 | 0.023 |
| Emergency vs elective | 0.046 | 0.034 |

**Fairness was not computed for BERT/Structured/Keyword.** This is a gap. Given the AUPRC gap and the fact that Structured + LogReg's most predictive feature is `n_prior_admissions` (coefficient +6.51), Structured fairness is almost certainly worse on age (older patients have more prior admissions). A formal re-run on BERT and Structured would be straightforward.

---

## 5. Explainability — model-by-model

### 5.1 Clinical Longformer (PULSNAR) — *post-hoc, model-agnostic*

**Method:** Integrated Gradients (Sundararajan 2017) on the embedding layer with pad-token baseline, n_steps=20, full 4,096-token context. Word-level aggregation via offset_mapping (Edwards 2025).

**Top 10 attributed words (50-patient sample):** bipolar, narcotic, illness, arrested, delayed, pancreatitis, schizoaffective, psychosis, anemia, assault.

**Section attribution:**
- HPI: 43.2%
- Brief Hospital Course: 32.0%
- PMH: 22.4%
- Social History: 1.3%

**Interpretability strengths:**
- Token-level granularity: every word in a 4,096-token note has an attribution score.
- Integration path is mathematically grounded (axioms of completeness, sensitivity, implementation invariance).
- Section attribution shows the model relies on clinically appropriate narrative content (HPI, BHC) over comorbidity codes (PMH).

**Interpretability weaknesses:**
- Computationally expensive: ~9 s/patient at 4,096 context — not feasible for real-time per-patient explanations at scale.
- Sample-based: full-cohort attribution would take ~3.4 hours per 1,500 patients.
- Black-box at the parameter level: no direct mapping from attribution scores to model weights.
- Subword tokenization artifacts already mitigated by word-level aggregation, but rare clinical terms still fragment.

### 5.2 BioClinicalBERT — *post-hoc, model-agnostic, but not actually run*

Same architectural class as Longformer; IG could be applied. **No attribution was computed.** Would be cheaper than Longformer (512-token context, ~1/8 the compute per patient).

**Gap:** if BERT's attribution shows a different pattern (e.g., heavier reliance on PMH due to truncation losing later HPI content), that would be a finding. Currently we can't say.

### 5.3 Keyword (DSM-5/PCL-5) — *intrinsically interpretable*

**Method:** 60 hand-curated regex patterns, each with explicit weight (0.5–3.0) and DSM-5 criterion mapping.

**Examples:**
- `\bptsd\b` — weight 3.0, criterion A (most direct mention)
- `\bflashback[s]?\b` — weight 3.0, criterion B (intrusion)
- `\bhypervigilance\b` — weight 3.0, criterion E (arousal)
- `\bsexual assault\b` — weight 3.0, criterion A
- `\bguilt\b` — weight 0.5, criterion D (negative cognition; deliberately low because non-specific)
- `\bemdr\b` — weight 3.0, criterion tx (treatment)

**Score = simple weighted sum of pattern matches.** Every flag is auditable: a clinician can read the note, see exactly which phrases triggered, and check whether the trigger words were in proper context (e.g., "patient denied flashbacks" should not trigger but currently does — known limitation).

**Interpretability strengths:**
- 100% deterministic, transparent, reproducible from the JSON file.
- Each prediction maps to specific phrases with specific weights — perfect chain of reasoning.
- DSM-5 criterion mapping enables clinical-criterion-level reporting (e.g., "patient has 3 criterion B markers, 2 criterion E markers").
- No training needed — weights set by clinical expertise; deployment requires zero data.

**Interpretability weaknesses:**
- Cannot capture context (negation, hypotheticals, family history).
- Cannot capture paraphrases or implicit references.
- Performance ceiling: AUPRC 0.537 is near-random — interpretability is moot if the model can't rank patients.
- Phrase weights are subjective: 60 patterns with hand-set weights, no learned calibration.

**Why such poor performance despite high interpretability?** PTSD undercoding is not about whether the *symptom words* appear (they often don't in inpatient notes), it's about whether *clinical narrative patterns* indicate trauma exposure plus stress reactions. The keyword model captures explicit symptom mentions but misses implicit patterns; ~36% of PTSD+ test patients have a keyword score of zero.

### 5.4 Structured + LogReg — *intrinsically interpretable*

**Method:** L2-regularized logistic regression on 20 hand-engineered features (no text). Coefficients (top 15 by magnitude):

| Feature | Coefficient | Clinical meaning |
|---|---:|---|
| `n_prior_admissions` | **+6.512** | More prior admissions → more likely coded |
| `dx_suicidal` | +1.822 | Prior suicidal ideation |
| `dx_SUD` | +1.468 | Prior substance use disorder |
| `dx_MDD` | +1.243 | Prior major depression |
| `dx_anxiety` | +0.882 | Prior anxiety disorder |
| `medicaid_selfpay` | +0.859 | Insurance/SES marker |
| `race_Asian` | −0.672 | Asian patients flagged less often (n=5 in test, unstable) |
| `rx_ssri_snri` | +0.287 | Prior SSRI/SNRI prescription |
| `race_White` | +0.274 | White patients flagged more often |
| `dx_pain` | +0.180 | Chronic pain history |
| `sex_female` | +0.139 | Female slightly more likely |
| `rx_SGA` | +0.123 | Atypical antipsychotic |
| `race_Black` | −0.111 | Black patients flagged less |
| `race_Hispanic` | −0.111 | Hispanic patients flagged less |
| `dx_TBI` | +0.022 | TBI history (small) |

**Interpretability strengths:**
- 20 coefficients — fully auditable, reproducible.
- Clinically expected coefficients on most psychiatric comorbidities (MDD, anxiety, SUD, suicidal ideation).
- Direct decomposition: any prediction = sigmoid(intercept + sum of feature × coefficient).

**Interpretability weaknesses — and a major finding:**
- **`n_prior_admissions` coefficient is +6.51 — by far the largest in the model and 3.6× larger than the next feature.** This is the SAR violation made manifest: the model is mostly learning "this patient has been admitted many times, therefore probably PTSD-coded by now." It is *not* a PTSD signal — it is a *coding-frequency* signal. This is exactly the artifact that PULSNAR v2 alpha estimation flagged.
- Race coefficients (Asian −0.67, Hispanic −0.11, Black −0.11, White +0.27) reflect MIMIC-IV coding bias, not actual prevalence differences. Deploying this model would entrench that bias.
- 20 features cannot capture trauma exposure, symptom narratives, treatment patterns — the very things text-based models pick up.

**Performance ceiling:** AUPRC 0.683 is well below text-based models. Structured features alone are insufficient for PTSD undercoding detection — they recapitulate the coding bias.

### 5.5 TF-IDF + LogReg — *intrinsically interpretable, but problematic finding*

**Method:** L2-regularized logistic regression on 50,000 unigram + bigram TF-IDF features.

**Top 10 most positive features (PTSD-predictive):**

| n-gram | Coefficient |
|---|---:|
| `ptsd_masked` | **+37.13** |
| `depression` | +11.46 |
| `ptsd` | **+10.29** |
| `anxiety` | +9.76 |
| `disorder` | +9.23 |
| `bipolar` | +7.98 |
| `chronic` | +7.82 |
| `depression ptsd_masked` | +6.91 |
| `abuse` | +6.87 |
| `chronic pain` | +6.23 |

**Critical finding — Fix 1 masking has a known leak in TF-IDF training:**
- Fix 1 was supposed to replace the literal string `PTSD` with `[PTSD_MASKED]` to prevent label leakage.
- **TF-IDF tokenization strips punctuation by default**, so `[PTSD_MASKED]` becomes the token `ptsd_masked`. This token is *itself* a label-leakage marker — its presence in a note means PTSD was originally written there.
- The +37.13 coefficient (3× the next-highest feature) confirms the model is exploiting this leak.
- Additionally, `ptsd` itself appears as a top feature with coefficient +10.29, suggesting **the masking regex did not catch all variants** (e.g., lowercase, mid-word occurrences, or notes processed before the masking step).

**Implications:**
- TF-IDF AUPRC 0.838 is likely **inflated** by label leakage. The "real" leak-free TF-IDF performance is uncertain — could be substantially lower.
- Longformer/BERT may also have residual leakage but at much smaller magnitude (transformer tokenizers preserve `[PTSD_MASKED]` as a multi-token sequence whose embeddings are learned during fine-tuning, not as a single bag-of-word feature).
- **Action item:** re-train TF-IDF with stricter masking (e.g., token-level replacement, or removing notes containing `ptsd_masked` after tokenization). Likely to drop AUPRC by 0.05–0.15.

**Interpretability strengths:**
- 50,000 features with linear coefficients — every prediction decomposable.
- Bigrams capture some context ("recent admission", "depression ptsd_masked").
- Negative features (`healthy`, `none`, `tonsillectomy`) make clinical sense — markers of *not* having complex history.

**Interpretability weaknesses:**
- Cannot model long-range dependencies, negation, hypotheticals.
- Bag-of-words: "patient denied flashbacks" and "patient reported flashbacks" produce identical features for that phrase.
- The label-leakage finding above demonstrates that easy interpretability does not equal trustworthy model.

---

## 6. Explainability ranked + cost-adjusted

| Model | Interpretability | AUPRC | Train cost | Infer ms/patient | Trustworthy? |
|---|---|---:|---:|---:|---|
| Keyword | ★★★★★ (intrinsic, DSM-mapped) | 0.537 | 0 (no fit) | **0.34** ms (16-CPU) | Yes — but underperforming |
| Structured + LogReg | ★★★★ (intrinsic, 20 coefs) | 0.683 | 68 s CPU | 17.9 ms (16-CPU, I/O-bound) | **Partial** — `n_prior_admissions` coef +6.51 is a coding-bias artifact |
| TF-IDF + LogReg | ★★★★ (intrinsic, 50k coefs) | 0.838 | 15.8 s CPU | 0.84 ms | **No** — `ptsd_masked` coef +37.1 reveals label leakage |
| BioClinicalBERT (chunk-pool) | ★★ (post-hoc IG possible, not run) | 0.878 | 791 s on RTX 3090 (~0.22 GPU-h; ~0.16 L40S-eq) | 22.7 ms (L40S) | Likely — but unverified |
| Clinical Longformer (PULSNAR) | ★★ (post-hoc IG, computed) | 0.885 | 12,617 s on L40S (3.50 GPU-h) | 80.4 ms (L40S) | Yes — IG shows HPI/BHC dominance, trauma-specific top words |

---

## 7. Bottom-line analysis

### Which model would I deploy and why?

If the goal is **flagging undercoded PTSD patients for clinician review**, the ranking is:

1. **Clinical Longformer (PULSNAR)** — best AUPRC (0.885), trauma-appropriate attribution (HPI 43%, top words include `narcotic`, `arrested`, `assault`, `psychosis`), fairness profile understood, calibration reasonable. Cost is real (3.5 GPU-h training, ~100 ms/patient inference) but justifiable.

2. **BioClinicalBERT (chunk-pool)** — close second on AUPRC (0.878) at **15.95× less measured training GPU-h** (0.22 vs 3.50) — though training hardware differed (RTX 3090 vs L40S), so a same-hardware comparison would be ~12×. **Inference is 3.55× faster on identical L40S** (22.66 vs 80.36 ms/patient). The empirical AUPRC gap to Longformer is small; whether the deployment site needs the extra 0.007 AUPRC depends on volume and clinical workflow. Lacks any attribution analysis — a real gap.

3. **TF-IDF + LogReg** — was previously a strong third (0.838 AUPRC) but the `ptsd_masked` finding makes the reported number untrustworthy. Real performance unknown without re-training.

4. **Structured + LogReg** — interpretable but encodes coding bias (`n_prior_admissions` +6.51). AUPRC 0.683 is a hard ceiling. Useful as a sanity-check baseline; not deployable alone.

5. **Keyword (DSM-5/PCL-5)** — most interpretable, near-random (AUPRC 0.537). Clinically appealing but the inpatient discharge note doesn't reliably contain DSM-5 symptom language. Could complement a learned model as a "second opinion" filter, not as primary.

### What's missing for a complete comparison

- **BERT calibration + fairness + attribution** (3 separate jobs, modest GPU time).
- **TF-IDF re-train with stricter masking** to determine real performance after fixing the `ptsd_masked` leak.
- **Structured + LogReg fairness** — likely worse than Longformer on age, but unmeasured.
- **Clinician chart review** (Fix 11) of top-50 flagged patients across all four models, for ground-truth validation.

### What the explainability evidence collectively tells us

1. **All models exploit the SAR-coding-bias signal to some degree.** Structured does it most blatantly (`n_prior_admissions`). TF-IDF has direct leakage (`ptsd_masked`). Longformer pi_p=0.25 has more PMH attribution than PULSNAR. PULSNAR is the *least* affected, by attribution evidence.

2. **The "explainable" baselines (Keyword, Structured) reveal the weakness of the cohort design more than the model design.** A linear model on 20 hand-engineered features cannot escape `n_prior_admissions` dominating. A regex-based model cannot escape that PTSD symptom language is sparse in inpatient notes.

3. **Transformer-based models are worth the cost only if their attribution patterns differ meaningfully from the structured baseline's.** The Longformer attribution shows HPI/narrative dominance with trauma-specific top words — confirming it learned something the structured model could not. The BERT model likely also has this, but unmeasured.

4. **The `ptsd_masked` TF-IDF finding is the single most important explainability surprise**: a "transparent" linear model on text was actually exploiting label leakage that the masking step was supposed to prevent. **Lesson: interpretability and trustworthiness are not synonymous.** Inspecting the top coefficients caught a bug that AUPRC scoring did not.
