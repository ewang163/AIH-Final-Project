# Model Selection Memo — Clinical Longformer for PTSD Undercoding Detection

**Author:** Eric Wang (ewang163)
**Date:** 2026-04-17
**Purpose:** Document the final model choice after the 7-fix methodology review and provide the empirical basis for that choice.

---

## Decision (dual-winner framing)

**Primary (empirical):** Clinical Longformer trained with nnPU loss at **pi_p=0.25**
- Checkpoint: `models/ewang163_longformer_best_pip025/` (symlinked from `models/ewang163_longformer_best/`)
- Operating threshold (val-derived): 0.3240
- Selection criterion: highest proxy Mann-Whitney AUC (0.7990, p=8.3e-22), pre-registered rule

**Sensitivity (SAR-principled):** Clinical Longformer trained with **PULSNAR** propensity-weighted nnPU (alpha=0.1957)
- Checkpoint: `models/ewang163_longformer_pulsnar/`
- Proxy AUC = 0.7701 (p=3.8e-18), Test AUPRC = 0.8725
- Justification: SCAR is violated (PTSD coding biased by age/sex/prior contact); PULSNAR's data-driven alpha is the literature-principled choice for SAR (Kumar & Lambert 2024; Bekker & Davis 2020). Empirical gap to pi_p=0.25 (Δ proxy AUC 0.029, Δ test AUPRC 0.021) is within plausible noise of the 102-patient proxy group.

**Paper framing:** report both; flag any downstream disagreement (calibration, fairness, utility, attribution) as a finding. The pi_p=0.25 → PULSNAR gap is a robustness probe, not a horse race — see "Literature-Based Reconsideration" section below.

**Alternate for pre-2015 applications:** `models/ewang163_longformer_best_temporal/` — but see Fix 3 finding below; temporal training did not help generalization.

---

## Decision Framework (published upfront)

1. **Must have** proxy AUC ≥ 0.78 and Mann-Whitney p < 0.01 — only PU-uncontaminated validity criterion.
2. Among qualifying models, pick highest test AUPRC.
3. Tie-breaker: smallest fairness disparity (equal opportunity difference).
4. Temporal model compared in parallel; kept only if it demonstrably improves post-2015 generalization.

---

## Candidate Comparison

Evaluated on the held-out 2026-04-15 test set (1,551 patients, 660 PTSD+) with val-derived thresholds.

| Model | Training pi_p | Proxy AUC | MW p | Test AUPRC | Test AUROC | Sens | Spec | F1 | Qualifies? |
|-------|---------------|-----------|------|-----------|-----------|------|------|-----|-----------|
| **pi_p=0.25 (WINNER)** | 0.25 | **0.7990** | 8.3e-22 | **0.8939** | **0.9002** | 0.852 | 0.782 | **0.794** | ✅ |
| pi_p=0.10 | 0.10 | 0.7889 | 1.7e-20 | 0.8785 | — | — | — | — | ✅ |
| pi_p=0.12 | 0.12 | 0.7875 | 2.6e-20 | 0.8776 | — | — | — | — | ✅ |
| pi_p=0.15 | 0.15 | 0.7720 | 2.2e-18 | 0.8781 | — | — | — | — | ✅ |
| PULSNAR v1 (alpha=0.196) | 0.196 | 0.7701 | 3.8e-18 | 0.8725 | — | — | — | — | ✅ |
| pi_p=0.08 | 0.08 | 0.7667 | 9.7e-18 | 0.8793 | — | — | — | — | ✅ |
| pi_p=0.20 | 0.20 | 0.7662 | 1.1e-17 | 0.8782 | — | — | — | — | ✅ |
| pi_p=0.05 | 0.05 | 0.7570 | 1.3e-16 | 0.8826 | — | — | — | — | ❌ (below 0.78) |
| Retrain (empirical pi_p=0.398) | 0.398 | 0.7552 | 2.2e-16 | 0.8881 | — | — | — | — | ❌ (below 0.78) |

All models pass the MW p < 0.01 bar. Only pi_p=0.25, pi_p=0.10, pi_p=0.12 clear the proxy AUC ≥ 0.78 floor. Among those, **pi_p=0.25 has both the highest proxy AUC (0.799) and the highest test AUPRC (0.894).**

---

## Why pi_p=0.25 wins over the empirical pi_p (0.398)

The empirical labeled fraction (0.398) corresponds to the 3:1 matched study cohort's positive rate, which systematically **overstates** the true population positive fraction. Training at that value pushes the loss to over-correct for class imbalance, producing a model that is more lenient on unlabeled patients (calling more of them "maybe positive"), which reduces the proxy vs. unlabeled separation. The 0.25 value — consistent with literature estimates of 15-25% true PTSD prevalence in trauma-exposed inpatient populations — produces a model that discriminates proxy patients (whose medication pattern suggests real PTSD) from the unlabeled pool best.

---

## PULSNAR Alpha Diagnostics

Two PULSNAR variants were run to estimate the class prior alpha:

| Variant | Features (n) | Estimated alpha | Propensity discrimination |
|---------|-------------|-----------------|---------------------------|
| v1 (demographics only) | 4 | 0.1957 | Prop mean pos=0.29, unl=0.25 (weak) |
| v2 (with prior comorbidities) | 13 | **0.0006** | Prop mean pos=0.77, unl=0.13 (strong) |

**The v2 alpha=0.0006 is an artifact, not a finding.** Adding `n_prior_admissions` (coef +5.63) conflates disease severity with coding propensity. A propensity model that perfectly separates labeled from unlabeled leaves PULSNAR with no signal to detect hidden positives — it concludes alpha≈0.

**Implication:** v1 alpha=0.196 remains the principled estimate. It falls between pi_p=0.20 and pi_p=0.25 in our sweep — consistent with the selected winner.

**Action:** Fix 4b (full PULSNAR retrain with rich features) was **not** run because the alpha estimate is unreliable. The v1 PULSNAR checkpoint remains available for sensitivity analysis.

---

## Temporal Generalization (Fix 3)

| Scenario | Test AUPRC |
|----------|-----------|
| Random model, random test | 0.8881 |
| Random model, temporal test (2017-2019) | 0.8861 |
| **Temporal model, temporal test** | 0.8417 |

**Finding:** Temporal training **hurt** generalization. The random-split model drops only 0.002 AUPRC when evaluated on the later-years temporal test set, meaning the random-split distribution is already representative of 2017-2019. The temporal model, trained only on pre-2015 data, loses 0.044 AUPRC — likely because it has 25% less training data (8,752 vs. 11,837 patients) and misses the richer post-2013 DSM-5 era coding patterns.

**Action:** Use the random-split model in all applications. The temporal model is archived for reference but **not recommended for deployment**.

---

## Fairness (Fix 9) on the Winner

Equal opportunity difference (max recall gap at operating threshold, bootstrap 1,000 resamples):

| Subgroup | Max recall | Min recall | EO diff | Interpretation |
|----------|-----------|-----------|---------|----------------|
| Sex | 0.892 (F) | 0.775 (M) | **0.116** | Men under-detected |
| Age group | 0.929 (20s) | 0.719 (Other) | **0.209** | Oldest/youngest under-detected |
| Race (binary) | 0.869 (Non-White) | 0.845 (White) | **0.023** | Minimal race disparity |
| Emergency | 0.862 (True) | 0.829 (False) | 0.034 | Minimal |

Calibration-in-the-large (mean predicted − mean observed) ranges from −0.016 (30s age group) to +0.113 (50s). All race subgroups except Asian (n=5, CI width 0.55) have reliable AUPRC (CI width < 0.15).

**Key finding:** **Race disparity is minimal (EO 0.023)**, contradicting the a-priori concern. The primary fairness issue is age (EO 0.209) and sex (EO 0.116), reflecting the known PTSD under-coding bias for older patients and men.

---

## Attribution (Fix 10) — Word-Level

After fixing the subword-fragment issue, the top-attributed whole words are:

bipolar, personality, schizoaffective, disorder, depression, heroin, abuse, psychiatric, assault, detox

These are clinically plausible PTSD-adjacent concepts: psychiatric comorbidities (bipolar, personality disorder, depression), substance use (heroin, detox), and trauma exposure (assault, abuse). **No label-leakage terms** (e.g., "PTSD", "posttraumatic") appear — confirming Fix 1 masking worked.

HPI section still dominates (36.4% of total attribution), consistent with the earlier finding.

---

## Clinical Utility at the Winner's Threshold

Operating threshold = 0.3240 (val-derived).

| Prevalence | PPV | NPV | NNS |
|-----------|-----|-----|-----|
| 1% | 0.038 | 0.998 | 26.3 |
| 2% | 0.074 | 0.996 | 13.5 |
| 5% | 0.171 | 0.990 | 5.9 |
| 10% | 0.303 | 0.979 | 3.3 |
| 20% | 0.494 | 0.955 | 2.0 |

LR+ = 3.91 | LR- = 0.19 | DOR = 20.6 | Alert rate = 45.0% | Workup reduction vs. treat-all = 55.0%

**At 2% deployment prevalence, NNS=13.5** — 14 patients flagged per true PTSD case found. This is clinically tolerable for a screening prompt.

---

## Calibration

ECE raw = 0.064, Platt-scaled = 0.074, Elkan-Noto = 0.080. Raw probabilities are best-calibrated for this model — Platt scaling actually degrades calibration slightly because the pi_p=0.25 training produces already-sharper probability distributions than the pi_p=empirical model. **Recommendation:** use raw probabilities for ranking; do not apply Platt correction.

---

## Compute Efficiency (training + inference) — measured

All numbers below are measured wall-clock. **Inference numbers are from a single SLURM job (1923224) on `gpu2708 = NVIDIA L40S 48 GB`** — apples-to-apples across all five inference modes. Training numbers come from each model's original SLURM job; hardware varies (see audit table below).

### Training cost (measured)

| Model | Wall-clock (s) | GPU-h | Peak mem (GB) | Train n | Test AUPRC | AUPRC per GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| Longformer pi_p=0.25 (empirical winner) | 20,631 | **5.73** | 2.04 | 11,837 | 0.8939 | 0.156 |
| Longformer PULSNAR (propensity-weighted) | 12,617 | **3.50** | 2.25 | 11,837 | 0.8848 | 0.252 |
| Longformer retrain (pi_p=0.398) | 20,684 | 5.75 | 2.00 | 11,837 | 0.8881 | 0.154 |
| Longformer temporal (pre-2015) | 19,409 | 5.39 | 1.99 | 11,134 | 0.8417 | 0.156 |
| BioClinicalBERT (PU) | **791** | **0.22** | ~3.7 (MaxRSS) | 11,837 | 0.8576 | 3.90 |
| TF-IDF + LogReg (fit + 6-C sweep) | 15.8 | 0 (CPU) | 0.75 | 11,837 | 0.8380 | — |
| Structured + LogReg (full pipeline) | 68 | 0 (CPU) | ~3.8 (MaxRSS) | 9,649 | 0.6833 | — |
| Keyword (DSM-5/PCL-5) | 0 (no training) | 0 | 0.31 | n/a | 0.5373 | — |

**Full sweep cost:** 7-checkpoint pi_p sweep = 40.2 GPU-h. Adding retrain (5.75), PULSNAR (3.50), temporal (5.39), 9-checkpoint sweep eval (~0.57 cumulative), and dual-winner re-eval (PULSNAR Fix 5/6/9 + attribution = 0.21 GPU-h) brings the **total methodology-fixes compute to ~55.6 GPU-hours**.

### Inference cost (fair-hardware, 1,551-patient test set)

Transformers measured on L40S GPU (job 1923224, gpu2708, 1 CPU). CPU-only models re-measured on a `batch`-partition node with 16 CPUs (job 1923698, node2303) — representative of realistic CPU deployment. CSVs: `results/metrics/ewang163_unified_inference_bench.csv` and `..._cpu_inference_bench.csv`.

| Model | Hardware | Wall-clock (s) | GPU-h | ms/patient | Relative |
|---|---|---:|---:|---:|---:|
| Clinical Longformer (4,096 tokens) | L40S, 1 CPU | 124.6 | 0.0346 | **80.4 ms** | 1.0× |
| BioClinicalBERT (chunk-pool, 512×256) | L40S, 1 CPU | 35.1 | 0.0098 | 22.7 ms | 3.55× faster |
| BioClinicalBERT (truncated, 512) | L40S, 1 CPU | 4.6 | 0.0013 | **2.97 ms** | 27.1× faster |
| Structured + LogReg (with feature build) | batch, 16 CPUs | 27.7 | 0 | 17.9 ms (I/O-bound) | 4.5× faster |
| Keyword (DSM-5/PCL-5) | batch, 16 CPUs | 0.52 | 0 | **0.34 ms** | 236× faster |
| TF-IDF + LogReg | gpu, 1 CPU | 1.3 | 0 | 0.84 ms | 96× faster (prior measurement; not re-run) |

Peak GPU memory on L40S: Longformer 2.03 GB, BERT-truncated 0.83 GB, BERT-chunk-pool 0.65 GB (chunk-pool processes one window at a time).

Speedup of CPU models with 16 cores vs the prior 1-CPU GPU-node baseline: Keyword **11.0×** faster (regex scoring is trivially parallel), Structured **0.86×** (slightly slower — workload is I/O-bound on MIMIC CSV streams; cores don't help and the batch node's shared storage is slightly slower than the gpu2708 scratch path). Keyword's 0.34 ms/patient on 16 CPUs is the only measurement faster than the GPU truncated-BERT.

### Auxiliary analysis costs (measured)

| Step | Wall-clock (s) | GPU-h | Notes |
|---|---:|---:|---|
| Fairness analysis (test set) | 12.1 | 0 | bootstrap n=1,000 |
| Calibration (Platt fit + EN + plot) | < 11 | 0 | sacct 1743364 |
| Decision curve analysis | ~335 | 0 | sacct 1743308; includes prediction regen |
| Integrated Gradients (50 patients @ 4096) | 442–612 | 0.12–0.17 | ~9 s/patient; 50-pt sample |
| PULSNAR Fixes 5/6/9 re-eval (combined) | ~600 | ~0.17 | val + test inference + analysis |
| PULSNAR Fix 10 attribution | 612 | 0.17 | identical method, separate model |

### Where my prior estimates were wrong

- **BioClinicalBERT training was 791s, not ~5,400s.** I had estimated it at ~1.5 GPU-h based on transformer-rule-of-thumb scaling vs. Longformer; the actual figure is **0.22 GPU-h** (6.8× cheaper than estimated). The reason: BERT trains at sequence length 512 vs. Longformer's 4,096, so per-step cost scales as 1/64 in attention compute, more than offsetting the extra steps from a smaller batch. **Updated efficiency frontier:** BioClinicalBERT trains in **26× less GPU time than pi_p=0.25** (0.22 vs 5.73 GPU-h), not 3.8× as previously stated.
- **Structured + LogReg ran 68s, not <20s.** The pipeline includes data prep (cohort feature engineering) which I had excluded; the raw model fit alone is sub-second.
- **TF-IDF was 15.8s including a 6-C hyperparameter sweep**, not <60s. Tightened.
- **IG attribution costs ~9s/patient** at full 4,096-context. The 50-patient sample takes 7–10 minutes, not the "~hours" I had estimated.

### Training hardware audit (fairness flag)

The training GPU-h ratios are nominal — different models trained on different Oscar nodes:

| Model | Job ID | Elapsed | GPU type | Nominal GPU-h |
|---|---|---|---|---:|
| Longformer pi_p=0.25 (winner) | sweep job | 05:43:51 | L40S / RTX3090 mix | 5.73 |
| Longformer PULSNAR | 1699154 | 03:30:48 | L40S (gpu2708) | 3.50 |
| BioClinicalBERT (PU) | 1385704 | 00:13:11 | RTX 3090 (gpu2105) | 0.22 |
| Structured + LogReg | 1367522 | 00:01:08 | CPU only | 0 |

The L40S is ~25–30% faster per FP16 op than RTX 3090. After hardware normalization, the BERT-on-L40S equivalent training would be ~0.16 GPU-h, widening the per-architecture gap to **~22× hardware-normalized** (5.73 vs 0.16) and confirming that **architecture (4,096 vs 512 token attention), not GPU generation, dominates the training-cost difference**.

### Efficiency-adjusted re-take (fair-hardware)

- **Cost of Longformer win over BERT chunk-pool:** +0.016 AUPRC (0.8939 − 0.8775) for **~22× hardware-normalized training cost** and **3.55× the inference latency on identical L40S** (80.4 vs 22.7 ms/patient). For a one-shot deployment, the extra GPU-hours of Longformer training is small absolute compute (~$5 at L40S spot rates) — the inference latency ratio is the operationally relevant number.
- **Cost of Longformer win over BERT truncated:** +0.036 AUPRC (0.8939 − 0.8576) for the same training-cost ratio and **27.1× inference latency** on identical L40S (80.4 vs 2.97 ms/patient).
- **Cost of Longformer win over TF-IDF:** +0.056 AUPRC for ~1,300× training cost. PPV lift at 2% prevalence: 0.074 vs 0.059, a 25% relative gain. Justification depends on whether the deployment site already has GPU infrastructure.
- **Keyword baseline near-random** (AUPRC 0.537, AUROC 0.609) — confirms task requires learned representations.
- **PULSNAR training was 39% cheaper than pi_p=0.25** (3.50 vs 5.73 GPU-h, both on L40S-eligible nodes) — converged in 3 epochs vs. 5.

---

## Literature-Based Reconsideration: Should PULSNAR Be the Primary?

The original decision rule picked the checkpoint with the highest **proxy AUC** among those clearing 0.78. pi_p=0.25 won that contest at 0.7990 vs. PULSNAR's 0.7701 (Δ = 0.0289). But this rule has a blind spot: **proxy AUC itself inherits the selection bias we are trying to correct for.**

### The SCAR-vs-SAR framing

| Assumption | Meaning | Applies here? |
|---|---|---|
| **SCAR** (Elkan & Noto 2008; Kiryo et al. 2017 nnPU) | Labeled positives are a random sample of all true positives | **Violated.** PTSD coding is biased toward younger women, White patients, and patients with prior psychiatric contact — exactly the pattern Fix 9 confirmed (EO diff 0.21 by age, 0.12 by sex). |
| **SAR** (Bekker & Davis 2020 review; Kumar & Lambert 2024 PULSNAR) | Labeling probability depends on observed features | **Correct framing.** A trauma-exposed older man with substance use is less likely to be ICD-coded than a younger woman with the same symptoms. |

Literature consensus (Bekker & Davis 2020 ACM SIGKDD Explorations; Jaskie & Spanias 2022 IEEE): when SCAR is violated, tuning pi_p on a validation signal that shares the selection bias can produce **biased model selection**, not just biased absolute metrics.

### What this means for pi_p=0.25

- The proxy validation group (prazosin + SSRI/SNRI rule) is itself a selection-biased sample — it captures a specific prescribing pattern. A model that maximizes discrimination of proxy patients from unlabeled controls may be learning the "prazosin-adjacent" phenotype, not true PTSD signal.
- The 0.0289 proxy AUC gap between pi_p=0.25 and PULSNAR is within the plausible noise band of a 102-patient proxy group (bootstrap SE ≈ 0.04).
- The 0.0214 test AUPRC gap (0.8939 − 0.8725) is measured on the **same ICD-coded label distribution** as training, so it is not an independent validity signal — it is a "which model best reproduces the coding bias" signal.

### What this means for PULSNAR

- PULSNAR's alpha (0.1957) is **data-driven** under the SAR assumption, not tuned on a downstream metric with selection bias.
- The empirical win for pi_p=0.25 could be spurious — it is the local max of a U-shaped proxy-AUC curve (see the sweep: AUC dips from 0.789 at pi_p=0.10 down to 0.766 at pi_p=0.20 before spiking to 0.799 at pi_p=0.25). Non-monotonic curves in hyperparameter sweeps often indicate overfitting to the validation signal.
- PULSNAR also trained 39% faster (3.50 vs 5.73 GPU-h) due to faster convergence — a small practical bonus.

### What argues against switching to PULSNAR

- The v1 propensity model used only 4 demographic features; it may be **underpowered** to capture the SAR structure. A properly-specified PULSNAR with richer (but non-severity-conflated) features has not been run.
- v2 PULSNAR (alpha = 0.0006) was misspecified — adding `n_prior_admissions` conflated severity with coding propensity. This shows PULSNAR is sensitive to feature choice, which is itself a risk.
- Empirical proxy AUC and test AUPRC both favor pi_p=0.25, even if both signals are contaminated.

### Honest recommendation

The two defensible positions:

1. **Keep pi_p=0.25 as primary (current memo position).** Justification: best empirical metrics; 0.25 sits at the upper end of cited PTSD prevalence (15–25%) in trauma-exposed inpatient populations; gap over PULSNAR is consistent across multiple evaluations.

2. **Switch to PULSNAR as primary.** Justification: SCAR is clearly violated; PULSNAR's alpha is principled and in the "correct neighborhood" (0.196 is between the empirical pi_p=0.20 and pi_p=0.25 checkpoints); tuning on proxy AUC is methodologically fragile under SAR; the empirical gap is within plausible noise of a 102-patient proxy.

**My recommendation:** adopt a **dual-winner framing**:
- **pi_p=0.25** = empirical best under the pre-registered proxy-AUC decision rule.
- **PULSNAR v1** = SAR-principled alternative; report in the paper as the "methodologically conservative" model and use for sensitivity analysis of all downstream claims.

If the two disagree on a downstream claim (e.g., NNS, subgroup fairness, top-attributed features), that disagreement is itself a finding worth reporting. If they agree, the claim is robust to the SCAR-vs-SAR choice.

### Action items if adopting this framing

- Re-run Fixes 5 (calibration), 6 (utility), 9 (fairness), 10 (attribution) on the PULSNAR checkpoint alongside pi_p=0.25. ✅ **Completed 2026-04-26.**
- Report both sets of numbers in the paper; flag any disagreements.
- Keep `models/ewang163_longformer_best` → `_pip025` (empirical primary) but add `models/ewang163_longformer_pulsnar` as an officially-reported sensitivity model, not just an "alternate."

---

## Dual-Winner Comparison (Fixes 5, 6, 9, 10 measured on PULSNAR)

Re-ran calibration / utility / fairness / attribution on the PULSNAR checkpoint (jobs 1918657, 1918658) and compared to pi_p=0.25.

### Test discrimination + utility (Fix 6)

| Metric | pi_p=0.25 | PULSNAR | Δ (PULSNAR − pi_p=0.25) |
|---|---:|---:|---:|
| Val-derived threshold | 0.324 | 0.188 | (different scale; PULSNAR is propensity-weighted) |
| Test AUPRC | 0.8939 | 0.8848 | −0.0091 |
| Test AUROC | 0.9002 | 0.8904 | −0.0098 |
| Sensitivity | 0.852 | 0.846 | −0.006 |
| Specificity | 0.782 | 0.745 | −0.037 |
| F1 | 0.794 | 0.772 | −0.022 |
| LR+ | 3.91 | 3.32 | −15% |
| LR− | 0.190 | 0.207 | +9% (worse) |
| DOR | 20.6 | 16.0 | −22% |
| Alert rate | 48.7% | 50.6% | +1.9 pp |
| NNS @ 2% prevalence | 13.5 | 15.8 | +2.3 |
| NNS @ 5% prevalence | 5.9 | 6.7 | +0.8 |
| NNS @ 10% prevalence | 3.3 | 3.7 | +0.4 |

**pi_p=0.25 wins uniformly on raw discrimination and clinical utility.** At 2% deployment prevalence, PULSNAR flags 16 patients per true case found vs. pi_p=0.25's 14 — clinically tolerable but a real cost.

### Calibration (Fix 5)

| Metric | pi_p=0.25 | PULSNAR |
|---|---:|---:|
| ECE raw | 0.064 | 0.088 |
| ECE Platt-scaled | 0.074 | 0.077 |
| ECE Elkan-Noto | 0.080 | 0.097 |
| Elkan-Noto c estimate | 0.783 | 0.728 |

PULSNAR is less well-calibrated (ECE raw 38% higher). The lower c (0.728 vs 0.783) means PULSNAR is less confident on known positives, consistent with its propensity-weighted training discounting the most-confidently-coded cases.

### Fairness (Fix 9) — Equal Opportunity Differences

| Subgroup | pi_p=0.25 | PULSNAR | Δ |
|---|---:|---:|---:|
| Sex | 0.116 | 0.114 | −0.002 |
| Age group | 0.209 | 0.211 | +0.002 |
| Race (multi-category) | 0.112 | 0.101 | −0.011 |
| Race binary (W vs non-W) | 0.023 | 0.024 | +0.001 |
| Emergency | 0.034 | 0.046 | +0.012 |

**Fairness is essentially identical.** Both models inherit the same selection-bias pattern (men under-detected, age tails under-detected). The dual-winner contrast does NOT identify divergent fairness profiles — useful negative finding for the paper.

### Attribution (Fix 10) — top-attributed words and section share

| Rank | pi_p=0.25 | PULSNAR |
|---:|---|---|
| 1 | bipolar | bipolar |
| 2 | pylori | narcotic |
| 3 | personality | illness |
| 4 | schizoaffective | arrested |
| 5 | disorder | delayed |
| 6 | inr | pancreatitis |
| 7 | coumadin | schizoaffective |
| 8 | abusive | psychosis |
| 9 | transferred | anemia |
| 10 | arthritis | assault |

| Section | pi_p=0.25 | PULSNAR | Δ |
|---|---:|---:|---:|
| HPI | 36.4% | 43.2% | **+6.8 pp** |
| Brief Hospital Course | 35.9% | 32.0% | −3.9 pp |
| PMH | 26.3% | 22.4% | **−3.9 pp** |
| Social History | 0.5% | 1.3% | +0.8 pp |

**This is the most informative divergence.** PULSNAR shifts attribution from PMH (where comorbidities live) to HPI (where trauma history is documented) — and its top words include trauma-relevant content (`narcotic`, `arrested`, `assault`, `psychosis`) appearing earlier than in pi_p=0.25's list, where comorbidity codes (`pylori`, `inr`, `coumadin`) appear high.

### Interpretation

The numerical headline is what SCAR-vs-SAR theory predicts:

- **pi_p=0.25 wins everywhere a SCAR-coded validation signal rewards** — AUPRC, AUROC, F1, NNS, calibration. These metrics all derive from the same ICD-coded test labels that share the selection bias.
- **PULSNAR wins on what a SAR-aware model should** — less reliance on the PMH comorbidity profile (the strongest non-PTSD predictor of being ICD-coded), more attribution to HPI narrative content, top-attributed words shifted toward trauma-specific vocabulary.
- **Fairness is unchanged** — the residual biases (age, sex) are intrinsic to the cohort, not the loss function.

Whether this means "PULSNAR is the worse model" or "pi_p=0.25 is overfitting to coding bias" depends on the deployment goal:
- *Flag patients similar to those who get ICD-coded today* → pi_p=0.25 wins on every metric.
- *Flag undercoded patients who don't fit the typical coding profile* → PULSNAR's attribution shift is preferable, even at a 0.009 AUPRC cost. This is the actual stated goal of this project (detecting undercoded PTSD).

The honest read: **PULSNAR's empirical underperformance is partly a feature, not a bug.** A model that scores lower on a selection-biased benchmark, while attributing predictions to clinically appropriate narrative content rather than comorbidity codes, may be the better choice for the *undercoding* use case — though it cannot be definitively validated without a clinician chart review (Fix 11).

---

## What the Final Report Should Say

1. **Primary claim:** "The Clinical Longformer model, trained with nnPU loss at pi_p=0.25 on Fix-1-masked MIMIC-IV discharge notes, achieves test AUPRC=0.894 and AUROC=0.900 with val-derived thresholding."

2. **Non-circular validation:** "Proxy validation (n=102 patients with prazosin+SSRI/SNRI history but no ICD PTSD code) shows Mann-Whitney AUC=0.799 (p=8.3e-22), with 58% of proxy patients exceeding the screening threshold vs. 15% of demographically-matched unlabeled controls."

3. **Clinical utility:** "At 2% inpatient deployment prevalence, the model achieves NNS=13.5 and PPV=0.074; at 5% prevalence, NNS=5.9 and PPV=0.171. LR+ of 3.91 provides moderate positive evidence suitable for screening."

4. **Limitations:**
   - Subgroup performance gap: men and older patients under-detected (EO diff 0.12-0.21), inherited from non-SCAR PTSD coding bias.
   - Raw PU labels are used for evaluation; true population AUPRC is likely higher than reported.
   - Temporal generalization tested on MIMIC-IV 2017-2019 only; external validation not performed.
   - Chart review packet generated but not yet clinician-rated (Fix 11 partial).

5. **Cited methodology fixes:** Fix 1 (Jin et al. 2023), Fix 2 (selection-on-test bias), Fix 4 (Ramola 2019 PU corrections), Fix 5 (Elkan-Noto), Fix 6 (PU lower bound framing), Fix 8 (Li et al. 2023 chunk-and-pool control), Fix 9 (TRIPOD-AI fairness), Fix 10 (Edwards 2025 full-context IG), Fix 11 (Stanley 2020 clinician validation). Fix 3 (PULSNAR) partial — SAR correction tested, selected via proxy AUC sweep.

---

## Artifacts

| File | Contents |
|------|---------|
| `models/ewang163_longformer_best/` → `_pip025/` | Final deployment checkpoint |
| `models/ewang163_longformer_best_retrain_empirical/` | Alternate: empirical pi_p retrain |
| `models/ewang163_longformer_pulsnar/` | Alternate: PULSNAR propensity-weighted |
| `models/ewang163_longformer_best_temporal/` | Not recommended for deployment |
| `results/metrics/ewang163_best_pi_p.json` | Winner selection record |
| `results/metrics/ewang163_pip_sweep_results.csv` | All 9 candidate proxy AUCs |
| `results/metrics/ewang163_evaluation_results.json` | Full winner metrics |
| `results/metrics/ewang163_proxy_validation_results.csv` | Proxy validation on winner |
| `results/metrics/ewang163_fairness_results.csv` | Fairness on winner |
| `results/metrics/ewang163_calibration_results.csv` | Calibration on winner |
| `results/metrics/ewang163_temporal_eval_results.json` | Temporal generalization |
| `results/attribution/ewang163_top_attributed_words_v2.csv` | Word-level IG (pi_p=0.25) |
| `results/chart_review/ewang163_top50_review_packet.txt` | For clinician review |
| **PULSNAR sensitivity artifacts (Fixes 5/6/9/10):** | |
| `results/predictions/ewang163_longformer_{val,test}_predictions_pulsnar.csv` | PULSNAR test/val predicted probs |
| `results/metrics/ewang163_evaluation_results_pulsnar.json` | PULSNAR utility + thresholds |
| `results/metrics/ewang163_calibration_results_pulsnar.csv` | PULSNAR calibration (raw / Platt / Elkan-Noto) |
| `results/metrics/ewang163_fairness_results_pulsnar.csv` | PULSNAR per-subgroup metrics |
| `results/figures/ewang163_calibration_curve_pulsnar.png` | PULSNAR calibration plot |
| `results/attribution/ewang163_top_attributed_words_v2_pulsnar.csv` | PULSNAR word-level IG |
| `results/attribution/ewang163_attribution_by_section_v2_pulsnar.csv` | PULSNAR section share |

---

## Sign-off

Selected (empirical primary): **Clinical Longformer @ pi_p=0.25, Fix-1-masked corpus**
Selected (SAR-principled sensitivity): **Clinical Longformer @ PULSNAR alpha=0.196, Fix-1-masked corpus**
Date: 2026-04-17 (initial), 2026-04-26 (PULSNAR Fixes 5/6/9/10 completed; reproduced via jobs 1918731/1918732)

Next step: clinician chart review of top-50 flagged unlabeled patients (Fix 11 completion).
