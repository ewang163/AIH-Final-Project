# Methodology Decisions — PTSD NLP Project

Evidence-based methodology decisions, organised by topic. Each section
records the problem identified, the published evidence supporting the
chosen approach, the concrete code change, and where in the codebase the
implementation lives.

This document is the design rationale. The execution results are in
`ewang163_methodology_fixes_results.md`. The deployed model selection
narrative is in `ewang163_model_selection_memo.md`.

---

## Key Literature Anchors

These are the papers the methodology decisions below draw on. Cite each in
the final write-up where the corresponding methodology is justified.

| Citation | DOI / PMID | Why it matters here |
|---|---|---|
| Jin Y et al. 2023, JAMIA — *Learning from undercoded clinical records for automated ICD coding* | [10.1093/jamia/ocac230](https://doi.org/10.1093/jamia/ocac230) | Direct precedent: PU learning with reweighting on MIMIC-III for the *exact* undercoded-ICD problem. Outperforms supervised baselines as missing-label ratio grows. The single closest published paper to this project. |
| Li Y et al. 2023, JAMIA — *A comparative study of pretrained language models for long clinical text* | [10.1093/jamia/ocac225](https://doi.org/10.1093/jamia/ocac225) | Foundational paper for `yikuan8/Clinical-Longformer`. Establishes that long-sequence transformers beat ClinicalBERT specifically because discharge notes routinely exceed 512 tokens. Justifies architecture choice and the BERT chunk-pool comparison. |
| Kumar P & Lambert CG 2024, *PeerJ Comput Sci* — *PULSNAR: class proportion estimation without the SCAR assumption* | [10.7717/peerj-cs.2451](https://doi.org/10.7717/peerj-cs.2451) | The methodological backbone of the deployed model. Provides PULSCAR (SCAR) and PULSNAR (SAR) estimators for α with calibrated probabilities. Reference implementation on GitHub. |
| Bekker J & Davis J 2020, *Mach Learn* — *Learning from positive and unlabeled data: a survey* | [10.1007/s10994-020-05877-5](https://doi.org/10.1007/s10994-020-05877-5) | Foundational PU survey. Establishes the SCAR-vs-SAR distinction and the propensity-weighted nnPU formulation that PULSNAR builds on. |
| Kiryo R, Niu G, du Plessis MC, Sugiyama M 2017, NeurIPS — *Positive-unlabeled learning with non-negative risk estimator* | NeurIPS 2017 | The base nnPU loss. Used unmodified for the sensitivity-check Longformer (`train_longformer.py`); PULSNAR extends this loss with propensity reweighting. |
| Ramola R, Jain S, Radivojac P 2019, *Pac Symp Biocomput* — *Estimating classification accuracy in PU learning* | [PMID 30864316](https://pubmed.ncbi.nlm.nih.gov/30864316/) | Shows PU performance estimates are *wildly* inaccurate without correct α and noise estimates. Provides correction formulas for AUPRC, AUROC, sensitivity, and PPV in the PU setting. |
| Elkan C & Noto K 2008, *KDD* — *Learning classifiers from only positive and unlabeled data* | [10.1145/1401890.1401920](https://doi.org/10.1145/1401890.1401920) | Foundational SCAR paper. The `c = P(s=1|y=1)` correction is reused for absolute-probability calibration. |
| Stanley IH et al. 2020, *J Psychiatr Res* — *PTSD symptoms among trauma-exposed adults admitted to inpatient psychiatry for suicide-related concerns* | [10.1016/j.jpsychires.2020.12.001](https://doi.org/10.1016/j.jpsychires.2020.12.001) | Empirical justification for the undercoding hypothesis: trauma-exposed inpatients admitted for suicide-related concerns are ~4× more likely to screen PCL-5 positive than to have a chart PTSD diagnosis. Grounds the ~20% true-prevalence claim. |
| Bajor LA, Balsara C, Osser DN 2022, *Psychiatry Res* — *An evidence-based approach to psychopharmacology for PTSD — 2022 update* | [10.1016/j.psychres.2022.114840](https://doi.org/10.1016/j.psychres.2022.114840) | Justifies the prazosin + SSRI/SNRI proxy. Prazosin is first-line for PTSD nightmares; SSRIs are second-line for non-sleep symptoms. Cite for the proxy construct definition. |
| Kennedy CJ et al. 2024, *JAMA Psychiatry* — *Predicting Suicides Among US Army Soldiers After Leaving Active Service* | [10.1001/jamapsychiatry.2024.2744](https://doi.org/10.1001/jamapsychiatry.2024.2744) | Evaluation playbook precedent for high-stakes psychiatric ML in MIMIC-adjacent settings. Uses the same calibration + DCA + recall-anchored threshold suite this project follows. |
| Edwards ER et al. 2025, *Transl Psychiatry* — *Improving explainability of post-separation suicide attempt prediction models* | [10.1038/s41398-025-03248-z](https://doi.org/10.1038/s41398-025-03248-z) | Precedent for explainability requirements in psychiatric ML. Frames model as decision-support, not diagnostic — the right framing for the undercoded-PTSD use case. |
| Sundararajan M, Taly A, Yan Q 2017, ICML — *Axiomatic attribution for deep networks* | arXiv:1703.01365 | Integrated Gradients axioms (completeness, sensitivity, implementation invariance). Used for both Longformer (4,096-context) and BERT (512-context truncated + chunk-pool top-window) attribution. |
| Jain S, Wallace BC 2019, NAACL-HLT — *Attention is not explanation* | [10.18653/v1/N19-1357](https://doi.org/10.18653/v1/N19-1357) | Justifies using IG (not attention weights) for attribution. |

---

## 1. Universal PTSD-string masking

### Problem
The original design assumed pre-diagnosis notes (the primary training subsample,
n = 2,492 patients) could not contain explicit PTSD references because the
patient was not yet ICD-coded. This is a claim about ICD codes, not narrative
text — clinicians document PTSD history in HPI/PMH well before they code it
("h/o PTSD from MVA 2012" can sit in a 2014 note that predates the first
F43.1 code in 2016).

### Evidence
Jin Y et al. 2023 (JAMIA) explicitly identifies *annotation noise from
undercoded records* as the dominant failure mode for supervised models on
MIMIC. Their PU formulation only works because their text features are
independent of the missing labels.

### Decision
Apply universal PTSD-string masking to **all 5,950 PTSD+ notes** (pre-dx +
fallback), via case-insensitive regex `(?i)(post-traumatic|post\s+traumatic|posttraumatic|trauma-related\s+stress|ptsd|f43\.1|309\.81)` → `[PTSD_MASKED]`.

### Implementation
`scripts/02_corpus/ewang163_ptsd_corpus_build.py`. Audit confirms 8.6 % of
pre-dx notes contained explicit PTSD strings before masking.

### Post-implementation ablation
Post-hoc PTSD-string masking on the test set costs only −0.008 to −0.013 AUPRC
across all three text models (PULSNAR Longformer, BERT trunc, BERT chunk-pool)
— confirming the deployed transformers are not exploiting literal PTSD strings.

---

## 2. PULSNAR (SAR-aware PU loss) as the primary training method

### Problem
Plain Kiryo nnPU assumes SCAR (labeled positives are a uniform random sample
of all true positives). That assumption is **clearly violated here**: PTSD
coding is biased toward younger women, White patients, and patients with
prior psychiatric contact (the fairness analysis confirms exactly this
pattern). Tuning the class prior (α) on a downstream proxy metric is also
fragile because the proxy criterion (prazosin + SSRI/SNRI within 180 days)
itself selects a specific prescribing pattern that shares the SAR bias.

### Evidence
- Bekker & Davis 2020 (Mach Learn): foundational SAR-PU survey. Establishes
  that under SAR, propensity-weighted PU losses recover the true positive
  distribution; SCAR-tuned models reproduce the labelling bias.
- Kumar & Lambert 2024 (PeerJ CS): introduces PULSNAR — divide-and-conquer
  SAR-aware estimator for α with calibrated probabilities.

### Decision
PULSNAR end-to-end:

1. Propensity model `e(x) = P(coded | features)` via logistic regression on
   4 prior-admission demographic features (sex, age, emergency, medicaid).
   Clip to [0.05, 0.95].
2. Class-prior estimation via PULSNAR's xgboost-based KDE (`bin_method='rice'`,
   `bw_method='hist'`); fall back to PULSCAR (SCAR) then to empirical fraction.
   **α = 0.1957** for the 4-feature propensity model.
3. Modified nnPU loss with positives reweighted by `1/e(x)`, which up-weights
   labeled positives whose coding propensity is low (older men, no prior
   psychiatric contact) — exactly the under-detected population the screening
   tool is meant to surface.

**Why a 4-feature propensity model (not richer features):** A propensity model
that adds `n_prior_admissions` perfectly separates coded from uncoded
patients (coef +5.63), leaving PULSNAR no signal to detect hidden positives —
α collapses to 0.0006. This is a cohort-design artifact (Group 3 unlabeled
has index = first MIMIC-IV admission, so prior-admission count is always 0 by
construction). The same artifact dominates the structured baseline (coef +6.51).
The 4-feature propensity (demographics only) is the principled choice.

### Implementation
`scripts/03_training/ewang163_ptsd_train_pulsnar.py`. PULSNAR library installed
from `github.com/unmtransinfo/PULSNAR` with a small patch making the rpy2
dependency lazy (only triggered by non-default bandwidth methods).

### Sensitivity check
Plain Kiryo nnPU Longformer (`scripts/03_training/ewang163_ptsd_train_longformer.py`)
is retained as a sensitivity baseline.

---

## 3. Validation-derived operating threshold

### Problem
Earlier code computed the operating threshold (lowest threshold achieving
recall ≥ 0.85) on the test set itself. This is selection-on-test bias — it
inflates sensitivity / specificity / F1 by ~1–3 points.

### Evidence
Standard ML reporting practice (TRIPOD-AI, Kennedy et al. 2024 evaluation
suite). Threshold must be derived on a held-out validation set and frozen
before any test-set metrics are computed.

### Decision
For each model, compute the val-derived threshold via `threshold_at_recall(probs, labels, target_recall=0.85)` and use it for all downstream test-set
metrics (sensitivity, specificity, F1, NNS, alert rate, fairness, error
analysis, proxy validation).

### Implementation
- `scripts/04_evaluation/ewang163_ptsd_pulsnar_reeval.py` — Longformer val threshold = 0.188.
- `scripts/04_evaluation/ewang163_ptsd_bert_full_eval.py` — derives BERT
  thresholds **per inference mode** (truncated 0.976, chunk-pool 0.993 — the
  chunk-pool max-pool aggregation pushes the score distribution upward, so it
  needs its own threshold).
- `scripts/04_evaluation/ewang163_ptsd_evaluate.py` — produces structured + keyword val thresholds.

---

## 4. Elkan-Noto calibration correction

### Problem
PU loss outputs P(s=1|x) (probability of being coded), not P(y=1|x)
(probability of being PTSD+). For deployment, only the latter is interpretable
as a clinical probability.

### Evidence
Elkan & Noto 2008 (KDD): if c = P(s=1|y=1) is the labeling frequency, then
P(y=1|x) = P(s=1|x) / c. Estimate c as the mean raw model probability on
known positives in a held-out set.

### Decision
Compute c on validation positives, apply correction `clip(P(s=1|x) / c, 0, 1)` to
test probabilities, and report ECE for raw / Platt-scaled / Elkan-Noto variants on
10 equal-frequency bins with Wilson 95% CI per bin.

### Implementation
`scripts/04_evaluation/ewang163_ptsd_calibration.py` (Longformer-only legacy);
inline in `pulsnar_reeval.py` and `bert_full_eval.py` (canonical, both modes).

### Result
PULSNAR Longformer: c = 0.728 → ~27 % undercoding rate, consistent with
Stanley et al. (2020). BERT: c ≈ 0.99 because raw probabilities concentrate
near 1.0 — c is no longer informative as a labelling-frequency estimate; BERT
needs Platt scaling before any threshold-sensitive deployment.

---

## 5. PU-aware metric corrections (Ramola)

### Problem
Raw AUPRC / AUROC / precision computed against PU labels treat correct
detections of hidden positives as false positives, systematically
**under-estimating** model performance.

### Evidence
Ramola et al. 2019 (PSB) provide closed-form PU corrections parameterised by
the class prior α.

### Decision
Report **raw metrics labelled as "PU lower bounds"**, with Ramola corrections
alongside as the optimistic bound. With the PULSNAR-estimated α = 0.196:

- corrected_AUROC = (AUROC − 0.5α) / (1 − α) → 0.987 (vs raw 0.890)
- corrected_AUPRC = AUPRC / α → ceiling-clipped at 1.0
- corrected_precision = prec / (prec + (1 − prec)·(1 − α))

### Implementation
`scripts/04_evaluation/ewang163_ptsd_evaluate.py` (`ramola_corrected_metrics`).

---

## 6. Temporal generalization

### Problem
Random patient-level splits don't probe whether the model generalises across
the ICD-9 → ICD-10 transition (October 2015) or post-DSM-5 PTSD-criteria
reclassification.

### Evidence
Standard distribution-shift evaluation in clinical ML.

### Decision
Pre-2015-train / 2017–2019-test split using each patient's `anchor_year_group`
from `patients.csv`. Critically, MIMIC-IV applies per-patient random date
shifts (~100–200 years), so raw `admittime` is unusable for chronology — the
un-shifted `anchor_year_group` is the correct workaround.

### Implementation
`scripts/02_corpus/ewang163_ptsd_splits.py --temporal`;
`scripts/04_evaluation/ewang163_ptsd_temporal_eval.py`.

### Result
Temporal training **hurts** generalization. Random-split model on temporal
test loses only 0.002 AUPRC (random distribution is already representative of
late MIMIC-IV); temporal-trained model loses 0.044 AUPRC because it has
25 % less training data and misses richer post-2013 DSM-5-era coding patterns.
Random-split model recommended for deployment. Temporal model archived at
`models/ewang163_longformer_best_temporal/` but **not deployed**.

---

## 7. Fair architecture comparison: BERT chunk-and-pool

### Problem
Comparing Clinical Longformer (4,096 tokens) against BioClinicalBERT
(512 tokens, truncated) confounds two effects: model architecture and
inference context. A BERT model that *can* see the full note via chunk-pool
inference is the right comparator.

### Evidence
Li et al. 2023 (JAMIA) explicitly identify long-context inference as the
dominant driver of Longformer's lift on long clinical notes.

### Decision
Run BioClinicalBERT in **two inference modes** from the same trained
checkpoint:

- **Truncated** — first 512 tokens.
- **Chunk-and-pool** — overlapping 512-token windows with stride 256,
  max-pool of positive-class probabilities.

Each mode gets its own val-derived threshold and its own full downstream
evaluation suite (calibration, DCA, fairness, ablations, IG attribution,
proxy validation).

### Implementation
`scripts/04_evaluation/ewang163_ptsd_bert_full_eval.py` (both modes in one
job for compute efficiency).

### Result
Chunk-pool AUPRC = 0.878 vs truncated 0.858 (McNemar p = 4.5e-5 — chunk-pool
clearly preferred). PULSNAR Longformer (0.885) vs chunk-pool: McNemar
p = 0.107 — statistically indistinguishable.

---

## 8. Fairness reporting (calibration-in-the-large + EO + bootstrap CI)

### Problem
Per-subgroup AUPRC at small n_pos (e.g. Asian patients, n = 5 in test) has
huge variance and is misleading without confidence intervals.

### Evidence
TRIPOD-AI guidelines (Kennedy et al. 2024 follows the same pattern).

### Decision
Report per subgroup:

1. **Calibration-in-the-large** = mean(predicted) − mean(observed); Wilson
   CI on the observed proportion.
2. **Equal opportunity difference** (Hardt et al. 2016) = max(recall) − min(recall)
   at the val-derived threshold across subgroup levels.
3. **Bootstrap 95 % CI on AUPRC**: 1,000 resamples (`np.random.RandomState(42)`),
   percentile method (2.5th/97.5th). **Reliable iff CI width < 0.15.**

### Implementation
`scripts/04_evaluation/ewang163_ptsd_fairness.py` (Longformer-only legacy);
inline in `pulsnar_reeval.py` and `bert_full_eval.py` (canonical, all three
text models).

---

## 9. Integrated Gradients at full model context

### Problem
Earlier IG runs at MAX_LEN = 1024 only attributed the first quarter of each
note, missing Brief Hospital Course content that appears later in long
discharge notes.

### Evidence
Sundararajan et al. 2017 (ICML) IG axioms (completeness in particular) hold
only at the input length the model uses for inference. Edwards et al. 2025
(Transl Psychiatry) recommend matching IG context to inference context for
psychiatric ML attribution.

### Decision
Run IG at the model's full inference context:

- **Longformer:** 4,096 tokens, n_steps = 20, custom wrapper handling the
  hybrid local/global attention.
- **BioClinicalBERT:** 512 tokens — two slices: truncated input *and* the
  highest-scoring chunk-pool window per note (selects the window whose
  individual probability matches the max-pooled prediction).

Word-level aggregation merges contiguous BPE subwords with summed
attributions.

### Implementation
- `scripts/04_evaluation/ewang163_ptsd_attribution_v2.py` (Longformer)
- `scripts/04_evaluation/ewang163_ptsd_bert_attribution.py` (both BERT slices)

---

## 10. Cross-model agreement matrix

### Problem
Headline AUPRC differences between text models are small (~0.007 between
PULSNAR Longformer and BERT chunk-pool). Ranking metrics alone don't tell
you whether the models are picking up the same patients or different ones.

### Decision
Compute pairwise across all 5 deployed models:

- **McNemar's test** (continuity-corrected) on binary predictions at each
  model's val-derived threshold.
- **Cohen's κ** on binary predictions.
- **Pearson r** on continuous probabilities.
- **Top-quintile rank overlap** (Jaccard index of the top-20% by score).
- **Per-subgroup NNS** at deployment prevalences {1, 2, 5, 10 %}.
- **Max-pool / mean-pool ensemble probe** over rank-normalised
  probabilities, to test whether any combination beats the best individual
  model.

### Implementation
`scripts/04_evaluation/ewang163_ptsd_cross_model.py`.

### Results
- PULSNAR Longformer ⇄ BERT chunk-pool: McNemar p = 0.107 (n.s.), κ = 0.737,
  top-quintile overlap 83.5 %.
- BERT trunc ⇄ chunk-pool: Pearson r = 0.86 (same model, different aggregation).
- Anything ⇄ Structured/Keyword: κ < 0.15, overlap ~30–40 %.
- **No ensemble lift.** Max-pool 0.751, mean-pool 0.874 — neither beats the
  best individual model (0.885).

---

## Implementation status

All methodology decisions above are implemented in production. Per-decision
script paths are listed above. Execution results, runtime benchmarks, and
final headline metrics are in `ewang163_methodology_fixes_results.md`.
