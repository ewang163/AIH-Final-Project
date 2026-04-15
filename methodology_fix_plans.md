# Methodology Fix Plans — PTSD NLP Project

Evidence-based methodology improvements derived from a code-level review of all
scripts in `scripts/01_cohort/`, `scripts/02_corpus/`, `scripts/03_training/`,
and `scripts/04_evaluation/`, cross-referenced against PubMed literature.

Each fix lists: the problem, the published evidence, the concrete code change,
and the expected impact on validity.

---

## Key Literature Anchors

These are the papers the fixes below draw on. Cite each in the final write-up
where the corresponding methodology is justified.

| Citation | DOI / PMID | Why it matters here |
|---|---|---|
| Jin Y et al. 2023, JAMIA — *Learning from undercoded clinical records for automated ICD coding* | [10.1093/jamia/ocac230](https://doi.org/10.1093/jamia/ocac230) | Direct precedent: PU learning with reweighting on MIMIC-III for the *exact* undercoded-ICD problem. Outperforms supervised baselines as missing-label ratio grows. The single closest published paper to this project. |
| Li Y et al. 2023, JAMIA — *A comparative study of pretrained language models for long clinical text* | [10.1093/jamia/ocac225](https://doi.org/10.1093/jamia/ocac225) | Foundational paper for `yikuan8/Clinical-Longformer`. Establishes that long-sequence transformers beat ClinicalBERT specifically because discharge notes routinely exceed 512 tokens. Justifies architecture choice and the chunked-BERT fairness fix. |
| Kumar P & Lambert CG 2024, *PeerJ Comput Sci* — *PULSNAR: class proportion estimation without the SCAR assumption* | [10.7717/peerj-cs.2451](https://doi.org/10.7717/peerj-cs.2451) | The methodological backbone for fixing the SCAR violation. Provides PULSCAR (SCAR) and PULSNAR (SAR) estimators for π_p with calibrated probabilities. Reference implementation on GitHub. |
| Ramola R, Jain S, Radivojac P 2019, *Pac Symp Biocomput* — *Estimating classification accuracy in PU learning* | [PMID 30864316](https://pubmed.ncbi.nlm.nih.gov/30864316/) | Shows PU performance estimates are *wildly* inaccurate without correct π_p and noise estimates. Provides correction formulas for AUPRC, AUROC, sensitivity, and PPV in the PU setting. |
| Stanley IH et al. 2020, *J Psychiatr Res* — *PTSD symptoms among trauma-exposed adults admitted to inpatient psychiatry for suicide-related concerns* | [10.1016/j.jpsychires.2020.12.001](https://doi.org/10.1016/j.jpsychires.2020.12.001) | Empirical justification for the undercoding hypothesis: trauma-exposed inpatients admitted for suicide-related concerns are ~4× more likely to screen PCL-5 positive than to have a chart PTSD diagnosis. Use this to ground the prevalence claim. |
| Bajor LA, Balsara C, Osser DN 2022, *Psychiatry Res* — *An evidence-based approach to psychopharmacology for PTSD — 2022 update* | [10.1016/j.psychres.2022.114840](https://doi.org/10.1016/j.psychres.2022.114840) | Justifies the prazosin + SSRI/SNRI proxy. Prazosin is first-line for PTSD nightmares; SSRIs are second-line for non-sleep symptoms. Cite this for the proxy construct definition. |
| Kennedy CJ et al. 2024, *JAMA Psychiatry* — *Predicting Suicides Among US Army Soldiers After Leaving Active Service* | [10.1001/jamapsychiatry.2024.2744](https://doi.org/10.1001/jamapsychiatry.2024.2744) | Evaluation playbook precedent for high-stakes psychiatric ML in MIMIC-adjacent settings. Uses the same calibration + DCA + recall-anchored threshold suite. Cite for the evaluation methodology. |
| Edwards ER et al. 2025, *Transl Psychiatry* — *Improving explainability of post-separation suicide attempt prediction models* | [10.1038/s41398-025-03248-z](https://doi.org/10.1038/s41398-025-03248-z) | Modern precedent for explainability requirements in psychiatric ML. Frames model as decision-support, not diagnostic — the right framing for our undercoded-PTSD use case. |

Optional secondary citations to consider once the main fixes are in:

- **Elkan & Noto 2008** (PU learning under SCAR; original `c = P(s=1|y=1)` correction). Not in PubMed but the reference implementation for the calibration fix below. Cite as the foundational SCAR paper alongside Kiryo et al. 2017.
- **Bekker & Davis 2020** (SAR-PU). Cite as the methodological alternative to PULSNAR for the propensity-corrected loss.
- **Sundararajan, Taly & Yan 2017** (Integrated Gradients axioms). Already cited in CLAUDE.md; keep.
- **Jain & Wallace 2019** (Attention is not Explanation). Already cited in CLAUDE.md; keep.

---

## Fix 1 — Apply PTSD masking to pre-diagnosis training notes (HIGHEST PRIORITY)

### Problem
`scripts/02_corpus/ewang163_ptsd_corpus_build.py:66-72` only masks the
*fallback* PTSD+ notes. Pre-diagnosis notes (the primary training group,
n=2,492) pass through verbatim. The CLAUDE.md justification — "the PTSD
label cannot appear in notes from before the patient was ever coded" —
is a claim about ICD codes, not narrative text. Clinicians document PTSD
history in HPI/PMH well before they code it; "h/o PTSD from MVA 2012" can
sit in a 2014 note that predates the first F43.1 code in 2016.

### Evidence
Jin et al. 2023 ([10.1093/jamia/ocac230](https://doi.org/10.1093/jamia/ocac230))
explicitly identifies *annotation noise from undercoded records* as the
dominant failure mode for supervised models on MIMIC, and shows that any
residual leakage between text and "code-derived" labels biases the learned
representation. Their PU formulation only works because their text features
are independent of the missing labels.

### Concrete change
1. Audit first: count how many pre-dx notes match the existing `MASK_RE`:
   ```python
   df_predx = df[(df.group=='ptsd_pos') & df.is_prediagnosis]
   hit_rate = df_predx.note_text.str.contains(MASK_RE, case=False).mean()
   print(f"Pre-dx leakage hit rate: {hit_rate:.1%}")
   ```
   Expect 5–30%. If non-trivial, this single bug invalidates the primary
   training set.
2. In `corpus_build.py`, change line 67 from `fallback_mask = ... & (~is_prediagnosis)`
   to `all_pos_mask = (df['group'] == 'ptsd_pos')` and apply masking to **all**
   PTSD+ notes, primary and fallback alike.
3. Re-train Longformer and re-run all evaluation scripts.
4. Document the leakage hit rate in the paper's methods section.

### Expected impact
Closes the largest leakage path. Likely depresses raw AUPRC by 3–8 points
but the post-fix number is the only one a reviewer will believe.

---

## Fix 2 — Sweep the class prior π_p instead of inferring it from the empirical positive fraction

### Problem
`scripts/03_training/ewang163_ptsd_train_longformer.py:209` sets
`pi_p = n_pos / (n_pos + n_unl) ≈ 0.25`, which is the empirical labeled
fraction in the matched cohort. In Kiryo et al. 2017's nnPU theory, π_p is
**P(Y=1) in the population the unlabeled set is drawn from** — i.e., the
true latent positive fraction among unlabeled patients, not the labeled
fraction. The 3:1 matching artificially inflates the labeled fraction;
true π_p is unknown and is exactly the quantity the project is trying to
estimate.

### Evidence
Kumar & Lambert 2024 ([10.7717/peerj-cs.2451](https://doi.org/10.7717/peerj-cs.2451))
state directly: "in many real-world applications, such as healthcare,
positives are not SCAR (e.g., severe cases are more likely to be
diagnosed), leading to a poor estimate of α and poor model calibration,
resulting in an uncertain decision threshold for selecting positives."
Their PULSCAR/PULSNAR estimators are designed for exactly this case.

Ramola et al. 2019 ([PMID 30864316](https://pubmed.ncbi.nlm.nih.gov/30864316/))
formally show that PU performance estimates are wildly inaccurate without
correct π_p, and provide correction formulas for AUPRC/AUROC/sensitivity/PPV.

### Concrete change
1. Replace the single training run with a sweep over
   `π_p ∈ {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25}`. Each run takes
   ~6 GPU-hours; the full sweep is ~2 GPU-days on one card.
2. Pass `π_p` as a CLI argument to the training script and log it in
   `models/ewang163_longformer_best/training_config.json`.
3. For each trained model, evaluate on the proxy validation set
   (Mann-Whitney AUC and fraction-above-threshold) — these are the only
   PU-uncontaminated metrics. Pick the best by proxy AUC.
4. Report the full sweep in a supplementary table; report the best in the
   main paper.
5. If time permits, replace the manual sweep with PULSCAR
   (https://github.com/unmtransinfo/PULSNAR) for a principled estimator.

### Expected impact
The project's central methodological choice becomes defensible. Reviewers
will ask "how did you pick π_p?" and the answer "swept it and chose by
proxy validation AUC, with PULSCAR as a sanity check" is the right answer.

---

## Fix 3 — Address the SCAR violation with a propensity-weighted loss or PULSNAR

### Problem
CLAUDE.md acknowledges (line 233) that PTSD coding is biased toward
younger women with prior psychiatric contact — i.e., positives are *not*
selected completely at random. Vanilla nnPU assumes SCAR. This contradiction
is the project's biggest methodological gap.

### Evidence
Kumar & Lambert 2024 ([10.7717/peerj-cs.2451](https://doi.org/10.7717/peerj-cs.2451))
introduce PULSNAR — a divide-and-conquer SAR-aware PU estimator — for
exactly this scenario. Bekker & Davis 2020 (SAR-PU; not on PubMed but
referenced from the PULSNAR paper) provide the propensity-weighted nnPU
formulation, which is simpler to drop into existing code.

### Concrete change
**Option A (lighter):** propensity-weighted nnPU.
1. Fit a logistic regression or gradient boosted model
   `e(x) = P(coded | age, sex, n_prior_admissions, has_prior_psych,
   insurance, emergency)` on the labeled data.
2. In `pu_loss()` (line 57 of `train_longformer.py`), reweight the positive
   loss term by `1 / e(x_i)` per positive sample. The unlabeled term stays
   the same.
3. Clip propensities to `[0.05, 0.95]` to prevent variance explosion.

**Option B (heavier, recommended for final paper):** use PULSNAR end-to-end.
1. Install from https://github.com/unmtransinfo/PULSNAR.
2. Replace the custom `pu_loss()` with PULSNAR's training procedure.
3. PULSNAR also provides calibrated probabilities — this also fixes Fix 5
   below.

### Expected impact
Removes the methodological gap reviewers will hit hardest. Empirically,
SAR-correction usually shifts subgroup performance: race minorities and
older men (who are coded less often per ground-truth case) should see
higher predicted probabilities post-correction.

---

## Fix 4 — Move threshold selection from test to validation

### Problem
`scripts/04_evaluation/ewang163_ptsd_evaluate.py:191` calls
`threshold_at_recall(probs, labels, 0.85)` on the **test set**, then
reports sensitivity/specificity/F1/PPV at that test-tuned threshold.
This is selection-on-test and inflates F1/specificity.
The proxy validation script (`proxy_validation.py:117`) inherits this
biased threshold via `evaluation_results.json`.

### Evidence
Standard ML methodology, well-documented in TRIPOD-AI and PROBAST
guidelines. Kennedy et al. 2024
([10.1001/jamapsychiatry.2024.2744](https://doi.org/10.1001/jamapsychiatry.2024.2744))
explicitly select thresholds on a held-out development set, not on test.

### Concrete change
1. In `evaluate.py`, run inference on the validation split first, compute
   `threshold_recall_85_val` from val probabilities and val labels.
2. Save it to `evaluation_results.json` as the canonical threshold.
3. Use it (not the test-derived threshold) for all test sens/spec/F1/PPV
   reporting.
4. Update `proxy_validation.py` to load the same val-derived threshold.

### Expected impact
~1 hour of code change. Eliminates an obvious peer-review red flag.
Headline F1 may drop 1–3 points but the number is honest.

---

## Fix 5 — Calibrate probabilities against PU-corrected labels, not raw PU labels

### Problem
`scripts/04_evaluation/ewang163_ptsd_calibration.py:115-117` fits Platt
scaling on validation labels that treat unlabeled patients as 0. The
calibration target is the ICD-coded prevalence, not the true PTSD
prevalence. Reported ECE looks good but is calibrated to the wrong base
rate.

### Evidence
Ramola et al. 2019 ([PMID 30864316](https://pubmed.ncbi.nlm.nih.gov/30864316/))
provide explicit correction formulas for PU performance metrics including
calibration. Kumar & Lambert 2024
([10.7717/peerj-cs.2451](https://doi.org/10.7717/peerj-cs.2451))
note that PU calibration without prior correction is one of the main
sources of "uncertain decision threshold for selecting positives" in
healthcare PU applications. The Elkan & Noto (2008) constant
`c = P(s=1 | y=1)` is the standard correction.

### Concrete change
1. Estimate `c` either from PULSCAR (Fix 3 Option B), or via the proxy
   group: `c ≈ mean(model_prob_on_known_positives)` evaluated on a
   held-out subset of labeled positives.
2. After running Platt scaling, divide the calibrated probabilities by `c`
   (clipped to `[0, 1]`) before computing ECE.
3. Report both raw and Elkan-Noto-corrected ECE in the paper.
4. Or, alternatively, calibrate against the proxy patients (n=102 with
   notes) and a small (n≈100) chart-reviewed unlabeled subset — this gives
   you a cleaner calibration target at the cost of small-n CI width.

### Expected impact
Calibration becomes interpretable as "the predicted P(PTSD=1) approximates
the true P(PTSD=1)" rather than "the predicted P(coded=1) approximates the
true P(coded=1)". This is a substantial reframing that aligns the model's
output with the project's central goal.

---

## Fix 6 — Recompute all evaluation metrics as PU-aware lower bounds

### Problem
Every reported AUPRC, AUROC, specificity, F1, and PPV in `evaluate.py`
treats unlabeled test patients as confirmed negatives. A model that
correctly identifies the ~5–15% of unlabeled patients with hidden PTSD
will be penalized as producing false positives. These numbers are
**lower bounds** on true performance and should be labeled as such.

### Evidence
Ramola et al. 2019 ([PMID 30864316](https://pubmed.ncbi.nlm.nih.gov/30864316/))
quantify this bias formally and give corrected estimators that take
estimated π_p as input.

### Concrete change
1. In the paper, label all PU metrics as "PU lower bound (AUPRC_PU)".
2. Apply the Ramola correction formulas using the π_p chosen in Fix 2,
   reporting both raw and corrected metrics.
3. Treat the proxy validation Mann-Whitney AUC as the *only* metric that
   isn't PU-contaminated, and elevate it to a co-headline result.

### Expected impact
Reframes the entire results section. AUPRC of 0.4 is no longer "the model
is mediocre" — it becomes "the model achieves AUPRC ≥ 0.4 against a label
set that systematically penalizes correct detections of the undercoded
positives." This is the right interpretation.

---

## Fix 7 — Add temporal validation (pre-2015 train, post-2015 test)

### Problem
MIMIC-IV spans 2008–2019. PTSD coding habits changed substantially after
DSM-5 (2013) reclassified PTSD out of "anxiety disorders" and after
VA/DoD PTSD guideline updates (2017). Random patient-level splits ignore
this distribution shift entirely.

### Evidence
Kennedy et al. 2024
([10.1001/jamapsychiatry.2024.2744](https://doi.org/10.1001/jamapsychiatry.2024.2744))
use temporal splits (training on earlier years, testing on later) as
the closest available proxy for prospective validation in retrospective
EHR studies. Their model showed AUC degradation of ~5 points across a
5-year temporal gap — a meaningful effect that random splits hide.

### Concrete change
1. In `scripts/02_corpus/ewang163_ptsd_splits.py`, add an alternative
   split function that uses each patient's `index_admittime` to assign
   pre-2015 to train and post-2015 to test.
2. Stratify by `ptsd_label` within each time window.
3. Re-train Longformer on the pre-2015 split, evaluate on post-2015 test.
4. Report both random and temporal split results in the paper.

### Expected impact
This is the only generalization signal available within MIMIC-IV. A
model that holds up across the temporal gap is genuinely useful; one
that doesn't is a useful negative finding.

---

## Fix 8 — Make the BioClinicalBERT comparison fair via chunk-and-pool

### Problem
`evaluate.py:426` runs BioClinicalBERT at `max_len=512`, truncating
discharge notes that average several thousand tokens. Comparing this
to Longformer at 4096 tokens isn't an architecture comparison — it's a
context-length comparison. Li et al. 2023 already established that
context length matters for clinical text; re-establishing it here adds
nothing.

### Evidence
Li Y et al. 2023, JAMIA
([10.1093/jamia/ocac225](https://doi.org/10.1093/jamia/ocac225)) — the
foundational Clinical-Longformer paper — explicitly shows that
ClinicalBERT at 512 tokens loses to long-context transformers on long
clinical text. To say anything new about Clinical-Longformer vs.
BioClinicalBERT we have to control context length.

### Concrete change
1. In `train_bioclinbert.py` and `evaluate.py`, replace the single 512-token
   inference call with **chunk-and-pool**:
   - Split each note into overlapping 512-token windows (stride 256).
   - Run inference on each window.
   - Aggregate via max-pool or mean-pool over chunk probabilities.
2. Report both single-chunk (truncation) and chunk-and-pool BioClinicalBERT
   numbers. The chunk-and-pool number is the fair comparison.

### Expected impact
Whichever way the comparison goes, the result is now interpretable. If
chunk-and-pool BioClinicalBERT closes the gap, the project's contribution
is "long-context inference matters, not Longformer specifically." If the
gap remains, the contribution is "Longformer's pretraining and
architecture both help."

---

## Fix 9 — Replace AUPRC subgroup analysis with calibration-in-the-large + bootstrap CIs

### Problem
`evaluate.py:545-557` reports per-race AUPRC. With Asian patients at
~2.4% of PTSD+ (n≈14 in the test split), the AUPRC point estimate is
noise. Reporting it as if it's a comparable number to White (n>>100) is
misleading.

### Evidence
The fairness-in-clinical-ML literature consistently flags subgroup AUPRC
as unreliable below n_pos ≈ 30 per group. The Kennedy et al. 2024 JAMA
Psychiatry paper ([10.1001/jamapsychiatry.2024.2744](https://doi.org/10.1001/jamapsychiatry.2024.2744))
uses calibration-within-group and net benefit at fixed thresholds for
fairness reporting, not subgroup AUPRC.

### Concrete change
1. For each demographic subgroup, report:
   - **Calibration-in-the-large**: `mean(predicted_prob) - mean(observed_label)`
     — a single number with a Wilson CI.
   - **Equal opportunity difference**: difference in recall at the fixed
     val-derived threshold across groups.
   - **Bootstrap 95% CI on AUPRC** (1,000 resamples) — only report the AUPRC
     itself if the CI width is < 0.15.
2. Group race into White vs. non-White for the primary fairness contrast,
   with a supplementary per-race breakdown.
3. Add the same analysis at the end of `evaluate.py` and write the result
   into `results/metrics/ewang163_fairness_results.csv`.

### Expected impact
Fairness reporting becomes statistically defensible. Reviewers from
clinical journals will accept the methodology; reviewers from ML venues
will accept the rigor.

---

## Fix 10 — Validate Integrated Gradients on the same context length the model trains on

### Problem
`scripts/04_evaluation/ewang163_ptsd_attribution_v2.py:57` uses
`MAX_LEN_IG = 1024`. Training is at 4096. Attribution is computed only
on the first quarter of each note. If Brief Hospital Course (which
appears late) drives predictions, the attribution analysis misses it
entirely.

### Evidence
Edwards et al. 2025 (*Transl Psychiatry*,
[10.1038/s41398-025-03248-z](https://doi.org/10.1038/s41398-025-03248-z))
emphasize that attribution analyses must cover the same input the model
saw to support clinical interpretation, especially in long documents.

### Concrete change
1. Quantify what fraction of each test note's tokens lies in the first
   1024 vs. 1024–4096 tokens. If <50% of section content is in the first
   1024, the section attribution table is biased toward early sections by
   construction.
2. If memory permits, re-run IG at 4096 with `n_steps=20` and
   `internal_batch_size=1` (slower but covers the full input).
3. Otherwise, run IG twice per note (first half + second half) and stitch
   the attributions, weighted by the fraction of probability each half
   contributes.
4. Report section attribution stratified by which half of the note each
   section falls in.

### Expected impact
Section attribution becomes interpretable as a true measure of which
parts of the note the model actually uses. If HPI dominates regardless,
the IG result strengthens the clinical claim. If BHC actually matters,
the project gets a more nuanced clinical story.

---

## Fix 11 — Manual chart review of top-50 model-flagged unlabeled patients

### Problem
The proxy group is a weak external validator: n=102 with notes,
prazosin+SSRI selects nightmare-prominent treated PTSD (the easiest
phenotype to detect from text), and ~15–20% expected false-positive rate.
This is the only PU-uncontaminated validation in the project.

### Evidence
Stanley et al. 2020 ([10.1016/j.jpsychires.2020.12.001](https://doi.org/10.1016/j.jpsychires.2020.12.001))
established PTSD prevalence in trauma-exposed inpatients via the PCL-5,
not via ICD codes. The right validation for an undercoding model is a
clinician chart review of model-flagged patients, not an ICD/proxy
comparison.

### Concrete change
1. Take the top-50 model-flagged unlabeled patients (highest predicted
   `P(PTSD=1)` from the Longformer).
2. With advisor approval (and IRB review if required for the assignment),
   have a clinician (or the advisor) review the de-identified discharge
   notes and rate each as: probable PTSD, possible PTSD, unlikely PTSD.
3. Report **clinician-rated PPV at the top decile** as the primary
   external validation metric.
4. Even at n=50 this is the single most persuasive number the project can
   produce — it directly answers "does the model find real undercoded
   PTSD?"

### Expected impact
Transforms the project from "PU model with proxy validation" to "PU model
with clinician-validated decision support." The clinician-rated PPV is
the metric a JAMIA reviewer will weight most heavily.

---

## Priority Ranking

If only some fixes can be done, do them in this order:

| Priority | Fix | Cost | Why first |
|---|---|---|---|
| 1 | Fix 1 — mask pre-dx training notes | 0.5 day code + 1 GPU-day | Closes the biggest leakage path; cheapest fix with the largest validity impact |
| 2 | Fix 4 — move threshold to val | 1 hour | Eliminates an obvious selection bias for free |
| 3 | Fix 2 — π_p sweep | 2 GPU-days | Quantifies the project's most fragile design choice |
| 4 | Fix 6 — relabel metrics as PU lower bounds | 0.5 day | Reframes the entire results interpretation |
| 5 | Fix 7 — temporal split | 1 GPU-day | First real generalization signal |
| 6 | Fix 8 — chunk-and-pool BERT | 1 GPU-day | Fair architecture comparison |
| 7 | Fix 9 — fairness reporting | 0.5 day | Statistically defensible subgroup analysis |
| 8 | Fix 5 — Elkan-Noto calibration | 0.5 day | Calibration target becomes interpretable |
| 9 | Fix 3 — SAR-PU loss / PULSNAR | 2–3 days | Methodologically principled fix for SCAR violation |
| 10 | Fix 10 — IG at full context | 1 day | Attribution analysis becomes valid |
| 11 | Fix 11 — clinician chart review | 1 week wall-clock | Most persuasive single result possible |

Fixes 1, 2, and 11 together would fundamentally strengthen the project's
core claim. The rest are refinements to a study that's already
methodologically serious.
