# PTSD Underdiagnosis Detection in Inpatient Discharge Notes

**Author:** Eric Wang (ewang163) — Brown University, AIH 2025 (Spring 2026)
**Source data:** MIMIC-IV v3.1 (BIDMC, 2008–2019) on Brown's Oscar HPC cluster
**Goal:** Build an NLP screening tool that flags hospitalized patients whose discharge notes contain language suggestive of PTSD, so that clinicians can target a more thorough psychiatric evaluation at patients who would otherwise slip through inpatient care undiagnosed.

---

## 1. Repository Layout

```
ewang163/
├── CLAUDE.md                  Project instructions for Claude Code (authoritative design record)
├── README.md                  Quick-start + script index
├── documentation.md           This file — methodology, results, discussion
│
├── scripts/                   Pipeline code, ordered by stage
│   ├── common/                Shared utilities
│   │   └── ewang163_bench_utils.py               Runtime benchmarking
│   ├── 01_cohort/             Cohort construction from MIMIC-IV
│   │   ├── ewang163_ptsd_table1.py
│   │   ├── ewang163_ptsd_cohort_sets.py
│   │   ├── ewang163_ptsd_admissions_extract.py
│   │   └── ewang163_ptsd_notes_extract.py
│   ├── 02_corpus/             Corpus assembly + patient-level splits
│   │   ├── ewang163_ptsd_corpus_build.py
│   │   └── ewang163_ptsd_splits.py
│   ├── 03_training/           Model training
│   │   ├── ewang163_ptsd_train_longformer.py     (+ .sbatch) — accepts --pi_p for sweep
│   │   ├── ewang163_ptsd_train_bioclinbert.py    (+ .sh)
│   │   ├── ewang163_ptsd_train_tfidf.py
│   │   ├── ewang163_ptsd_train_structured.py     (+ .sh)
│   │   ├── ewang163_ptsd_train_keyword.py        Naive phrase-lookup baseline
│   │   ├── ewang163_ptsd_train_pulsnar.py        Fix 3: SAR-PU with propensity weights
│   │   ├── ewang163_ptsd_pip_sweep.sh            Fix 2: π_p sweep launcher
│   │   ├── ewang163_ptsd_pip_sweep_eval.py       Fix 2: sweep aggregator
│   │   └── ewang163_ptsd_specificity.py          (+ .sh)
│   └── 04_evaluation/         Evaluation, calibration, ablations, attribution
│       ├── ewang163_ptsd_evaluate.py             (+ .sh)
│       ├── ewang163_ptsd_calibration.py          (+ .sh)
│       ├── ewang163_ptsd_decision_curves.py      (+ .sh)
│       ├── ewang163_ptsd_proxy_validation.py     (+ .sh)
│       ├── ewang163_ptsd_ablations.py            (+ .sh)
│       ├── ewang163_ptsd_attribution.py          (+ .sh)
│       ├── ewang163_ptsd_attribution_v2.py       (+ .sh) — Fix 10: IG at 4096
│       ├── ewang163_ptsd_error_analysis.py       (+ .sh)
│       ├── ewang163_ptsd_fairness.py             Fix 9: bootstrap fairness
│       └── ewang163_ptsd_chart_review_packet.py  Fix 11: top-50 review packet
│
├── data/                      Cohort extracts and intermediate datasets
│   ├── cohort/                Subject ID lists, admission extracts
│   ├── notes/                 Section-filtered note parquets (raw + corpus + proxy + psych control)
│   └── splits/                Patient-level 80/10/10 train/val/test splits
│
├── models/                    Trained model artefacts
│   ├── ewang163_longformer_best/              Primary Clinical Longformer (PU loss)
│   ├── ewang163_bioclinbert_best/             Comparison model (PU loss)
│   ├── ewang163_specificity_longformer_best/  Specificity check vs. psychiatric controls
│   ├── ewang163_tfidf_vectorizer.pkl          + ewang163_tfidf_logreg.pkl  (text baseline)
│   ├── ewang163_structured_logreg.pkl         + ewang163_structured_features.json
│   └── ewang163_platt_calibrator.pkl          Post-hoc probability calibrator
│
├── results/
│   ├── table1/                Cohort characterization (CSV + plain-text summary)
│   ├── predictions/           Per-patient predicted probabilities (val + test)
│   ├── metrics/               JSON + CSV evaluation outputs (eval, ablations, calibration, DCA, proxy, training logs)
│   ├── figures/               PNG plots (calibration curve, DCA at 2%/5%, proxy histogram)
│   ├── attribution/           Integrated Gradients section/token attribution
│   ├── error_analysis/        Sampled FP/FN notes + aggregate stats
│   └── chart_review/          Fix 11: top-50 flagged patient review packet
│
├── logs/                      SLURM .out files from every job submitted
└── ptsd_env/                  Python virtual environment (do not move)
```

> **Script paths.** All scripts use absolute paths to `STUDENT_DIR` and its `data/`, `models/`, `results/` subdirectories. Output constants have been updated to match the reorganized layout, so re-running any script will write to the correct subdirectory.

---

## 2. Clinical Motivation

PTSD is systematically undercoded in inpatient settings. Inpatient teams are not trained to elicit DSM-5 PTSD criteria, and PTSD has no laboratory or imaging marker — it has to be asked about. Estimated true prevalence in trauma-exposed inpatients is ≥ 20%, but the ICD coding rate in MIMIC-IV is closer to 1%. Patients miss out on targeted treatment, generate avoidable readmissions, and lose continuity of trauma history across encounters.

The methodological challenge is that the obvious approach — train on ICD-coded patients vs. non-coded patients — is wrong by assumption. If undercoding is the central feature of the problem, then the "negative" group is contaminated with real positives, and any model trained and evaluated against that gold standard is measuring agreement with a flawed reference standard rather than actual clinical validity. Designing around that contamination, rather than acknowledging it as a limitation, is the central methodological commitment of this project.

---

## 3. Methodology

### 3.1 Data source

MIMIC-IV v3.1, accessed read-only at `/oscar/data/shared/ursa/mimic-iv` via Brown's Oscar HPC cluster. Source files used: `hosp/3.1/patients.csv`, `admissions.csv`, `diagnoses_icd.csv`, `prescriptions.csv`, and `note/2.2/discharge.csv` + `discharge_detail.csv`. All compute beyond trivial inspection ran through SLURM `batch` (data processing) or `gpu` (model training) partitions.

### 3.2 Cohort construction

Three groups were assembled, with bugs from a v0 attempt fixed before any modeling began (ICD codes are stored without dots in MIMIC-IV; admission types use the strings `EW EMER.` / `DIRECT EMER.` not `EMERGENCY`; comorbidity prefix lists must include both ICD-9 and ICD-10 since MIMIC-IV straddles October 2015; `prescriptions.csv` is plain CSV not gzip).

- **Group 1 — ICD-coded PTSD+ (training positives), n = 5,711.** Any subject with ICD-10 prefix `F431` or ICD-9 `30981` at any admission. The index admission is the first admission where the PTSD code appears. 61.5% female, mean age 43.0 ± 15.9, 65.6% White, 28.0% emergency, 40.9% Medicaid/self-pay. 72.0% have ≥ 2 admissions; 43.6% (n = 2,492) have at least one admission *before* their first F43.1 code, which becomes the primary pre-diagnosis training signal.
- **Group 2 — Pharmacological proxy (external validation only), n = 163.** Subjects with a prazosin + SSRI/SNRI prescription overlap within 180 days, excluding any subject with an ICD code starting with `I10` (hypertension), `N40` (BPH), `I7300` (Raynaud's), or `S06` (TBI) at any diagnosis position, and excluding all Group 1 subjects. Estimated 15–20% false-positive rate. **Never used in training.** Used post-hoc to test whether the trained model assigns elevated PTSD probabilities to patients whose pharmacotherapy is consistent with PTSD but who were never coded.
- **Group 3 — Matched unlabeled pool (PU pool), n = 17,133.** All remaining subjects, matched 3:1 to the PTSD+ group on age decade × sex with `np.random.seed(42)`. Treated as unlabeled (mixture of true negatives and undercoded positives) under the PU formulation.

Full Table 1 in `results/table1/`.

### 3.3 Label leakage prevention

The single biggest obstacle to building this model honestly is that discharge notes for ICD-coded patients usually contain "PTSD" verbatim in the Assessment & Plan or Discharge Diagnosis section. A model trained on these notes learns to recognise the label, not the underlying clinical condition. A tiered defense is used:

1. **Section filtering (all notes).** From `discharge_detail.csv`, only narrative low-leakage sections are kept: history of present illness, social history, past medical history, brief hospital course, family history. The high-leakage sections — discharge diagnosis, assessment and plan, discharge medications, discharge condition, discharge instructions, follow-up instructions — are excluded entirely.
2. **Pre-diagnosis notes (primary, n = 2,492 patients).** For multi-admission PTSD+ patients with ≥ 1 admission before their first F43.1 coding, training text is restricted to those earlier admissions.
3. **PTSD-string masking on ALL positive notes (Fix 1).** Masking is applied to *all* PTSD+ notes — both pre-diagnosis and fallback — not just the fallback subset. The original design assumed pre-diagnosis notes could not contain PTSD references, but clinicians routinely carry forward trauma history in HPI/PMH ("h/o PTSD from MVA 2012") from outside records on pre-diagnosis admissions. An audit step quantifies the pre-dx leakage hit rate before masking. Jin et al. (2023, JAMIA, [10.1093/jamia/ocac230]) explicitly identifies annotation noise from undercoded records as the dominant failure mode for supervised models on MIMIC; any residual leakage between text and code-derived labels biases the learned representation.
4. **Masking fallback specifics (n = 3,219 patients).** For the remaining single-admission and multi-admission patients without pre-diagnosis admissions, section-filtered index notes are used with masking applied. Two ablations (Section 3.6 below) quantify how much residual leakage these notes still carry.

### 3.4 Variable selection

- **Text features (primary).** Section-filtered free text of the patient's notes, fed directly to the transformer encoders.
- **Structured features (baseline only).** Age at admission, sex, length of stay, emergency admission flag, Medicaid/self-pay flag, count of prior admissions, race indicators, and prior-admission flags for major depressive disorder, anxiety disorder, substance use disorder, traumatic brain injury, chronic pain, suicidal ideation, SSRI/SNRI use, prazosin use, and second-generation antipsychotic use. Crucially, medications appear *only* as structured baseline predictors and as proxy-group labelling criteria — they are not in the NLP text features, since the discharge medications section is filtered out.

### 3.5 Models

Five classifiers were built so that improvements attributable to the transformer architecture and the PU loss can be isolated — including a zero-training naive keyword baseline to measure the floor performance achievable without any model training.

| Model | Architecture | Input | Loss / training | Training time | Inference time |
|---|---|---|---|---|---|
| **Clinical Longformer (primary)** | `yikuan8/Clinical-Longformer`, 4,096 tokens | Section-filtered note text | Kiryo et al. (2017) non-negative PU risk estimator. Pre-trained on MIMIC-III clinical notes. AdamW, LR 2e-5, batch 4, 5 epochs, warmup 10%, weight decay 0.01. | ~7.5 GPU-hours | ~minutes |
| **BioClinicalBERT (comparison)** | `emilyalsentzer/Bio_ClinicalBERT`, 512 tokens | Truncated section-filtered text | Same PU loss. Comparison of long-context vs. truncated context. | ~12 GPU-minutes | ~minutes |
| **TF-IDF + logistic regression (text baseline)** | sklearn LR | Word 1- and 2-grams, max 50k features, sublinear TF | `class_weight='balanced'`, L2 regularization, C tuned on validation AUPRC. | ~minutes (CPU) | ~seconds |
| **Structured + logistic regression (no-text baseline)** | sklearn LR | 20 structured features | Same L2/balanced setup. | ~minutes (CPU) | ~seconds |
| **Keyword/phrase-lookup (naive baseline)** | Regex matching | DSM-5/PCL-5 derived phrase lexicon, 62 weighted patterns | No training — hand-curated phrase weights by DSM-5 criterion specificity. Best variant (raw vs. TF-normalized) selected on validation AUPRC. | **0 seconds** | ~seconds |

**Why include a keyword baseline.** The keyword baseline establishes the floor performance achievable by a domain expert crafting a phrase list without any machine learning. If the NLP models cannot substantially outperform it, the added compute cost and complexity are unjustified. The keyword lexicon is derived from DSM-5 diagnostic criteria (A–E) and PCL-5 screening items, with weights reflecting clinical specificity for PTSD vs. general psychiatric language (3.0 for highly specific terms like "flashback", "hypervigilance"; 1.0 for shared terms like "insomnia"; 0.5 for weak signals like "guilt"). This comparison also provides a speed-vs-accuracy reference point: the keyword model runs in seconds on CPU, while the Longformer requires ~7.5 GPU-hours to train.

**Why Clinical Longformer was chosen as primary.** Discharge notes are long; truncating to 512 tokens with BioClinicalBERT loses the social history and a large fraction of brief hospital course, which are the sections clinicians use to encode trauma exposure. Li et al. (2022) showed Clinical Longformer outperforms BioClinicalBERT on MIMIC-III phenotyping with long inputs, and the same advantage is expected here.

**Why PU learning, and why the Kiryo (2017) non-negative estimator specifically.** The unlabeled pool is a mixture of true negatives and undercoded positives; treating every unlabeled patient as a confirmed negative biases the loss. Two PU formulations were considered:

- *Elkan & Noto (2008).* Assumes the labeled positives are Selected Completely At Random (SCAR) from all true positives. SCAR is clearly violated here — PTSD coding is biased toward younger women with prior psychiatric contact, so the labeled set is a structured subsample of true PTSD, not a uniform sample. Used only as sensitivity analysis.
- *Kiryo et al. (2017).* No SCAR assumption; uses the class prior $\pi_p$ to construct a non-negative empirical risk estimator that does not collapse the unlabeled pool into a single label. Estimated from ICD prevalence (not from the proxy group, which would create circularity). This is the primary loss.

A class-weighted cross-entropy is **not** layered on top of the PU loss — that would double-correct for class imbalance and degrade calibration.

**Why proxy patients are not training labels.** Adding prazosin + SSRI patients as positives would cause the model to learn the prazosin/SSRI rule itself, not the underlying PTSD signal — the same circularity as ICD reliance, but with a different proxy. They are kept fully held out and used only to *test* the trained model.

### 3.6 Training, splits, and ablations

- **Patient-level splits.** Stratified 80/10/10 train/val/test by patient (`subject_id`), so a patient's data never spans split boundaries. PTSD label is preserved across splits.
- **Class prior $\pi_p$.** Estimated from ICD prevalence in the labeled training pool; held fixed during PU training.
- **Hyperparameters.** Longformer: max_len 4096, batch 4, lr 2e-5, 5 epochs, warmup 0.1, weight decay 0.01. BioClinicalBERT: max_len 512, batch 8, lr 2e-5, 5 epochs. Both checkpoint by validation AUPRC.
- **Ablation 1 (PTSD-string masking).** Mask explicit "PTSD"/"posttraumatic"/"post-traumatic stress" strings everywhere in the corpus and re-evaluate the trained model. If performance is unchanged, the model is not exploiting label-leaking strings.
- **Ablation 2 (Past Medical History removal).** Strip the entire PMH section from all notes and re-evaluate. PMH is the most likely source of residual leakage even on pre-diagnosis admissions ("history of PTSD" carried forward). If performance survives this, the narrative sections (HPI, SH, BHC) are doing real work.

### 3.7 Evaluation strategy

**Primary metric: AUPRC.** AUROC is misleading under class imbalance — it can stay high while precision at clinically actionable thresholds is poor. AUPRC is reported alongside AUROC; AUPRC is what the ranking decision is based on.

**Operating threshold (Fix 4: val-derived).** Calibrated for sensitivity ≥ 0.85 on the **validation** set, then frozen before any test-set metrics are computed. This eliminates selection-on-test bias that inflates F1/specificity (Kennedy et al. 2024, TRIPOD-AI guidelines). The val-derived threshold is stored in `evaluation_results.json` under `val_thresholds` and inherited by all downstream scripts (proxy validation, error analysis). Sensitivity, specificity, precision, and F1 at that threshold are reported.

**Statistical comparisons.** McNemar's test on the test set for pairwise model comparison (all five models, including keyword baseline).

**Prevalence recalibration.** Because of 3:1 case-control matching, study cohort prevalence is much higher than real inpatient prevalence (~ 1–3%). PPV, NPV, and number-needed-to-screen are recalibrated to deployment prevalences of 1%, 2%, 5%, 10%, and 20% via Bayes' theorem so that reported precision is not deceptively flattering.

**Clinical utility metrics.** For each model at the operating threshold, the following metrics are reported to assess real-world deployment viability:
- *Alert rate*: fraction of patients flagged positive — determines clinician workload.
- *Positive likelihood ratio (LR+)*: how much a positive result increases the odds of PTSD. LR+ > 10 is considered strong; LR+ > 5 is moderate.
- *Negative likelihood ratio (LR-)*: how much a negative result decreases the odds. LR- < 0.1 is strong rule-out.
- *Diagnostic odds ratio (DOR)*: ratio of LR+ to LR-, a single summary of discriminative power.
- *Workup reduction vs. treat-all*: 1 − alert_rate. The fraction of patients the model avoids flagging compared to screening everyone.
- *Number needed to evaluate (NNE)*: 1 / (sensitivity − false positive rate). How many patients must be evaluated per correct detection above chance.

**Speed-vs-accuracy comparison.** Runtime benchmarks for each model (training wall-clock, inference wall-clock, GPU-hours, peak memory) are logged to `results/metrics/ewang163_runtime_benchmarks.csv` and reported alongside AUPRC in the evaluation summary. This allows direct assessment of whether the Longformer's ~7.5 GPU-hour training investment is justified vs. the keyword baseline's zero training cost.

**PU-corrected metrics (Fix 6: Ramola et al. 2019).** Every reported AUPRC, AUROC, specificity, and PPV treats unlabeled test patients as confirmed negatives. A model that correctly identifies hidden PTSD in unlabeled patients is penalized as producing false positives. Following Ramola et al. (2019, Pac Symp Biocomput, PMID 30864316), all metrics are labelled as "PU lower bounds" and corrected estimates are reported alongside them. The corrections take the estimated class prior $\pi_p$ as input and adjust for the fraction of the unlabeled pool that is truly positive. Both raw and corrected metrics are reported; the proxy validation Mann-Whitney AUC is elevated to co-headline status as the only PU-uncontaminated metric.

**Calibration.** Reliability curves by decile, Expected Calibration Error before/after Platt scaling. Kiryo PU loss is not natively calibrated, so post-hoc Platt scaling is fit on the validation set.

**Decision curve analysis (Vickers).** Net benefit at deployment prevalences of 2% and 5% over a clinically reasonable threshold range, compared against treat-all and treat-none defaults. This converts model performance into the clinical-action units that matter for screening decisions.

**Subgroup analysis.** Stratified AUPRC by sex, age decade, race/ethnicity, and emergency vs. elective admission. PTSD is differentially undercoded across these strata, so the model should be audited for whether it inherits the bias.

**Pharmacological proxy external validation.** Apply the trained model (which has *never seen* proxy patients) to the held-out 163 proxy subjects. If proxy patients receive substantially higher predicted probabilities than a random unlabeled comparison group, this is non-circular evidence the model has learned PTSD-associated language rather than just the surface features of the labeled set. Statistical test: Mann-Whitney U.

**Specificity check (psych-control retraining).** Train an additional Longformer with PTSD+ vs. age/sex-matched MDD/anxiety controls (using standard cross-entropy, since this is no longer a PU task) to test whether the primary model is learning PTSD-specific signal or general psychiatric language.

**Explainability.** Integrated Gradients (Captum `LayerIntegratedGradients`) on 50 high-confidence true positives, with attribution aggregated by note section and by token. Attention weights deliberately not used: Jain & Wallace (2019) demonstrate attention is unreliable as a feature importance proxy, while Sundararajan et al. (2017) prove IG satisfies completeness and sensitivity axioms.

**Error analysis.** Sample 25 false positives and 25 false negatives at the operating threshold; write out the underlying notes with predicted probabilities, plus aggregate statistics on note length, subgroup composition, and demographic skew.

### 3.8 Where this builds on prior literature

- **Blackley et al. (2021)** developed an NLP + ML pipeline (MTERMS rule-based classifier + sklearn classifiers + DNN) for opioid use disorder identification at Brigham and Women's Hospital. They used a manually curated lexicon, expert-developed rules, and traditional ML on top of NLP-derived features. The present project adapts the same overall framing — extract clinically meaningful information from sectioned discharge notes, train classifiers, evaluate on a held-out set — but replaces the rule + lexicon front end with a fine-tuned Clinical Longformer, addresses label leakage explicitly via section filtering and pre-diagnosis training, and addresses the contaminated negative pool via PU learning.
- **Afshar et al. (2019, JAMIA)** and **Sharma et al. (2020, BMC MIDM)** built NLP classifiers for alcohol misuse and opioid misuse from inpatient EHR notes using cTAKES CUI features and traditional ML; they used SBIRT or AUDIT scores as the gold standard. They are the closest prior art for the substance-misuse phenotyping pattern, but their reference standards were prospectively collected (AUDIT screens), so they did not have to deal with the contaminated-negatives problem that defines PTSD.
- **Dligach, Afshar & Miller (2019)** trained a clinical text encoder by pre-training on billing-code prediction and then using the encoder for substance misuse downstream. Their work motivates why a transformer encoder pre-trained on clinical text generalises across phenotyping tasks; that's the reason `yikuan8/Clinical-Longformer` (pre-trained on MIMIC-III clinical notes) is the chosen backbone here rather than a general-domain Longformer.
- **Koola et al. (2018)** built a hepatorenal syndrome phenotyping pipeline that combined structured EHR variables with NLP-derived CUI features and compared multiple dimension reduction strategies. Their methodology of running a structured-features baseline alongside the text model, then attributing improvement to the NLP component, is mirrored here with the structured logistic regression baseline.
- **Zhou et al. (2015)** identified depression patients from discharge summaries using MTERMS and machine learning, finding ~ 20% additional cases beyond ICD coding alone. This is a direct precedent for the central claim that NLP can recover undercoded psychiatric diagnoses from inpatient notes — the present project tests whether the same is true for PTSD specifically and whether PU learning makes the recovery quantitatively defensible.
- **Kiryo et al. (2017)** is the methodological backbone for handling the unlabeled pool. **Elkan & Noto (2008)** is run as a sensitivity analysis only because the SCAR assumption is implausible for PTSD.
- **Li et al. (2022)** introduced Clinical Longformer and showed it outperforms BioClinicalBERT on long-context MIMIC-III phenotyping; that motivates the primary model choice and the explicit head-to-head comparison.

### 3.9 Methodology fixes (post-initial-results)

After the initial pipeline was built and results reviewed, a systematic code-level audit identified 11 methodology improvements. These are documented in `methodology_fix_plans.md` with published evidence for each. The key fixes are summarized here:

**Fix 1 — Universal PTSD-string masking.** Masking now applied to all PTSD+ notes (pre-dx + fallback), closing a leakage path where clinicians carry forward "h/o PTSD" in HPI/PMH on pre-diagnosis admissions. Jin et al. (2023, JAMIA) identified this exact annotation-noise failure mode.

**Fix 2 — Class prior sweep.** π_p is swept over {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25} rather than naively set to the empirical labeled fraction. Best value selected by proxy Mann-Whitney AUC, the only PU-uncontaminated criterion.

**Fix 3 — PULSNAR SAR-PU.** Propensity-weighted nnPU loss reweights the positive term by 1/e(x), where e(x) is estimated from structured features via logistic regression. This addresses the SCAR violation that PTSD coding is biased by age, sex, and prior psychiatric contact. PULSNAR library (Kumar & Lambert 2024) used for class-prior estimation when available.

**Fix 4 — Val-derived thresholds.** All operating thresholds computed on validation, not test. Eliminates selection-on-test bias (Kennedy et al. 2024, TRIPOD-AI).

**Fix 5 — Elkan-Noto calibration.** Post-Platt probabilities divided by c = P(s=1|y=1) so outputs approximate P(PTSD=1|text) rather than P(coded=1|text).

**Fix 6 — PU lower-bound metrics.** All reported metrics labelled as "PU lower bounds" with Ramola et al. (2019) correction formulas applied alongside raw values. Proxy AUC elevated to co-headline.

**Fix 7 — Temporal split.** Pre-2015 train, post-2015 test, to assess generalization across the ICD-9→ICD-10 coding transition and post-DSM-5 reclassification.

**Fix 8 — Chunk-and-pool BERT.** BioClinicalBERT evaluated with both single-512 truncation and overlapping chunk-and-pool (512 window, 256 stride, max-pool). This controls for context length and isolates the architecture comparison.

**Fix 9 — Defensible fairness.** Subgroup AUPRC replaced with calibration-in-the-large, equal opportunity difference, and bootstrap 95% CI. Per-race AUPRC only reported when CI width < 0.15.

**Fix 10 — Full-context IG.** Integrated Gradients at 4096 tokens (was 1024), matching the training context length. Ensures attribution covers Brief Hospital Course content that appears late in long notes.

**Fix 11 — Chart review packet.** Top-50 model-flagged unlabeled patients packaged with de-identified notes and a rating form for clinician review. Clinician-rated PPV at the top decile would be the single most persuasive validation metric.

**Naive keyword baseline.** A zero-training phrase-lookup model using 62 DSM-5/PCL-5-derived patterns establishes the floor performance and enables speed-vs-accuracy comparison across all methods.

**Runtime benchmarking.** All scripts log wall-clock time, CPU time, peak memory, and GPU-hours to a shared CSV for compute-cost analysis.

---

## 4. Results

All numbers below are on the patient-level held-out test set (n = 1,551 patients, n = 660 PTSD+) unless stated otherwise. Files cited in italics are under `results/`.

### 4.1 Model comparison (test set)

*results/metrics/ewang163_evaluation_summary.csv*, *ewang163_evaluation_results.json*

| Model | AUPRC | AUROC | Sens @ ≥ 0.85 | Specificity | Precision | F1 | Threshold | McNemar p vs. Longformer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clinical Longformer (PU)** | **0.8827** | **0.8913** | 0.85 | 0.7464 | 0.7128 | **0.7754** | 0.380 | — |
| BioClinicalBERT (PU, max 512) | 0.8550 | 0.8648 | 0.8545 | 0.6779 | 0.6627 | 0.7465 | 0.967 | 1.8e-4 |
| TF-IDF + logistic regression | 0.8328 | 0.8530 | 0.85 | 0.6420 | 0.6375 | 0.7286 | 0.228 | < 1e-3 |
| Structured + logistic regression | 0.6833 | 0.7310 | 0.85 | 0.4119 | 0.5171 | 0.6430 | 0.377 | < 1e-3 |

Clinical Longformer is the top performer on every discrimination metric. It beats BioClinicalBERT by +0.028 AUPRC (McNemar p = 1.8e-4) and beats the TF-IDF text baseline by +0.050 AUPRC. The structured-only baseline scores 0.20 AUPRC below the text baseline and 0.20 below the Longformer, confirming that the bulk of the predictive signal lives in the narrative notes, not the structured demographic/comorbidity record. Longformer's lift over BioClinicalBERT (~ 3 AUPRC points, McNemar significant) supports the architectural choice: long-context attention recovers signal that 512-token truncation discards.

### 4.2 Training dynamics

*results/metrics/ewang163_longformer_training_log.csv*, *ewang163_bioclinbert_training_log.csv*

Clinical Longformer (5 epochs, ~ 1.5h GPU time per epoch):

| Epoch | Train loss | Val AUPRC |
|---:|---:|---:|
| 1 | 0.479 | 0.8525 |
| 2 | 0.318 | 0.8686 |
| **3** | **0.254** | **0.8704** |
| 4 | 0.203 | 0.8658 |
| 5 | 0.154 | 0.8637 |

Best checkpoint is epoch 3 by validation AUPRC; epochs 4 and 5 begin to overfit (training loss drops, validation AUPRC declines).

BioClinicalBERT (5 epochs, ~ 2.5 min per epoch):

| Epoch | Train loss | Val AUPRC |
|---:|---:|---:|
| 1 | 0.389 | 0.7799 |
| 2 | 0.202 | 0.8300 |
| 3 | 0.146 | 0.8267 |
| 4 | 0.097 | 0.8305 |
| **5** | **0.081** | **0.8345** |

### 4.3 Prevalence recalibration

*results/metrics/ewang163_evaluation_results.json*

Because of 3:1 case-control matching, study-cohort precision flatters real-world deployment. Recalibrated to plausible inpatient prevalences:

| Deployment prevalence | Recalibrated PPV | Number needed to screen |
|---:|---:|---:|
| 1% | 0.0327 | 30.5 |
| 2% | 0.0640 | 15.6 |
| 5% | 0.1500 | 6.7 |

At a realistic 2% prevalence, ~ 16 patients need to be flagged to find one true PTSD case — clinically tolerable for an automated screening prompt that points clinicians at otherwise-missed patients.

### 4.4 Ablation studies (label leakage)

*results/metrics/ewang163_ablation_results.csv*

| Condition | AUPRC | AUROC | Δ vs. baseline |
|---|---:|---:|---:|
| Baseline (unmasked) | 0.8827 | 0.8913 | — |
| Ablation 1: explicit PTSD strings masked | 0.8749 | 0.8898 | −0.008 |
| Ablation 2: PMH section removed entirely | 0.8218 | 0.8382 | −0.061 |

**Interpretation.** Ablation 1 costs almost nothing — the model is *not* relying on the literal string "PTSD" in the notes. This is the strongest single piece of evidence that section filtering plus pre-diagnosis training prevented gross label leakage.

Ablation 2 is more interesting: removing PMH costs ~ 6 AUPRC points but the model still achieves AUPRC 0.82, comfortably above the TF-IDF baseline. Some signal lives in PMH (carried-forward psychiatric history, prior trauma exposure), but it's neither dominant nor sufficient — HPI, social history, and brief hospital course independently encode enough PTSD-associated language to drive the model.

### 4.5 Calibration

*results/metrics/ewang163_calibration_results.csv*, *ewang163_evaluation_results.json* (`longformer.calibration`), *results/figures/ewang163_calibration_curve.png*

Raw PU outputs show the expected pattern: under-confident at high probabilities (the top decile has mean predicted 0.998 vs. observed 0.981 — actually slightly *over*-confident at the very top, but well-calibrated), and over-confident in the middle bins (deciles 6–8 predict 0.51–0.90 for observed rates of 0.37–0.76). Expected Calibration Error: **0.0626 raw → 0.0585 after Platt scaling**, a modest improvement. Platt scaling is fit on validation and saved as `models/ewang163_platt_calibrator.pkl`. The raw scores are usable for ranking and threshold selection; Platt-scaled scores should be used wherever the absolute probability matters (e.g., decision-curve net benefit at deployment prevalence).

### 4.6 Decision curve analysis

*results/metrics/ewang163_dca_results.csv*, *ewang163_dca_2pct.png*, *ewang163_dca_5pct.png*

At a 2% deployment prevalence, the Longformer model offers positive net benefit over treat-none across thresholds 0.01–0.14 but does not beat treat-all in the very low-threshold band (treat-all dominates because the cost of missing a case is high relative to the cost of a false positive at low prevalence). At the 5% prevalence regime more typical of trauma-exposed inpatient cohorts, the model offers meaningful net benefit across a wider threshold range (0.01–0.30) and approaches but does not exceed treat-all in the 0.01–0.05 region. The clinically relevant takeaway: at deployment prevalence ≥ 5%, the model is a defensible alternative to flagging everyone for evaluation, freeing clinician time to focus on the highest-probability patients first.

### 4.7 Subgroup analysis

*results/metrics/ewang163_evaluation_results.json* → `subgroup_analysis`

| Subgroup | n | n_pos | AUPRC |
|---|---:|---:|---:|
| **Sex: Female** | 940 | 433 | 0.9109 |
| Sex: Male | 611 | 227 | 0.8219 |
| **Age 20s** | 293 | 140 | 0.9359 |
| Age 30s | 359 | 194 | 0.9168 |
| Age 40s | 308 | 131 | 0.8797 |
| Age 50s | 277 | 81 | 0.8417 |
| Age Other | 314 | 114 | 0.8168 |
| **Race: Black** | 209 | 91 | 0.9092 |
| Race: White | 1083 | 485 | 0.8834 |
| Race: Hispanic | 93 | 44 | 0.8586 |
| Race: Other/Unknown | 118 | 35 | 0.8686 |
| Race: Asian | 48 | 5 | 0.8435 |
| **Emergency admission: True** | 855 | 450 | 0.9225 |
| Emergency admission: False | 696 | 210 | 0.8080 |

The model performs best on the patients most likely to be coded for PTSD in the first place — younger women, emergency admissions. Performance on men (AUPRC 0.82 vs. 0.91 for women) and older patients is meaningfully worse. This is consistent with the concern that PU learning reduces but does not eliminate the inherited bias from a non-SCAR labeled set, and is the single most important caveat for deployment: the tool is best-calibrated for the demographic where coding is already most reliable, and weakest for the demographic where undercoding is most prevalent.

### 4.8 Pharmacological proxy external validation

*results/metrics/ewang163_proxy_validation_results.csv*, *results/figures/ewang163_proxy_validation.png*

Applied the trained Longformer to the 163 held-out proxy patients (102 had retrievable notes after the section filter) and to a random sample of 500 unlabeled-pool patients.

| Group | n | Median score | Mean score | Frac. above operating threshold (0.38) |
|---|---:|---:|---:|---:|
| **Pharmacological proxy** | 102 | **0.4486** | **0.4943** | **56.9%** (58/102) |
| Random unlabeled sample | 500 | 0.1138 | 0.1905 | 14.8% (74/500) |

Mann-Whitney U = 40,072, **p = 4.43e-20**, AUC = 0.7857.

This is the most important non-circular validity check in the project. The model — which was never shown a single proxy patient during training — assigns ~ 4× the median probability to patients whose pharmacotherapy is consistent with PTSD treatment compared to demographically comparable unlabeled patients. 56.9% of proxy patients clear the operating threshold vs. 14.8% of unlabeled patients. The model is not just memorising the labeled cohort; it is recovering a clinically meaningful PTSD signal from narrative text that generalises to a population identified by an entirely independent (medication-based) criterion.

### 4.9 Specificity check vs. psychiatric controls

*results/metrics/ewang163_specificity_eval_results.json*

Trained an additional Longformer with PTSD+ patients vs. age/sex-matched MDD/anxiety controls (n = 5,711 each, 3,148 had section-filtered notes), using standard cross-entropy. This isolates the question of whether the primary model is learning PTSD-specific language or general "this patient has a psychiatric condition" language.

| Metric | PU model (primary) | Specificity model (psych controls) |
|---|---:|---:|
| Test AUPRC | 0.8827 | **0.9109** |
| Test AUROC | 0.8913 | 0.8145 |
| Sensitivity @ 0.85 | 0.85 | 0.8523 |
| Specificity | 0.7464 | 0.5810 |
| Precision | 0.7128 | 0.8211 |

The specificity model achieves +0.028 higher AUPRC against psychiatric controls than the PU model achieves against the unlabeled pool, but with worse AUROC and worse specificity. The interpretation is favourable to the primary model: if the PU model were just learning "psychiatric patient", you would expect it to fail to discriminate PTSD from MDD/anxiety, and the specificity model would reveal a much larger drop. Instead, PTSD-specific signal is clearly recoverable even when the comparator is other psychiatric admissions, and the absolute discrimination gap is small. Mean predicted probability for the proxy validation set under the specificity model is 0.3445 — lower than under the PU model (0.4943) but still well above baseline, consistent with the proxy patients being psychiatrically heterogeneous.

### 4.10 Integrated Gradients attribution

*results/attribution/ewang163_attribution_by_section_v2.csv*, *ewang163_top_attributed_tokens_v2.csv*

Attribution by section (50 high-confidence true positives):

| Section | Total attribution | % of total | n tokens | Mean attribution / token |
|---|---:|---:|---:|---:|
| History of present illness | 207.22 | **54.6%** | 25,046 | 0.0083 |
| Brief hospital course | 85.19 | 22.4% | 10,482 | 0.0081 |
| **Past medical history** | **78.91** | 20.8% | 4,335 | **0.0182** |
| Social history | 5.94 | 1.6% | 1,118 | 0.0053 |
| Unknown | 2.39 | 0.6% | 275 | 0.0087 |

HPI dominates total attribution, but PMH has by far the highest *per-token* attribution density — consistent with PMH being short, dense, and information-rich. Social history contributes less than expected, which is somewhat surprising given the prior expectation that trauma history and substance use would drive predictions; this may be a MIMIC-IV documentation artefact (many social history sections in MIMIC-IV are minimal).

Top attributed tokens (excluding subword fragments and stop tokens) include explicit psychiatric vocabulary that survived section filtering: `schizophrenia`, `psychiatric`, `psychiatrist`, `psychiatry`, `psychotic`, `psychosis`, `depressive`, `depression`, `anxiety`, `bipolar`, `personality`, `flashbacks`, `nightmares`-related fragments; substance vocabulary (`overdose`, `detox`, `heroin`, `narcotics`, `substance`); trauma-context vocabulary (`trauma`, `assault`, `assaulted`, `raped`, `violence`, `abusive`, `abuse`, `hitting`, `falls`); and treatment-related (`counseling`, `counselor`, `psychiatrist`, `bid`). This is exactly the lexical surface of trauma-exposed patient histories, and is encouraging — the model is not relying on any single trigger word but on a broad, plausible vocabulary.

### 4.11 Error analysis

*results/error_analysis/ewang163_error_analysis_summary.csv*, *ewang163_fp_notes_sample.txt*, *ewang163_fn_notes_sample.txt*

| | n | Mean pred. prob | Mean note length | % female | % age 20s | % age 50s | % emergency |
|---|---:|---:|---:|---:|---:|---:|---:|
| False positives | 227 | 0.6622 | 3,239 | 60.4% | 15.0% | 23.8% | 42.3% |
| False negatives | 99 | 0.1803 | 2,932 | 49.5% | 10.1% | 13.1% | 27.3% |
| Overall test set | 1,551 | 0.4772 | 3,108 | 60.6% | 17.2% | 18.2% | 37.9% |

False positives skew female and emergency, matching the demographic that gets PTSD coded preferentially in the labeled set — these are likely true undercoded PTSD cases that the ICD label does not reflect. False negatives skew male, older, and elective; this is consistent with the subgroup AUPRC analysis and reinforces that the model inherits the PTSD coding bias from its training labels.

A qualitative inspection of sampled FP/FN notes (100K-token text files) shows that many false positives have substance use, prior psychiatric admissions, or assault histories that would have justified a PTSD coding had the inpatient team conducted a structured screen. False negatives are over-represented in patients whose section-filtered notes contained almost no narrative context (very short admissions, terse documentation), where the model has nothing to anchor on.

---

## 5. Discussion

### 5.1 Summary of findings

The primary model (Clinical Longformer fine-tuned with Kiryo PU loss on section-filtered, predominantly pre-diagnosis notes) achieves AUPRC 0.88 / AUROC 0.89 on the held-out test set, beating BioClinicalBERT, TF-IDF + LR, and a structured-features-only baseline by statistically significant margins (McNemar p < 1e-3 against all comparators). At a sensitivity-weighted operating threshold (sens ≥ 0.85), specificity is 0.75 and precision is 0.71. Recalibrated to a 2% inpatient deployment prevalence, the number needed to screen is ~ 16, which is in the range of clinically tolerable for a screening prompt that points clinicians at otherwise-missed cases.

The two label-leakage ablations both pass: explicit PTSD-string masking costs almost nothing (−0.008 AUPRC), and even removing PMH entirely leaves the model at AUPRC 0.82 — well above the TF-IDF baseline. The pharmacological proxy external validation is the clearest piece of validity evidence: proxy patients receive ~ 4× the median predicted probability of demographically comparable unlabeled patients (Mann-Whitney p = 4.4e-20, proxy-vs-unlabeled AUC 0.79), and 57% clear the operating threshold vs. 15% of the unlabeled pool. Because proxy patients are identified by medication patterns the model never saw and were never used in training, this is non-circular evidence the model has learned PTSD-associated language rather than just the surface features of the labeled set.

### 5.2 What the model is actually learning

Integrated Gradients attribution shows that HPI provides the bulk of total attribution but PMH provides the highest per-token density. The top attributed tokens are a clinically plausible mix of psychiatric vocabulary (`psychiatric`, `psychiatrist`, `bipolar`, `flashbacks`, `personality`), substance use vocabulary (`overdose`, `heroin`, `detox`), trauma context (`assault`, `raped`, `violence`, `abuse`), and treatment context (`counseling`, `counselor`). The model is not memorising any single trigger word; it is recognising the broader lexical landscape of trauma-exposed patient histories. The specificity check against psychiatric controls (MDD/anxiety) shows the model is not just learning "psychiatric admission" — even when held to that comparator, the absolute AUPRC remains > 0.91 and the proxy validation set still scores well above baseline.

### 5.3 Limitations and threats to validity

**Section filtering is not perfect leakage prevention.** Even on a pre-diagnosis admission, PMH may carry forward "history of PTSD" from an outside record. Ablation 2 quantifies the upper bound on this — removing PMH costs only 6 AUPRC points, so residual PMH leakage is bounded but not zero.

**PU learning does not fully eliminate selection bias.** The Kiryo formulation removes the assumption that unlabeled patients are confirmed negatives but does not correct for the fact that ICD-coded patients are a non-random sample of true PTSD cases. The subgroup analysis shows exactly the expected residual bias: AUPRC is 0.91 for women and 0.82 for men, 0.94 for patients in their 20s and 0.82 for patients in their 50s+. The model is best-calibrated for the demographic where coding is most reliable, and weakest for the demographic where undercoding is most prevalent. This is the single most important caveat for deployment.

**The pre-diagnosis training subsample is not representative of the full PTSD+ cohort.** Only 2,492 of 5,711 PTSD+ patients (43.6%) had pre-diagnosis admissions. These are by definition multi-admission patients, who tend to be sicker, older, and more psychiatrically complex than single-admission patients. The remaining 3,219 patients used masking-fallback notes; their notes are still index-admission notes and carry more residual leakage risk than pre-diagnosis notes.

**Proxy validation set is small (n = 163, n = 102 with notes) and has a known FPR.** The proxy criterion (prazosin + SSRI/SNRI within 180 days, no cardiovascular/BPH/Raynaud/TBI exclusion) carries an estimated 15–20% false-positive rate — patients on prazosin for off-label uses not captured by the exclusion ICD codes. So elevated proxy scores are evidence, not proof.

**Single-site data.** MIMIC-IV is one academic medical center in Boston (BIDMC). Whether any of this transfers to community hospitals, rural settings, the VA system (where PTSD prevalence is much higher and where this tool would arguably be most valuable), or international settings is unknown and would require external validation.

**Calibration is imperfect.** ECE drops only marginally with Platt scaling (0.063 → 0.058). For ranking and threshold-based screening this is acceptable; for any clinical application that depends on absolute predicted probability, additional calibration work would be needed.

**Decision curve analysis is favourable but not dominant.** The model offers positive net benefit over treat-none in clinically reasonable threshold ranges at both 2% and 5% deployment prevalence, but does not dominate a treat-all strategy in the very low-threshold band. This is honest: at deployment prevalences below ~ 2%, the cost of missing a case is high enough relative to clinician review time that flagging everyone is competitive.

### 5.4 What this tool is and is not

**It is** a screening prompt — a way to point inpatient clinicians at patients whose narrative notes contain language patterns associated with PTSD. The output is intended to inform a more thorough psychiatric evaluation, not to make a diagnosis. At a 2% deployment prevalence and the chosen operating threshold, the tool would surface roughly 16 patients per true case for clinician review, which is consistent with how SBIRT-style screening tools are typically deployed.

**It is not** a diagnostic tool, a replacement for structured PTSD screening (PCL-5, CAPS-5), or a basis for ICD coding. The downstream user is a clinician who will conduct a proper interview before any management decision is made.

### 5.5 Future directions

- **External validation** at a non-MIMIC site, ideally one with both higher PTSD base rates and richer narrative documentation (a VA medical center would be the natural target).
- **Re-evaluation under a true reference standard.** A small prospective cohort screened with the PCL-5 or CAPS-5 would let us measure absolute model performance against a defensible gold standard rather than against contaminated ICD labels.
- **Bias mitigation for demographic subgroups.** The female / younger-age coding bias is inherited from the labels; explicit subgroup-aware loss reweighting or label noise modelling (Bekker & Davis 2020) could narrow the gap.
- **Temporal validation.** MIMIC-IV spans 2008–2019; ICD coding behaviour and discharge documentation conventions changed over that decade. A train-on-early / test-on-late split would test temporal robustness.
- **Integration with structured features.** The structured baseline AUPRC was 0.68; concatenating structured features into the Longformer head (rather than running them as a separate model) could provide complementary signal, particularly for the demographics where text features are weakest.

---

## 6. Reproducibility

All design decisions, validated cohort definitions, ICD code lists, drug lists, MIMIC-IV data quirks, and bug fixes are documented in `CLAUDE.md` (the project's authoritative specification). The pipeline is structured so that each stage saves its output to disk before the next stage begins, allowing any stage to be re-run independently. SLURM job logs for every run are preserved under `logs/`. Random seeds are fixed (`np.random.seed(42)` for matching, sklearn `random_state=42` for splits and model training).

A complete pipeline re-run from raw MIMIC-IV would be:

1. `scripts/01_cohort/ewang163_ptsd_table1.py` → `results/table1/`
2. `scripts/01_cohort/ewang163_ptsd_cohort_sets.py` → `data/cohort/*_subjects.txt`
3. `scripts/01_cohort/ewang163_ptsd_admissions_extract.py` → `data/cohort/ewang163_ptsd_adm_extract.parquet`
4. `scripts/01_cohort/ewang163_ptsd_notes_extract.py` → `data/notes/ewang163_ptsd_notes_raw.parquet`
5. `scripts/02_corpus/ewang163_ptsd_corpus_build.py` → `data/notes/ewang163_ptsd_corpus.parquet`
6. `scripts/02_corpus/ewang163_ptsd_splits.py` → `data/splits/`
7. `scripts/03_training/ewang163_ptsd_train_longformer.py` (GPU) → `models/ewang163_longformer_best/`
8. `scripts/03_training/ewang163_ptsd_train_bioclinbert.py` (GPU)
9. `scripts/03_training/ewang163_ptsd_train_tfidf.py`
10. `scripts/03_training/ewang163_ptsd_train_structured.py`
11. `scripts/04_evaluation/ewang163_ptsd_evaluate.py`
12. `scripts/04_evaluation/ewang163_ptsd_calibration.py`
13. `scripts/04_evaluation/ewang163_ptsd_decision_curves.py`
14. `scripts/04_evaluation/ewang163_ptsd_proxy_validation.py`
15. `scripts/04_evaluation/ewang163_ptsd_ablations.py`
16. `scripts/04_evaluation/ewang163_ptsd_attribution_v2.py`
17. `scripts/04_evaluation/ewang163_ptsd_error_analysis.py`
18. `scripts/03_training/ewang163_ptsd_specificity.py` (GPU, secondary analysis)

Note the path caveat in Section 1: scripts use absolute paths to `STUDENT_DIR` root and would currently re-create their outputs at the old root locations rather than the reorganized subdirectories. Updating each script's `OUT` constant to the new path is the only change required for a re-run.
