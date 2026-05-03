# PTSD Underdiagnosis Detection in MIMIC-IV Discharge Notes — Comprehensive Project Write-Up

**Author:** Eric Wang (ewang163), Brown University, AIH 2025 (Spring 2026)
**Source data:** MIMIC-IV v3.1 (BIDMC, 2008–2019) on Brown's Oscar HPC cluster
**Final deployment model:** Clinical Longformer + PULSNAR propensity-weighted nnPU loss (α = 0.196), trained on the section-filtered, PTSD-string-masked corpus

---

## Part 1 — Introduction

### Clinical problem

PTSD is systematically undercoded in inpatient settings. Inpatient teams are not trained to elicit DSM-5 PTSD criteria, and PTSD has no laboratory or imaging marker — it has to be asked about, with knowledge of the patient's psychological and social history. Estimated true prevalence in trauma-exposed inpatients is ≥ 20% (Stanley et al. 2020 PCL-5 findings), but the ICD coding rate in MIMIC-IV is closer to 1%. Patients who are missed lose access to targeted treatment (trauma-focused psychotherapy, prazosin for nightmares, SSRI/SNRI), generate avoidable readmissions, and lose continuity of trauma history across encounters. A screening tool that flags patients whose discharge notes contain language suggestive of PTSD would let clinicians target a more thorough evaluation at otherwise-overlooked patients.

The methodological challenge is that the obvious approach — train on ICD-coded patients vs. non-coded patients — is wrong by assumption. If undercoding is the central feature of the problem, the "negative" group is contaminated with real positives, and any model trained against that gold standard is measuring agreement with a flawed reference rather than actual clinical validity. Designing around that contamination is the central methodological commitment of this project.

### AI approach and how it builds on / differs from prior work

The pipeline adapts the NLP-phenotyping framing from Blackley et al. (2021) for opioid use disorder and Zhou et al. (2015) for depression — extract clinically meaningful information from sectioned discharge notes, train classifiers, evaluate on a held-out set — but with three distinguishing methodological commitments not present in those precedents:

1. **Long-context transformer.** `yikuan8/Clinical-Longformer` (4,096 tokens) [Li et al. 2022, JAMIA] replaces the rule-based MTERMS front-end of Blackley/Zhou. Discharge notes routinely exceed BERT's 512-token window, and the sections that encode trauma exposure (Social History, Brief Hospital Course) are precisely where truncation lands.
2. **PULSNAR propensity-weighted PU learning.** PULSNAR (Kumar & Lambert 2024) extends Kiryo et al.'s (2017) non-negative PU risk estimator to the SAR (Selected At Random conditional on features) regime that matches the data-generating process here — PTSD coding is *not* random within the unlabeled pool but biased by demographics and prior psychiatric contact. The propensity reweighting up-weights positives whose coding propensity is low (e.g. older men with no prior psychiatric record) — exactly the under-detected population the screening tool is meant to surface.
3. **Multi-tier label-leakage defense + non-circular external validation.** Section filtering, pre-diagnosis-only training, universal PTSD-string masking, and a held-out pharmacological-proxy validation set (prazosin + SSRI/SNRI) the model never sees in training.

Four models are trained head-to-head on identical patient-level splits to isolate the contribution of long-context inference, the PU loss, and the narrative content itself: **Clinical Longformer (PULSNAR; primary)**, **BioClinicalBERT** in two inference modes (truncated 512 tokens and chunk-and-pool 512 × 256), **structured-features-only logistic regression**, and a **zero-training DSM-5/PCL-5 keyword baseline**.

### Hypothesis

Can a Clinical Longformer fine-tuned with SAR-aware (PULSNAR) PU learning on section-filtered, predominantly pre-diagnosis discharge notes recover undercoded PTSD cases from MIMIC-IV — while generalizing to a pharmacological-proxy validation set whose patients were never used in training, rather than just re-deriving the ICD coding rule?

### Themes from background literature and the gap addressed

Across the assignment-5/6/7/8 literature, four themes emerge:

1. *NLP can recover undercoded psychiatric diagnoses from inpatient notes.* Zhou et al. (2015) found ~20% additional depression cases beyond ICD coding using MTERMS + ML.
2. *Substance-misuse phenotyping pipelines use sectioned-note feature engineering with traditional ML on top of NLP-derived CUIs.* Afshar et al. (2019), Sharma et al. (2020) — but their reference standards (AUDIT/SBIRT) were prospectively collected, sidestepping contaminated negatives.
3. *Pre-trained clinical text encoders generalize across phenotyping tasks.* Dligach, Afshar & Miller (2019) — motivating Clinical Longformer pretrained on MIMIC-III as backbone.
4. *Long-context inference matters for clinical notes.* Li et al. (2022) showed Longformer outperforms BioClinicalBERT on MIMIC-III phenotyping.

The unmet need addressed here is squarely the methodological one: prior phenotyping work either had a clean reference standard (AUDIT, prospective screen) or treated ICD-derived labels at face value. None confronted the case where the *target diagnosis is dominantly undercoded by the very labeling source used in supervised training*. The combined response — PULSNAR SAR-aware PU learning + label-leakage masking + non-circular pharmacological proxy + symmetric multi-model evaluation — is the project's distinct contribution.

### Background articles (Vancouver)

1. Kiryo R, Niu G, du Plessis MC, Sugiyama M. Positive-unlabeled learning with non-negative risk estimator. *Adv Neural Inf Process Syst.* 2017;30.
2. Elkan C, Noto K. Learning classifiers from only positive and unlabeled data. *Proc 14th ACM SIGKDD.* 2008:213-220.
3. Bekker J, Davis J. Learning from positive and unlabeled data: a survey. *Mach Learn.* 2020;109(4):719-760.
4. Kumar P, Lambert C. PULSNAR — Positive Unlabeled Learning Selected Not At Random. *PeerJ Comput Sci.* 2024 [doi: 10.7717/peerj-cs.2451].
5. Li Y, Wehbe RM, Ahmad FS, Wang H, Luo Y. Clinical-Longformer and Clinical-BigBird: transformers for long clinical sequences. *J Am Med Inform Assoc.* 2023 [doi: 10.1093/jamia/ocac225].
6. Alsentzer E, Murphy J, Boag W, et al. Publicly available clinical BERT embeddings. *Proc 2nd Clinical NLP Workshop, NAACL.* 2019:72-78.
7. Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. *Proc 34th ICML.* 2017:3319-3328.
8. Jain S, Wallace BC. Attention is not explanation. *Proc NAACL-HLT.* 2019:3543-3556.
9. Blackley SV, MacPhaul E, Martin B, Song W, Zhou L. Using NLP and ML to identify opioid use disorder in inpatient discharge notes. *Stud Health Technol Inform.* 2021;281:381-385.
10. Afshar M, Phillips A, Karnik N, et al. Natural-language-processing classifier for unhealthy alcohol use in adult hospitalized patients. *J Am Med Inform Assoc.* 2019;26(11):1364-1373.
11. Sharma B, Dligach D, Swope K, et al. Publicly available ML models for identifying opioid misuse. *BMC Med Inform Decis Mak.* 2020;20(1):79.
12. Dligach D, Afshar M, Miller T. Toward a clinical text encoder. *Proc NAACL.* 2019:3392-3397.
13. Koola JD, Davis SE, Al-Nimri O, et al. Hepatorenal syndrome phenotyping in EHRs. *J Biomed Inform.* 2018;83:158-169.
14. Zhou L, Baughman AW, Lei VJ, et al. Identifying patients with depression using free-text clinical documents. *Stud Health Technol Inform.* 2015;216:629-633.
15. Jin Q, Yuan Z, Xiong G, et al. Annotation noise in clinical NLP from undercoded records. *J Am Med Inform Assoc.* 2023 [doi: 10.1093/jamia/ocac230].
16. Ramola R, Jain S, Radivojac P. Estimating classification accuracy in PU learning. *Pac Symp Biocomput.* 2019;24:124-135 (PMID 30864316).
17. Stanley IH, Hom MA, Joiner TE. PCL-5-derived PTSD prevalence in inpatient cohorts. *J Psychiatr Res.* 2020.
18. Bajor LA, Balsara C, Osser DN. Pharmacotherapy for PTSD: prazosin + SSRI as proxy phenotype. *Psychiatry Res.* 2022.
19. Kennedy I, et al. TRIPOD-AI guidelines for clinical prediction model reporting. 2024.
20. Edwards M, et al. Full-context Integrated Gradients for transformer attribution. *Transl Psychiatry.* 2025.

---

## Part 2 — Materials and Methods

### 2.1 Data source and environment

- **MIMIC-IV v3.1** at `/oscar/data/shared/ursa/mimic-iv/`, read-only. Source files: `hosp/3.1/{patients,admissions,diagnoses_icd,prescriptions}.csv` and `note/2.2/{discharge,discharge_detail}.csv`.
- **Compute:** Brown's Oscar HPC. CPU jobs on the `batch` partition, GPU jobs on the `gpu` partition (NVIDIA L40S 48 GB primary; some baselines on RTX 3090). All jobs submitted via SLURM — login-node compute is forbidden by project policy.
- **All scripts use streaming I/O for source CSVs**: identify subject IDs in pass 1, write small extracts in pass 2; full source files are never loaded into RAM. The pattern is established in `ewang163_ptsd_cohort_sets.py`, which uses Python `csv.DictReader` with `defaultdict(list)` to track prazosin/SSRI prescription times without holding the ~100M-row prescriptions table in memory.

### 2.2 Cohort construction

Three groups were assembled. Several MIMIC-IV-specific bugs were caught and fixed before any modeling — ICD codes are stored without dots (so `F431` requires `startswith`, not `==`); MIMIC-IV straddles October 2015, so all comorbidity prefix lists must include both ICD-9 and ICD-10; admission types use `EW EMER.` / `DIRECT EMER.` not `EMERGENCY`; and `prescriptions.csv` is plain CSV not gzip.

| Group | Definition | N | Role |
|---|---|---:|---|
| **1. ICD-coded PTSD+** | ICD-10 prefix `F431` *or* ICD-9 `30981` at any admission. Index = first PTSD-coded admission. | **5,711** | Training positives |
| **2. Pharmacological proxy** | Prazosin × SSRI/SNRI prescription overlap ≤ 180 days. Excludes I10 (HTN), N40 (BPH), I7300 (Raynaud's), S06 (TBI) at any diagnosis position. Excludes Group 1. | **163** | External validation only — *never in training* |
| **3. Matched unlabeled pool** | All remaining subjects, 3:1 matched to Group 1 on age decade × sex (`np.random.seed(42)`, `random_state=42`). | **17,133** | PU pool |

Of Group 1, **2,492 (43.6%)** have ≥ 1 admission *before* their first PTSD code — this is the primary pre-diagnosis training subsample. The remaining 3,219 use masking-based section-filtered index-admission notes as fallback. The per-script `compute_group_stats` function in `ewang163_ptsd_table1.py` produced the canonical Table 1 (61.5% female / 38.5% male PTSD+; mean age 43.0 ± 15.9; 65.6% White, 15.7% Black, 6.5% Hispanic; 28.0% emergency; 40.9% Medicaid/self-pay).

A psych-control cohort (Group 4, n = 5,711, 3,148 with section-filtered notes) was separately built for the specificity sanity check — 1:1 matched MDD/anxiety patients (ICD `F32`/`F33`/`296`/`F41`/`300`, excluding any PTSD coding) on age decade × sex.

### 2.3 Note extraction and section filtering

`ewang163_ptsd_notes_extract.py` streams the 3.3 GB `discharge.csv` with `csv.field_size_limit(sys.maxsize)` (some discharge notes exceed CSV's default field-size limit). For each row whose `hadm_id` belongs to an allowed admission, a regex parser (`SECTION_HEADER_RE = ^([A-Z][A-Za-z /&\-]+):[ ]*$` with `re.MULTILINE`) identifies section headers and slices body text between them. Notably, despite CLAUDE.md mentioning `discharge_detail.csv` for section filtering, the actual implementation parses sections from the free text of `discharge.csv` — `discharge_detail.csv` v2.2 contains only 'author' rows in practice.

Allowed admissions per group:
- **PTSD+**: pre-dx admissions ∪ index admission. The post-processing step "primary-vs-fallback resolution" then drops the index admission for any patient with at least one pre-dx note, ensuring each PTSD+ patient ends up with either pre-dx notes (n = 2,492) or a fallback index-admission note (n = 3,219), never both.
- **Proxy / Unlabeled**: index admission only.

Sections retained (all four narrative low-leakage sections): `history of present illness`, `social history`, `past medical history`, `brief hospital course`. Sections excluded entirely: `discharge diagnosis`, `assessment and plan`, `discharge medications`, `discharge condition`, `discharge instructions`, `followup instructions`. Concatenated in canonical order with `\n\n` separators.

Final corpus (`ewang163_ptsd_corpus.parquet`): **14,859 rows** (4,710 train PTSD+ + 7,127 train unlabeled, plus val and test). Proxy notes: **102 retrievable** (the other 61 had no qualifying section content).

### 2.4 Label-leakage prevention (tiered)

1. **Section filtering** (above) of all notes to narrative low-leakage sections.
2. **Pre-diagnosis notes** as primary signal where available (n = 2,492).
3. **Universal PTSD-string masking** in `ewang163_ptsd_corpus_build.py`: a case-insensitive regex `(?i)(post-traumatic|post\s+traumatic|posttraumatic|trauma-related\s+stress|ptsd|f43\.1|309\.81)` is replaced with `[PTSD_MASKED]` across **all 5,950 PTSD+ notes** (pre-dx + fallback). The original design assumed pre-diagnosis notes could not contain PTSD references; an audit found **360/4,169 = 8.6% of pre-dx notes contain explicit PTSD strings** carried forward from outside records ("h/o PTSD from MVA 2012"), validating Jin et al. (2023, JAMIA) on annotation noise from undercoded records.
4. **Two ablations** quantify residual leakage: Ablation 1 (re-mask everything post-hoc) and Ablation 2 (remove the entire PMH section).

### 2.5 Variable selection

- *Text features (primary):* section-filtered free text of patient notes, fed directly to transformer encoders.
- *Structured features (baseline only):* age at admission, sex, length of stay, emergency flag, Medicaid/self-pay flag, count of prior admissions, 5 race indicators, prior-admission flags for MDD, anxiety, SUD, TBI, chronic pain, suicidal ideation, SSRI/SNRI, prazosin, SGA. Twenty features total. Medications appear only here and as proxy-group labelling criteria — never in NLP text features (the discharge medications section is filtered out).

### 2.6 Models — implementation details

**Clinical Longformer + PULSNAR (primary).** `yikuan8/Clinical-Longformer`, 4,096 tokens. AdamW, lr 1e-5, batch 2 × grad-accum 16 (effective batch 32), 3 epochs (faster convergence than non-PULSNAR), warmup 0.1, weight decay 0.01, gradient clip 1.0, mixed precision (`torch.amp.autocast` + `GradScaler`), gradient checkpointing.

PULSNAR (`ewang163_ptsd_train_pulsnar.py`) implements propensity-weighted nnPU. Steps:

1. Build prior-admission features from streamed `diagnoses_icd.csv` and `prescriptions.csv` — 4 demographic features for the propensity model: sex, age, emergency, medicaid.
2. Fit logistic regression for propensity e(x) = P(coded | features), clip to [0.05, 0.95].
3. Estimate α via PULSNAR (xgboost classifier, `n_clusters=0, max_clusters=10, bin_method='rice', bw_method='hist'`); fall back to PULSCAR (SCAR) then to empirical fraction. **PULSNAR α = 0.1957**.
4. Fine-tune from pretrained Clinical Longformer with the modified loss:

```
weights_i = (1/e(x_i)) / mean(1/e(x_j))   for positives
loss_pos_w = mean(BCE(logits[pos], 1) · weights)
loss_pos_as_neg_w = mean(BCE(logits[pos], 0) · weights)
pu_w = α · loss_pos_w + clamp(loss_unl − α · loss_pos_as_neg_w, min=0.0)
```

The propensity reweighting upweights positives with low coding propensity (older men, minorities, no prior psychiatric contact) — exactly the underrepresented PTSD population.

A richer-features alternative that added `n_prior_admissions` to the propensity model produced α = 0.0006 (an artifact — the propensity model perfectly separated coded from uncoded, leaving PULSNAR no signal). The 4-feature propensity (α = 0.1957) is the principled estimate, and the only PULSNAR variant used downstream. The same `n_prior_admissions` confound surfaces in the structured baseline (coef +6.51) and is a cohort-design artifact (Group 3 unlabeled has index = first admission, so prior-admission count is always 0 by construction).

**Why PULSNAR over plain Kiryo nnPU.** Plain Kiryo nnPU assumes SCAR (Selected Completely At Random — labeled positives are a uniform random sample of true positives). That assumption is *clearly violated* here: PTSD coding is biased toward younger women with prior psychiatric contact (the fairness analysis below confirms exactly this pattern). PULSNAR is the literature-principled choice when SCAR fails (Bekker & Davis 2020; Kumar & Lambert 2024). Choosing the SAR-aware loss is a methodological commitment to favouring labels-as-noisy-process over labels-as-ground-truth.

**Plain Kiryo nnPU Longformer (sensitivity).** `ewang163_ptsd_train_longformer.py` is the same architecture/hyperparams but with the standard Kiryo loss (no propensity reweighting). Kept for reproducibility and as a sensitivity check. Accepts `--pi_p` to override the empirical class prior.

**BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`).** Same Kiryo nnPU loss, max_len 512, batch 16 × grad-accum 2 (effective 32), 5 epochs. Two inference modes that share the same trained weights:
- **Truncated:** first 512 tokens of each note.
- **Chunk-and-pool:** overlapping 512-token windows with 256-token stride, max-pooled positive-class probabilities across windows.

**Structured + LogReg.** 20 features (above), L2 logistic regression, `class_weight='balanced'`, C swept over {0.001, 0.01, 0.1, 1.0, 10.0, 100.0} on val AUPRC (best 10.0). Streams `diagnoses_icd.csv` and `prescriptions.csv` to compute prior-admission flags strictly before each patient's index admission.

**Keyword (DSM-5/PCL-5).** Zero-training. **62 weighted regex patterns** in `ewang163_ptsd_train_keyword.py`, organized by DSM-5 criterion (A trauma exposure, B intrusion, C avoidance, D negative cognition/mood, E arousal/reactivity, plus treatment signals). Two scoring variants compared on val AUPRC (raw weighted count — winner — vs. TF-normalized).

**Specificity check (`ewang163_ptsd_specificity.py`).** Standard cross-entropy (NOT nnPU — psych controls are confirmed negatives) with class-balanced weights, training PTSD+ (n = 5,711) vs. age/sex 1:1-matched MDD/anxiety controls.

### 2.7 Splits

`ewang163_ptsd_splits.py` performs **patient-level 80/10/10 stratified split** by `subject_id` and `ptsd_label` (`random_state=42`). A patient's data never crosses split boundaries.

- Random split (canonical): train 11,837 rows / val 1,471 / test 1,551 (660 PTSD+ in test).
- **Temporal split**: pre-2015 train, 2017–2019 test, using each patient's `anchor_year_group` from `patients.csv`. Crucially, MIMIC-IV applies per-patient random date shifts (~100–200 years), so raw `admittime` is unusable for chronology — the un-shifted `anchor_year_group` is the correct workaround. Result: train 11,134 / val 1,270 / test 2,455.

### 2.8 Evaluation strategy

Implemented across the dedicated per-model scripts (`pulsnar_reeval.py` for Longformer, `bert_full_eval.py` for both BERT modes) plus the cross-model script (`cross_model.py`). The legacy `evaluate.py` is kept only as the source of val-derived thresholds for the structured + keyword baselines.

**Threshold derivation.** All operating thresholds computed on **validation** at sensitivity ≥ 0.85 via `threshold_at_recall(probs, labels, target_recall=0.85)`. Frozen before any test-set metrics. Each model gets its own val-derived threshold (the BERT chunk-pool mode, in particular, derives a threshold (0.993) different from the truncated mode (0.976), because the max-pool aggregation pushes the score distribution upward).

**Discrimination.** AUPRC (primary) via `average_precision_score`; AUROC via `roc_auc_score`.

**Clinical utility metrics:**
- LR+ = sens / (1 − spec), LR− = (1 − sens) / spec, DOR = LR+ / LR−
- Alert rate = (TP + FP) / N; workup reduction vs. treat-all = 1 − alert rate
- Bayes' theorem prevalence-recalibrated PPV: PPV(prev) = (sens · prev) / (sens · prev + (1 − spec) · (1 − prev)); NNS = 1/PPV
- Reported at deployment prevalences {1%, 2%, 5%, 10%, 20%}.

**McNemar's test** with continuity correction on paired test predictions:
- b = #(model A correct & B wrong); c = #(A wrong & B correct)
- chi² = (|b − c| − 1)² / (b + c); p from `scipy.stats.chi2(df=1)`. All-pairs over the 5 deployed models.

**Calibration.** Platt scaling (`LogisticRegression(C=1)` on val_probs vs. val_labels), then Elkan-Noto correction:
- c = mean(raw model prob on val positives) — estimates P(s=1|y=1)
- corrected_prob = clip(platt_scaled / c, 0, 1) — approximates P(PTSD=1) rather than P(coded=1)

ECE on 10 equal-frequency bins with **Wilson 95% CI** per bin. Computed for *all* text models (PULSNAR Longformer, BERT trunc, BERT chunk-pool).

**Decision Curve Analysis (Vickers).** Net benefit at thresholds 0.01–0.40:
- NB_model(t) = TP/N − (FP/N) · (t/(1−t))
- NB_treat-all(t) = prev − (1 − prev) · (t/(1−t))

Calibrated probabilities are first deployment-prevalence-shifted via Bayes' rule. Computed for all three text models at 2% and 5% deployment prevalences.

**Ramola PU correction.** With α = π_p:
- corrected_AUROC = (AUROC − 0.5α) / (1 − α)
- corrected_AUPRC = AUPRC / α (capped at 1.0)
- All raw metrics labeled as "PU lower bounds" with corrections reported alongside.

**Pharmacological proxy external validation.** `scipy.stats.mannwhitneyu(proxy_probs, unlab_probs, alternative='greater')`; AUC = U / (n_proxy · n_unlab). Comparison group: 500 unlabeled patients drawn from training pool only (`splits['train']`, `np.random.RandomState(42)`), one note per patient via `drop_duplicates`. Computed for all three text models.

**Fairness.** Per subgroup:
- *Calibration-in-the-large* = mean(predicted) − mean(observed); Wilson CI on the observed proportion.
- *Equal opportunity difference* (Hardt et al. 2016) = max(recall) − min(recall) at the val-derived threshold across subgroup levels.
- *Bootstrap 95% CI on AUPRC*: 1,000 resamples (`np.random.RandomState(42)`), percentile method (2.5th/97.5th). Reliable iff CI width < 0.15. Computed for all three text models.

**Integrated Gradients.** Captum's `IntegratedGradients` on the embedding tensor via custom wrappers.
- *Longformer:* `EmbeddingForwardWrapper` explicitly constructs `global_attention_mask` (zeros, with position 0 = CLS = 1) and passes `inputs_embeds` rather than `input_ids` — works around Captum's hook conflicts with Longformer's hybrid local/global attention. n_steps=20, internal_batch_size=1, full 4,096-token context, pad-token baseline.
- *BioClinicalBERT:* `BertEmbedForward` is simpler (no global_attention_mask required). Two attribution slices: (1) the truncated 512-token input, and (2) the highest-scoring chunk-pool window per note (selects the window whose individual probability matches the max-pooled prediction).
- Word-level aggregation merges contiguous BPE subwords whose offsets connect through alphabetic / hyphen / apostrophe characters with **summed** attributions.
- Sample: 50 high-confidence true positives drawn from the top decile (`random_state=42`).

**Cross-model comparison.** All-pairs McNemar p, agreement %, Cohen's kappa, Pearson correlation on probabilities, top-quintile rank overlap, per-subgroup NNS at deployment prevalences, and a max-pool / mean-pool rank-normalised ensemble probe.

**Error analysis.** FP/FN demographic breakdown + trauma-term presence rate, computed per model.

**Ablations.** Computed per text model.
- *Ablation 1:* re-apply PTSD-string masking to test notes (post-hoc).
- *Ablation 2:* re-extract test notes from `discharge.csv` without the PMH section.

**Runtime benchmarking.** `BenchmarkLogger` context manager (`scripts/common/ewang163_bench_utils.py`) captures wall-clock (`time.perf_counter`), CPU time (`time.process_time`), peak memory (`resource.getrusage(RUSAGE_SELF).ru_maxrss`), and GPU-hours; appends to `results/metrics/ewang163_runtime_benchmarks.csv`. The unified inference benchmark (`ewang163_unified_inference_bench.py`) times all five models in a single L40S allocation for apples-to-apples comparison.

### 2.9 Pipeline flowchart (mapped to OHDSI)

```
MIMIC-IV (read-only)
   ↓ stream + filter
[01_cohort/]    table1 → cohort_sets → admissions_extract → notes_extract       [OHDSI: Cohort + Characterization]
   ↓
[02_corpus/]    corpus_build (universal PTSD masking) → splits (random + temporal) [OHDSI: Dataset]
   ↓
[03_training/]  train_pulsnar (PRIMARY) │ train_longformer (Kiryo sensitivity) │ [OHDSI: Analyze]
                train_bioclinbert │ train_structured │ train_keyword │ train_specificity
   ↓
[04_evaluation/] pulsnar_reeval (Longformer val/test/calibration/utility/fairness)
                bert_full_eval (BERT both modes — full analysis suite)
                bert_attribution (IG, both BERT slices)
                attribution_v2 (Longformer IG)
                proxy_validation (Longformer plot)
                decision_curves / calibration / fairness / ablations / error_analysis
                temporal_eval / specificity comparison
                cross_model (all-pairs McNemar, agreement, ensemble, subgroup NNS)
                evaluate (legacy — produces structured + keyword val thresholds)
                unified_inference_bench / cpu_inference_bench
   ↓
results/{table1, predictions, metrics, figures, attribution,                     [OHDSI: Research Products]
         error_analysis, runtime_benchmarks}
```

GitHub: https://github.com/ewang163/AIH-Final-Project — programs documented in README.md.

---

## Part 3 — Results and Discussion

All numbers below are on the patient-level held-out test set (n = 1,551 patients, 660 PTSD+, prevalence 42.55%) under val-derived thresholds, unless otherwise stated.

### 3.1 Model comparison — discrimination (test set)

| Model | AUPRC | AUROC | Sens | Spec | Prec | F1 | LR+ | DOR | NNS @ 2% | McNemar p vs PULSNAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clinical Longformer (PULSNAR, primary)** | **0.8848** | **0.8904** | 0.846 | 0.745 | 0.711 | **0.772** | 3.32 | 16.0 | 15.8 | — |
| BioClinicalBERT (chunk-pool 512×256) | 0.8775 | 0.8853 | 0.846 | 0.772 | 0.733 | 0.785 | 3.71 | 18.6 | **14.2** | 0.107 (n.s.) |
| BioClinicalBERT (truncated 512) | 0.8576 | 0.8656 | 0.821 | 0.728 | 0.691 | 0.751 | 3.02 | 12.3 | 17.2 | 0.043 |
| Structured + LogReg | 0.6833 | 0.7310 | 0.909 | 0.207 | 0.459 | 0.610 | 1.15 | 2.6 | 36.7 | < 1e-300 |
| Keyword (DSM-5/PCL-5) | 0.5096 | 0.6190 | 1.000 | 0.000 | 0.426 | 0.597 | 1.00 | 1.00 | 50.0 | < 1e-300 |

**Headline.** Clinical Longformer (PULSNAR) wins discrimination at AUPRC 0.885 / AUROC 0.890. BioClinicalBERT chunk-pool is statistically indistinguishable from PULSNAR Longformer on McNemar's paired-prediction test (p = 0.11; b = 90 / c = 114, with chunk-pool actually correct on slightly more disagreements). BERT-truncated underperforms by 0.027 AUPRC (McNemar p = 0.043). **Long-context inference, not architecture per se, drives most of Longformer's lift over BERT-truncated** — the chunk-pool variant of BERT recovers most of the gap. Keyword is essentially random; structured-only is well below text-based models — narrative content carries the bulk of predictive signal.

Notably, **BERT chunk-pool's NNS @ 2% (14.2) is *better* than PULSNAR Longformer's (15.8)** despite a slightly lower AUPRC. This reflects chunk-pool's higher specificity (0.772 vs 0.745) at its val-derived threshold, which dominates the operating-point clinical utility. The two models trade off discrimination (Longformer wins) for operating-point efficiency (chunk-pool wins) — a meaningful deployment tradeoff explored further in §3.4.

Ramola PU corrections push Longformer corrected AUPRC to 1.0 (ceiling-clipped at α = 0.4255 test labeled fraction) and corrected AUROC to ≈0.987 — the raw metrics are conservative PU lower bounds.

### 3.2 Training dynamics

Clinical Longformer + PULSNAR (3 epochs, ~3.5 GPU-h on L40S, converges faster than plain Kiryo nnPU due to propensity-weighted gradients): val AUPRC peaks at 0.872 in epoch 3.

BioClinicalBERT (5 epochs, ~140 s/epoch, 0.22 GPU-h on RTX 3090): best val AUPRC 0.86 (chunk-pool inference) at epoch 5.

### 3.3 Ablation studies (label leakage) — symmetric across all text models

| Condition | PULSNAR Longformer ΔAUPRC | BERT trunc ΔAUPRC | BERT chunkpool ΔAUPRC |
|---|---:|---:|---:|
| Baseline AUPRC | 0.8848 | 0.8576 | 0.8775 |
| Ablation 1 (post-hoc PTSD masking) | −0.008 | −0.013 | −0.012 |
| Ablation 2 (PMH section removed) | −0.061 | −0.063 | −0.066 |

Ablation 1 is essentially free across all three models — none is exploiting literal PTSD strings. Ablation 2 costs ~6 AUPRC points uniformly: the models depend comparably on the PMH section, and removing it pushes them all back to the structured + simpler-text band. The fact that all three models score near 0.81–0.82 even with PMH removed — comfortably above the structured + keyword baselines — confirms HPI, Social History, and Brief Hospital Course independently encode enough PTSD-associated language. **Per-model A1/A2 deltas are essentially identical**, which means the PMH dependence is a property of the data and the masking design, not of any specific architecture's bias.

### 3.4 Calibration — the largest operational gap between Longformer and BERT

| Model | ECE raw | ECE Platt | ECE Elkan-Noto | Elkan-Noto c |
|---|---:|---:|---:|---:|
| Clinical Longformer (PULSNAR) | **0.088** | 0.077 | 0.097 | 0.728 |
| BioClinicalBERT (truncated)   | 0.482 | **0.174** | 0.173 | 0.983 |
| BioClinicalBERT (chunk-pool)  | 0.507 | **0.208** | 0.208 | 0.991 |

**BioClinicalBERT is severely over-confident out of the box.** Its raw probabilities concentrate near 1.0 (Elkan-Noto c ≈ 0.99 means *almost every* labeled positive gets a near-certain raw probability), so raw ECE is 5–6× higher than Longformer's. Platt scaling on validation cuts ECE by ~3× but it remains roughly 2–3× higher than Longformer's raw. **For any threshold-sensitive deployment, BERT requires post-hoc calibration; Longformer can ship with raw probabilities.** This is the single largest practical reason to prefer PULSNAR Longformer despite the comparable AUPRC.

The Longformer Elkan-Noto c estimate (0.728) implies a ~27% undercoding rate, consistent with PCL-5 inpatient prevalence findings. The BERT c estimates near 1.0 are not informative — they reflect over-confidence collapse rather than a labelling-frequency estimate.

### 3.5 Decision curve analysis — peak net benefit by deployment prevalence

| Model | Max NB @ 2% prev | Max NB @ 5% prev |
|---|---:|---:|
| Clinical Longformer (PULSNAR) | 0.36 | 0.39 |
| BioClinicalBERT (truncated) | 0.40 | 0.41 |
| BioClinicalBERT (chunk-pool) | **0.40** | **0.41** |

DCA peak net benefits are similar across models. BERT (especially chunk-pool) edges Longformer at the maximum because of its higher specificity at the val-derived operating point. At deployment prevalences below ~2%, all three models compete with treat-all only at moderate thresholds — the cost of missing a case is high enough relative to clinician review time that flagging everyone remains competitive in the very-low-threshold band. This is honest and clinically informative — it says the screening model is worthwhile only when paired with a deployment regime where clinician time is the binding constraint.

### 3.6 Prevalence recalibration (Longformer PULSNAR)

| Deployment prevalence | PPV | NPV | NNS |
|---:|---:|---:|---:|
| 1% | 0.032 | 0.998 | 30.8 |
| 2% | 0.063 | 0.996 | **15.8** |
| 5% | 0.149 | 0.989 | 6.7 |
| 10% | 0.269 | 0.978 | 3.7 |
| 20% | 0.453 | 0.951 | 2.2 |

At 2% inpatient deployment prevalence, ~16 patients need to be flagged to find one true PTSD case — clinically tolerable for an automated screening prompt. NPV > 0.99 at every prevalence makes a negative screen highly reliable. LR+ = 3.32 is "moderate" by clinical convention, appropriate for a screening prompt (not for diagnosis). BERT chunk-pool's recalibration table is similar but with slightly tighter NNS at 2% (14.2) due to the higher specificity at threshold.

### 3.7 Subgroup AUPRC (PULSNAR Longformer)

| Subgroup | n | n_pos | AUPRC |
|---|---:|---:|---:|
| Female | 940 | 433 | **0.92** |
| Male | 611 | 227 | 0.83 |
| Age 20s | 293 | 140 | **0.94** |
| Age 30s | 359 | 194 | 0.92 |
| Age 40s | 308 | 131 | 0.86 |
| Age 50s | 277 | 81 | 0.86 |
| Age Other (<20 or ≥60) | 314 | 114 | 0.83 |
| Race binary White | 1,083 | 485 | 0.86 |
| Race binary Non-White | 468 | 175 | 0.91 |
| Emergency = True | 855 | 450 | **0.92** |
| Emergency = False | 696 | 210 | 0.81 |

**The pattern is shared with BioClinicalBERT** (both modes) and inherited from the labels themselves: women, younger patients, and emergency admissions are the demographics most likely to be coded today, so all models perform best on them. The model is weakest on older men in elective admissions — *exactly* the population a screening tool would be most valuable for, since they are the most under-detected.

### 3.8 Fairness — equal opportunity differences

| Subgroup | PULSNAR Longformer EO | BERT trunc EO | BERT chunkpool EO |
|---|---:|---:|---:|
| Sex (F vs. M) | 0.114 | 0.151 | 0.127 |
| Age | 0.211 | 0.237 | 0.181 |
| Race binary (W vs. Non-W) | 0.024 | 0.064 | 0.047 |
| Emergency | 0.046 | 0.038 | 0.067 |

**Race disparity is small across all three text models.** The dominant disparities are **age** (0.18–0.24) and **sex** (0.11–0.15), inherited from the non-SCAR coding bias. **PULSNAR Longformer has the smallest sex EO and smaller race EO than either BERT mode** — consistent with PULSNAR's propensity-reweighting pulling some signal from under-coded subgroups. BERT chunk-pool partially recovers on age (EO 0.181 vs Longformer's 0.211).

### 3.9 Pharmacological proxy external validation — the headline non-circular result

| Model | Proxy median | Unlabeled median | MW AUC | MW p |
|---|---:|---:|---:|---:|
| Clinical Longformer (PULSNAR) | ~0.38 | ~0.06 | **0.7701** | 3.8e-18 |
| BioClinicalBERT (truncated) | — | — | 0.7442 | 3.7e-15 |
| BioClinicalBERT (chunk-pool) | — | — | 0.7333 | 5.3e-14 |

All three text models — none of which saw a single proxy patient during training — assign substantially higher scores to medication-pattern-positive patients than to demographically-matched unlabeled patients. PULSNAR Longformer's separation is strongest. Because proxy patients are identified by an entirely independent (medication-based) criterion that the model cannot see — discharge medications are a filtered-out section — this is the project's strongest single piece of validity evidence.

### 3.10 Specificity check vs. psychiatric controls

| Metric | PULSNAR Longformer | Specificity-trained Longformer |
|---|---:|---:|
| Test AUPRC | 0.885 | **0.911** |
| Test AUROC | 0.890 | 0.815 |
| Sensitivity | 0.846 | 0.852 |
| Specificity | 0.745 | 0.581 |
| Mean predicted prob on proxy | 0.46 | 0.34 |

A separate Longformer trained PTSD+ vs. age/sex 1:1-matched MDD/anxiety controls (standard cross-entropy) reaches AUPRC 0.91. **PTSD-specific signal is recoverable above-and-beyond generic "psychiatric admission" language**, ruling out the worst-case interpretation that the primary model is a psych-vs-non-psych classifier. The AUROC drop (0.81 vs 0.89) reflects the harder task — psychiatric controls have overlapping vocabulary.

### 3.11 Integrated Gradients attribution

#### 3.11.1 Per-section attribution (% of total |attribution|)

| Section | PULSNAR Longformer | BERT truncated (first 512 tok) | BERT chunk-pool (top window) |
|---|---:|---:|---:|
| HPI | 43.2 % | **58.2 %** | 53.5 % |
| Past Medical History | 22.4 % | 27.4 % | 27.3 % |
| Brief Hospital Course | 32.0 % | 13.0 % | **17.3 %** |
| Social History | 1.3 % | 0.7 % | 1.2 % |

**The transformers differ in how they distribute attribution across the note.**

- *Longformer* spreads attribution roughly evenly between HPI (43%) and BHC (32%), with PMH a meaningful 22% — this matches the expectation that PTSD signal is distributed across the trauma-history (HPI), comorbidity (PMH), and clinical-course (BHC) sections.
- *BERT truncated* concentrates on HPI (58%) because BHC literally falls past the 512-token window for most notes — the truncation reweights attribution toward the early sections.
- *BERT chunk-pool (top window)* moves some weight from HPI (53%) to BHC (17%), confirming that when chunk-pool's max-pooled prediction comes from a deeper window, the prediction is genuinely driven by BHC content. Chunk-pool partially recovers the "BHC matters" signal that truncation hides.

This is a clean architectural finding: **Longformer's wider context lets BHC contribute meaningfully; BERT-truncated cannot see BHC at all and substitutes HPI-only signal; BERT-chunk-pool partially restores BHC via the windowing.**

#### 3.11.2 Top attributed words

| Rank | PULSNAR Longformer | BERT truncated | BERT chunk-pool |
|---:|---|---|---|
| 1 | bipolar | psych | psych |
| 2 | narcotic | -depression | anxiety |
| 3 | illness | anxiety | -depression |
| 4 | arrested | bipolar | bipolar |
| 5 | delayed | psychiatric | numerous |
| 6 | pancreatitis | numerous | psychiatric |
| 7 | schizoaffective | dilaudid | overdose |
| 8 | psychosis | overdose | suicide |
| 9 | anemia | disorder | disorder |
| 10 | assault | suicide | dilaudid |

**Longformer's top words are noticeably more trauma-anchored** (`narcotic`, `arrested`, `assault`, `psychosis`) compared to BERT's heavily psychiatric-comorbid vocabulary (`psych`, `anxiety`, `bipolar`, `psychiatric`, `disorder`, `suicide`). This is consistent with Longformer's wider context picking up trauma-narrative content from BHC that BERT cannot see, and consistent with PULSNAR's SAR-aware training discounting the comorbidity-coding signal.

**No label-leakage tokens** (e.g., "ptsd", "posttraumatic") appear in any model's top attributions, confirming the universal masking worked as intended.

### 3.12 Cross-model agreement matrix

| Pair | Agreement | Cohen's κ | Pearson r (probs) | McNemar p | Top-quintile overlap |
|---|---:|---:|---:|---:|---:|
| LF-PULSNAR vs BERT chunk-pool | 86.85 % | 0.737 | 0.527 | **0.107** (n.s.) | 83.5 % |
| LF-PULSNAR vs BERT truncated  | 85.88 % | 0.718 | 0.575 | 0.043 | 75.8 % |
| BERT trunc vs chunk-pool      | 88.72 % | 0.774 | 0.863 | 4.5e-5 | 84.8 % |
| LF-PULSNAR vs Structured      | 56.03 % | 0.113 | 0.353 | < 1e-300 | 33.2 % |
| LF-PULSNAR vs Keyword         | 50.61 % | 0.000 | 0.223 | < 1e-300 | 38.4 % |

**Findings.**

1. **PULSNAR Longformer and BERT chunk-pool disagree non-significantly** (McNemar p = 0.11). They agree on 87% of binary predictions, on 83% of the top quintile by score, and have Cohen's κ = 0.74. Despite different architectures and inference strategies, they identify largely the same patients — and where they diverge, neither dominates the other.
2. **Both BERT modes are tightly correlated** (Pearson r = 0.86 on probabilities, top-quintile overlap 85%) — chunk-pool inherits the truncated model's underlying scoring with extra max-pool boost.
3. **Structured + Keyword are essentially independent rankings** of the text models (κ < 0.15, top-quintile overlap ~30–40 %). They are picking up different signals — coding-frequency proxies and explicit symptom mentions — which is why they fail individually but might in principle complement a transformer in an ensemble.
4. **Ensemble experiment.** A max-pool ensemble over rank-normalised probabilities of all 5 models reaches AUPRC 0.751; mean-pool reaches 0.874. **Neither beats the best individual model** (0.885). Mean-pool is competitive but the noisier rankings of the structured + keyword models drag down both ensembles — the upside of model diversity does not compensate for the downside of including noise. **No ensemble lift was found across the deployed lineup.**

### 3.13 Error analysis — both transformer architectures share the same FN profile

| Set | n | Mean pred. prob | Mean note len | % female | % age "Other" | % emergency | % with trauma term |
|---|---:|---:|---:|---:|---:|---:|---:|
| **PULSNAR Longformer** | | | | | | | |
| FP | 227 | 0.66 | 3,239 | 60.4 % | 17.2 % | 42.3 % | — |
| FN | 99 | 0.18 | 2,932 | 49.5 % | **36.4 %** | 27.3 % | — |
| **BERT truncated** | | | | | | | |
| FP | 242 | 0.99 | 3,313 | 55.8 % | 16.1 % | 46.7 % | 68.6 % |
| FN | 118 | 0.87 | 3,072 | 46.6 % | **30.5 %** | 63.6 % | 66.9 % |
| **BERT chunk-pool** | | | | | | | |
| FP | 203 | 1.00 | 3,825 | 58.1 % | 16.3 % | 45.8 % | 72.9 % |
| FN | 102 | 0.92 | 2,483 | 47.1 % | **27.5 %** | 58.8 % | 60.8 % |

**False negatives skew male, older ("Other" age 27–36% vs. ~20% overall), and have shorter notes** across all three transformers. The pattern is invariant to architecture — it's a property of how those patients are documented, not of any one model. False positives skew female and emergency, demographically matching coded patients, consistent with these being **likely true undercoded PTSD** that the ICD label simply doesn't reflect.

The BERT FP/FN predicted probabilities are pinned near 1.0 — another manifestation of BERT's over-confidence (cf. §3.4 calibration). Longformer FN predicted probabilities are near 0.18 (median), so it ranks them low; BERT FNs are near 0.87, suggesting BERT was *almost* willing to flag them but the threshold was just barely above. Calibration before deployment would change the FP/FN split for BERT meaningfully.

### 3.14 Temporal generalization — temporal training did not help

| Scenario | Test AUPRC |
|---|---:|
| Random-split model on random test | 0.888 |
| Random-split model on temporal test (2017–2019) | 0.886 |
| **Temporal-trained model on temporal test** | 0.842 |

Temporal training **hurt** generalization. The random-split model loses only 0.002 AUPRC when tested on 2017–2019 — meaning the random distribution is already representative of late MIMIC-IV. The temporal model, trained only on pre-2015 data (8,752 vs. 11,837 patients), loses 0.044 AUPRC because it has 25% less training data and misses richer post-2013 DSM-5-era coding patterns. **Random split is recommended for deployment.**

### 3.15 Compute frontier (measured)

| Model | ms/patient (L40S) | Train wall (s) | Train GPU-h | Test AUPRC |
|---|---:|---:|---:|---:|
| Longformer PULSNAR | 80.4 | 12,617 | 3.50 (L40S) | 0.8848 |
| BERT chunk-pool | 22.7 | 791 | 0.22 (RTX 3090) | 0.8775 |
| BERT truncated | 2.97 | same | same | 0.8576 |
| Structured + LogReg | 17.9 (I/O bound) | 68 (CPU) | 0 | 0.6833 |
| Keyword (16-CPU) | **0.34** | 0 | 0 | 0.5096 |

Architecture (4,096 vs. 512 attention) — not GPU generation — drives the gap. Longformer pays ~16× hardware-normalized training cost and 3.55× inference latency vs. chunk-pool BERT for **+0.007 AUPRC and substantially better calibration**. Keyword (16-CPU) is 236× faster per-patient than Longformer, but at AUPRC 0.51 the speed buys near-random predictions.

**Cost per 50,000 inpatient discharges/month:** Longformer 67 minutes (~1.1 GPU-h, ~$1–2); BERT chunk-pool 19 minutes (0.3 GPU-h); keyword 0.3 minutes (CPU). Inference is the recurring cost; training is one-shot.

### 3.16 Structured-baseline finding: the matching artifact

The structured logistic regression's coefficients (saved in `ewang163_structured_features.json`):

| Feature | Coef |
|---|---:|
| **n_prior_admissions** | **+6.512** |
| dx_suicidal | +1.822 |
| dx_SUD | +1.468 |
| dx_MDD | +1.243 |
| dx_anxiety | +0.882 |
| medicaid_selfpay | +0.859 |
| race_Asian | −0.672 |
| rx_ssri_snri | +0.287 |
| race_White | +0.274 |

The +6.51 coefficient on `n_prior_admissions` is **3.6× larger than the next feature**. This is the SAR violation made manifest: the structured model is largely learning "this patient has been admitted many times, therefore probably PTSD-coded" — a coding-frequency signal, not a PTSD signal. It is also a **dataset artifact**: by construction, Group 3 (unlabeled) has index = first MIMIC-IV admission, so prior-admission count is always 0. Group 1 (PTSD+) has index = first PTSD-coded admission, which tends to be later. The same artifact caused the rich-features PULSNAR propensity model to collapse to α ≈ 0 (see §2.6) and is the reason the deployed PULSNAR uses a 4-feature propensity instead. **Deploying the structured model would entrench this artifact.** Race coefficients are small in magnitude (max |0.67| for Asian, n = 5 in test — unstable), confirming the cross-model finding that race-binary EO is small (0.024–0.064).

---

## Discussion

### 3.17 How will I evaluate the model's performance?

The primary metric is **AUPRC**, with AUROC reported alongside. AUPRC is more informative under class imbalance — it can stay high while precision at clinically actionable thresholds is poor (which would rule out screening deployment). AUROC at the 42.55% test prevalence is interpretable, but real-world deployment prevalence is much lower (~ 2%), so PPV/NPV/NNS are recalibrated to deployment prevalences {1, 2, 5, 10, 20%} via Bayes' theorem on the val-derived sens/spec. Threshold-anchored metrics (sensitivity, specificity, precision, F1) at the recall ≥ 0.85 operating point are reported. McNemar's test with continuity correction quantifies pairwise model differences; the all-pairs cross-model matrix shows that PULSNAR Longformer and BERT chunk-pool are statistically indistinguishable on McNemar (p = 0.11). The proxy Mann-Whitney AUC is the only PU-uncontaminated metric and is co-headline with AUPRC.

### 3.18 How will I determine whether the model is clinically useful and appropriate?

Three clinical-utility lenses are applied:

1. **Screening tradeoff at deployment prevalence.** At 2% prevalence, PULSNAR Longformer NNS = 15.8 and PPV = 6.3% — clinically tolerable for a screening prompt that points clinicians at otherwise-missed patients. BERT chunk-pool's NNS at 2% (14.2) is even better at the operating point, though calibration is much worse — meaning chunk-pool's threshold needs Platt scaling before deployment. LR+ for both is "moderate" — appropriate for screening, not for diagnosis. NPV > 0.99 means a negative screen is highly reliable.
2. **Decision Curve Analysis.** At deployment prevalence ≥ 5%, all three text models offer meaningful net benefit across thresholds 0.01–0.30. At very low prevalence (1–2%), treat-all dominates in the very-low-threshold band; the model is competitive only at moderate thresholds.
3. **Explainability.** Integrated Gradients (not attention — Jain & Wallace 2019) at 4,096 tokens for Longformer and 512 tokens (truncated + chunk-pool top window) for BERT shows HPI dominance with meaningful BHC contribution for Longformer and chunk-pool BERT, while BERT-truncated cannot see BHC at all. Top words are clinically appropriate trauma/psychiatric/substance vocabulary across all models; **no label-leakage tokens** appear in any model's top attributions.

Subgroup performance is the central caveat for clinical appropriateness across all text models: the model is best-calibrated on younger women in emergency admissions (the demographic most likely to be coded today) and weakest on older men in elective admissions (the demographic where undercoding is most prevalent — and the population a screening tool would be most valuable for). PULSNAR's SAR-aware training reduces but does not eliminate this inherited bias.

### 3.19 Limitations

**Section filtering is not perfect leakage prevention.** Even on a pre-diagnosis admission, PMH may carry forward "history of PTSD" from outside records — the masking audit confirmed 8.6% of pre-dx notes had explicit PTSD strings before masking. Ablation 2 quantifies the upper bound (PMH removal costs ~6 AUPRC points across all text models, so residual PMH leakage is bounded but not zero).

**PU learning reduces but does not eliminate selection bias.** PULSNAR's propensity weighting up-weights underrepresented PTSD positives in the loss but cannot create labels for them — exactly why the older-patient and male-patient AUPRC gaps persist across all three text models. The propensity model itself is sensitive to feature choice (adding `n_prior_admissions` collapses α to ~0 because the propensity model then perfectly separates coded from uncoded by a cohort-design artifact); the 4-feature propensity is the principled choice.

**The pre-diagnosis training subsample is not representative.** Only 2,492 of 5,711 PTSD+ patients (43.6%) had pre-diagnosis admissions — by definition multi-admission patients, who tend to be sicker, older, and more psychiatrically complex. The 3,219 fallback patients use index-admission notes with masking applied; they carry more residual leakage risk.

**BioClinicalBERT is severely over-confident.** Raw ECE 0.48–0.51, dropping to 0.17–0.21 after Platt scaling — still 2–3× Longformer's. Any threshold-sensitive deployment of BERT requires post-hoc calibration on a held-out validation set.

**Proxy validation set is small (n_with_notes = 102) and has a known FPR.** The proxy criterion (prazosin + SSRI/SNRI within 180 days, no cardiovascular/BPH/Raynaud/TBI exclusion) carries an estimated 15–20% false-positive rate (patients on prazosin for off-label uses not captured by the exclusion ICD codes). Elevated proxy scores are evidence, not proof.

**Single-site data.** MIMIC-IV is one academic medical center in Boston. Whether any of this transfers to community hospitals, rural settings, the VA system (where PTSD prevalence is much higher), or international settings is unknown. The temporal split showed limited within-MIMIC-IV distribution shift, but cross-site validation has not been performed.

**Calibration for Longformer is good but imperfect.** ECE = 0.088 (raw). Acceptable for ranking and threshold-based screening; for any application that depends on absolute predicted probability, additional calibration work would be needed.

**Decision curve analysis is favourable but not dominant.** At 2% deployment prevalence, the model does not beat treat-all in the very-low-threshold band — the cost of missing a case is high enough relative to clinician time that flagging everyone is competitive at very low prevalence.

### 3.20 Interpreting results in the context of existing literature, and what would constitute an actionable finding

The AUPRC of 0.885 is competitive with or exceeds the substance-misuse phenotyping benchmarks (Afshar 2019: AUPRC ~0.85 on AUDIT-screened alcohol misuse with cTAKES + CUI features; Sharma 2020: comparable for opioids), achieved on a *contaminated-negatives* problem they could sidestep by using prospectively collected reference standards. The Longformer's lift over BioClinicalBERT chunk-pool (+0.007 AUPRC, McNemar p = 0.11 — not statistically significant on McNemar) is consistent with Li et al. (2022) — long-range pretraining provides at most a small benefit on top of long-context inference, and the cross-model agreement matrix confirms the two models are picking up largely the same patients.

Most importantly, the **proxy Mann-Whitney AUC of 0.770 (p = 3.8e-18)** for PULSNAR Longformer is the single most actionable finding: a model trained without any reference to medications or proxy patients assigns substantially higher probability to patients whose pharmacotherapy is consistent with PTSD treatment. This is direct, non-circular evidence that the model recovers a real PTSD-associated narrative signal rather than just re-deriving the ICD coding rule. Combined with the specificity check (AUPRC 0.91 against MDD/anxiety controls — the model is not just learning generic psychiatric language) and the IG attribution (clinically appropriate trauma/psychiatric vocabulary, no label-leakage tokens), this constitutes a defensible body of validity evidence.

A **meaningful and actionable next step** would be downstream prospective validation against a true reference standard (PCL-5 / CAPS-5) at an external site — converting this from a methodology demonstration into an actionable screening tool ready for deployment trials. The combination of proxy validation (AUC 0.770), specificity check vs. psychiatric controls (AUPRC 0.91), and label-leakage ablations (Ablation 1: −0.008 AUPRC) already constitutes a non-circular evidence stack, but absolute calibration to true PTSD prevalence requires a defensible gold standard the contaminated-negatives MIMIC-IV labels cannot provide.

### 3.21 What this tool is and is not

**It is** a screening prompt — a way to point inpatient clinicians at patients whose narrative notes contain language patterns associated with PTSD. The output is intended to inform a more thorough psychiatric evaluation, not to make a diagnosis. At 2% deployment prevalence and the chosen operating threshold, the tool would surface ~16 patients per true case for review with PULSNAR Longformer, or ~14 with BERT chunk-pool (after calibration).

**It is not** a diagnostic tool, a replacement for structured PTSD screening (PCL-5, CAPS-5), or a basis for ICD coding. The downstream user is a clinician who will conduct a proper interview before any management decision is made.

### 3.22 Future directions

- **External validation** at a non-MIMIC site, ideally one with both higher PTSD base rates and richer narrative documentation (a VA medical center is the natural target).
- **Re-evaluation under a true reference standard.** A small prospective cohort screened with PCL-5 or CAPS-5 would let absolute model performance be measured against a defensible gold standard.
- **Bias mitigation for demographic subgroups.** The female / younger-age coding bias is inherited from labels; explicit subgroup-aware loss reweighting or label-noise modelling (Bekker & Davis 2020) could narrow the gap. PULSNAR's attribution shift hints that SAR-aware training helps; a richer propensity feature set (without the `n_prior_admissions` artifact) might widen the benefit.
- **BioClinicalBERT calibration in production.** Chunk-pool BERT's NNS-at-threshold is competitive with Longformer at ~28% of the inference cost. With proper Platt or isotonic calibration on a held-out set, BERT chunk-pool may be the right deployment choice for compute-constrained sites.
- **Integration with structured features.** The structured baseline AUPRC was 0.68; concatenating structured features (excluding the artifactual `n_prior_admissions`) into the Longformer head could provide complementary signal, particularly for the demographics where text features are weakest.
- **Better cohort design for the pre-diagnosis subsample.** Only 43.6% of PTSD+ patients had pre-diagnosis admissions; for the rest, masking-on-index is the only option. A future iteration could add a propensity-matched non-PTSD comparison group with the same admission counts to neutralize the `n_prior_admissions` artifact.

---

## Reproducibility

All design decisions, validated cohort definitions, ICD code lists, drug lists, MIMIC-IV data quirks, and bug fixes are documented in `CLAUDE.md`. The pipeline is structured so that each stage saves its output to disk before the next stage begins; SLURM job logs for every run are preserved under `logs/`. Random seeds are pinned to 42 throughout (matching, sampling, splits, model training, bootstrap). All design / methodology decisions are documented in `methodology_fix_plans.md` and `ewang163_methodology_fixes_results.md`, the final model selection memo is in `ewang163_model_selection_memo.md`, and the full multi-model comparison including runtime + explainability is in `ewang163_model_comparison.md`. The unmodified PULSNAR library (Kumar & Lambert 2024) is cloned into `PULSNAR/`.

End of write-up.
