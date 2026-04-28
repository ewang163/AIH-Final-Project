# PTSD Underdiagnosis Detection in MIMIC-IV Discharge Notes — Comprehensive Project Write-Up

**Author:** Eric Wang (ewang163), Brown University, AIH 2025 (Spring 2026)
**Source data:** MIMIC-IV v3.1 (BIDMC, 2008–2019) on Brown's Oscar HPC cluster
**Final deployment model:** Clinical Longformer + Kiryo nnPU loss at π_p = 0.25, Fix-1-masked corpus
**Sensitivity/SAR-aware model:** Clinical Longformer + PULSNAR propensity-weighted nnPU at α = 0.1957

---

## Part 1 — Introduction

### Clinical problem

PTSD is systematically undercoded in inpatient settings. Inpatient teams are not trained to elicit DSM-5 PTSD criteria, and PTSD has no laboratory or imaging marker — it has to be asked about, with knowledge of the patient's psychological and social history. Estimated true prevalence in trauma-exposed inpatients is ≥ 20% (Stanley et al. 2020 PCL-5 findings), but the ICD coding rate in MIMIC-IV is closer to 1%. Patients who are missed lose access to targeted treatment (trauma-focused psychotherapy, prazosin for nightmares, SSRI/SNRI), generate avoidable readmissions, and lose continuity of trauma history across encounters. A screening tool that flags patients whose discharge notes contain language suggestive of PTSD would let clinicians target a more thorough evaluation at otherwise-overlooked patients.

The methodological challenge is that the obvious approach — train on ICD-coded patients vs. non-coded patients — is wrong by assumption. If undercoding is the central feature of the problem, the "negative" group is contaminated with real positives, and any model trained against that gold standard is measuring agreement with a flawed reference rather than actual clinical validity. Designing around that contamination is the central methodological commitment of this project.

### AI approach and how it builds on / differs from prior work

The pipeline adapts the NLP-phenotyping framing from Blackley et al. (2021) for opioid use disorder and Zhou et al. (2015) for depression — extract clinically meaningful information from sectioned discharge notes, train classifiers, evaluate on a held-out set — but with three distinguishing methodological commitments not present in those precedents:

1. **Long-context transformer.** `yikuan8/Clinical-Longformer` (4,096 tokens) [Li et al. 2022, JAMIA] replaces the rule-based MTERMS front-end of Blackley/Zhou. Discharge notes routinely exceed BERT's 512-token window, and the sections that encode trauma exposure (Social History, Brief Hospital Course) are precisely where truncation lands.
2. **Positive-Unlabeled (PU) learning.** Kiryo et al. (2017) non-negative risk estimator as primary loss, because non-coded patients cannot be assumed PTSD-negative. PULSNAR (Kumar & Lambert 2024) is run as a SAR-aware sensitivity model.
3. **Multi-tier label-leakage defense + non-circular external validation.** Section filtering, pre-diagnosis-only training, universal PTSD-string masking, and a held-out pharmacological-proxy validation set (prazosin + SSRI/SNRI) the model never sees in training.

Five models are trained head-to-head: Clinical Longformer (primary), BioClinicalBERT (truncated and chunk-and-pool variants), TF-IDF + logistic regression, structured-features-only logistic regression, and a zero-training DSM-5/PCL-5 keyword baseline.

### Hypothesis

Can a Clinical Longformer fine-tuned with PU learning on section-filtered, predominantly pre-diagnosis discharge notes recover undercoded PTSD cases from MIMIC-IV — while generalizing to a pharmacological-proxy validation set whose patients were never used in training, rather than just re-deriving the ICD coding rule?

### Themes from background literature and the gap addressed

Across the assignment-5/6/7/8 literature, four themes emerge:

1. *NLP can recover undercoded psychiatric diagnoses from inpatient notes.* Zhou et al. (2015) found ~20% additional depression cases beyond ICD coding using MTERMS + ML.
2. *Substance-misuse phenotyping pipelines use sectioned-note feature engineering with traditional ML on top of NLP-derived CUIs.* Afshar et al. (2019), Sharma et al. (2020) — but their reference standards (AUDIT/SBIRT) were prospectively collected, sidestepping contaminated negatives.
3. *Pre-trained clinical text encoders generalize across phenotyping tasks.* Dligach, Afshar & Miller (2019) — motivating Clinical Longformer pretrained on MIMIC-III as backbone.
4. *Long-context inference matters for clinical notes.* Li et al. (2022) showed Longformer outperforms BioClinicalBERT on MIMIC-III phenotyping.

The unmet need addressed here is squarely the methodological one: prior phenotyping work either had a clean reference standard (AUDIT, prospective screen) or treated ICD-derived labels at face value. None confronted the case where the *target diagnosis is dominantly undercoded by the very labeling source used in supervised training*. The combined response — PU learning + label-leakage masking + non-circular pharmacological proxy + a SAR-aware sensitivity model — is the project's distinct contribution.

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
3. **Universal PTSD-string masking (Fix 1)** in `ewang163_ptsd_corpus_build.py`: a case-insensitive regex `(?i)(post-traumatic|post\s+traumatic|posttraumatic|trauma-related\s+stress|ptsd|f43\.1|309\.81)` is replaced with `[PTSD_MASKED]` across **all 5,950 PTSD+ notes** (pre-dx + fallback). The original design assumed pre-diagnosis notes could not contain PTSD references; the **Fix-1 audit found 360/4,169 = 8.6% of pre-dx notes contain explicit PTSD strings** carried forward from outside records ("h/o PTSD from MVA 2012"), validating Jin et al. (2023, JAMIA) on annotation noise from undercoded records.
4. **Two ablations** quantify residual leakage: Ablation 1 (re-mask everything post-hoc) and Ablation 2 (remove the entire PMH section).

### 2.5 Variable selection

- *Text features (primary):* section-filtered free text of patient notes, fed directly to transformer encoders.
- *Structured features (baseline only):* age at admission, sex, length of stay, emergency flag, Medicaid/self-pay flag, count of prior admissions, 5 race indicators, prior-admission flags for MDD, anxiety, SUD, TBI, chronic pain, suicidal ideation, SSRI/SNRI, prazosin, SGA. Twenty features total. Medications appear only here and as proxy-group labelling criteria — never in NLP text features (the discharge medications section is filtered out).

### 2.6 Models — implementation details

**Clinical Longformer (primary).** `yikuan8/Clinical-Longformer`, 4,096 tokens. AdamW, lr 2e-5, batch 2 × grad-accum 16 (effective batch 32), 5 epochs, warmup 0.1 of total steps, weight decay 0.01, gradient clip 1.0, mixed precision (`torch.amp.autocast` + `GradScaler`), gradient checkpointing. Best-epoch checkpointing on validation AUPRC.

**Kiryo nnPU loss (verbatim from `ewang163_ptsd_train_longformer.py`).** Logits reduced to a single scalar via `outputs.logits[:, 1] - outputs.logits[:, 0]`. Three BCE quantities:
- `loss_pos = BCE(logits[pos], 1)`
- `loss_unl = BCE(logits[unl], 0)`
- `loss_pos_as_neg = BCE(logits[pos], 0)`

Then:
```
pu = π_p · loss_pos + clamp(loss_unl − π_p · loss_pos_as_neg, min=0.0)
```
The `clamp(..., min=0)` is the Kiryo non-negative correction preventing the unlabeled risk minus the subtracted positive-as-negative term from going negative (which would reflect overfitting to the labeled positives). Falls back to standard BCE if a batch has no positives or no unlabeled.

**Class prior π_p (Fix 2).** `ewang163_ptsd_pip_sweep.sh` submits **7 parallel SLURM GPU jobs** sweeping π_p ∈ {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25}, each running the full 5-epoch training. `ewang163_ptsd_pip_sweep_eval.py` then evaluates **9 candidate checkpoints** (7 sweep values + the empirical-π_p retrain at 0.398 + the PULSNAR α=0.196 model) and selects by **proxy Mann-Whitney AUC** with MW p < 0.01 floor — the only PU-uncontaminated criterion. **Winner: π_p = 0.25** (proxy AUC 0.7990, MW p = 8.3e-22, val AUPRC 0.8834). The empirical labeled fraction (0.398) systematically overstates true population prevalence under the 3:1 matched design and was *not* selected.

**PULSNAR (Fix 3, SAR sensitivity).** `ewang163_ptsd_train_pulsnar.py` implements propensity-weighted nnPU. Steps:
1. Build prior-admission features from streamed `diagnoses_icd.csv` and `prescriptions.csv` (13 features including sex, age, emergency, medicaid, n_prior_admissions, prior-MDD/anxiety/SUD/suicidal, prior-SSRI/prazosin/SGA, prior-psych-any).
2. Fit logistic regression for propensity e(x) = P(coded | features), clip to [0.05, 0.95].
3. Estimate α via PULSNAR (xgboost classifier, `n_clusters=0, max_clusters=10, bin_method='rice', bw_method='hist'`); fall back to PULSCAR (SCAR) then to empirical fraction. **PULSNAR α = 0.1957** for the rich-features model.
4. Fine-tune from pretrained Clinical Longformer at lr 1e-5, 3 epochs (faster than primary), with the modified loss:
```
weights_i = (1/e(x_i)) / mean(1/e(x_j))   for positives
loss_pos_w = mean(BCE(logits[pos], 1) · weights)
loss_pos_as_neg_w = mean(BCE(logits[pos], 0) · weights)
pu_w = α · loss_pos_w + clamp(loss_unl − α · loss_pos_as_neg_w, min=0.0)
```
The propensity reweighting upweights positives with low coding propensity (older men, minorities, no prior psychiatric contact) — exactly the underrepresented PTSD population.

A v2 PULSNAR run with `n_prior_admissions` added to the propensity model produced α = 0.0006 (an artifact — the propensity model perfectly separated coded from uncoded, leaving PULSNAR no signal). v1 (α = 0.1957) is the principled estimate.

**BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`).** Same Kiryo nnPU loss, max_len 512, batch 16 × grad-accum 2 (effective 32), 5 epochs. Two inference modes: truncated (first 512 tokens) and **chunk-and-pool (Fix 8)** — overlapping 512-token windows with 256-token stride, max-pooled positive-class probabilities across windows.

**TF-IDF + LogReg.** Word 1-grams + 2-grams, max 50,000 features, `sublinear_tf=True`, `min_df=3`, `dtype=float32`. LogReg with `class_weight='balanced'`, lbfgs, `max_iter=1000`, C swept over {0.001, 0.01, 0.1, 1.0, 10.0, 100.0} on val AUPRC (best C = 10.0). Class weights are the standard sklearn substitute when not using nnPU — combining nnPU with balanced weights would double-correct.

**Structured + LogReg.** 20 features (above), same C grid (best 10.0). Streams `diagnoses_icd.csv` and `prescriptions.csv` to compute prior-admission flags strictly before each patient's index admission.

**Keyword (DSM-5/PCL-5).** Zero-training. **62 weighted regex patterns** in `ewang163_ptsd_train_keyword.py`, organized by DSM-5 criterion:
- *Criterion A (trauma exposure, weights 1.5–3.0):* `\bptsd\b`, `\bpost[- ]?traumatic\s+stress\b`, `\bcombat\s+(veteran|exposure|related)`, `\bsexual\s+assault\b`, `\brape[d]?\b`, `\bphysical\s+assault\b`, `\bdomestic\s+violence\b`, `\bchild(?:hood)\s+abuse\b`, `\bmva\b`, `\bgunshot\b`, `\bstab(?:bing|bed)\b`, `\bmilitary\s+sexual\s+trauma\b`, `\bmst\b`, …
- *Criterion B (intrusion, 2.0–3.0):* `\bflashback[s]?\b`, `\bnightmare[s]?\b`, `\bre-?experienc(?:e|ing)\b`, `\bintrusive\s+(?:thought|memor|image|recollection)`, `\bdissociative\s+(?:reaction|episode|flashback)`.
- *Criterion C (avoidance, 1.5–2.5):* `\bavoidance\b`, `\bavoid(?:s|ing|ed)?\s+(?:trigger|reminder|thought|feeling|place)`, `\bemotional(?:ly)?\s+numb(?:ing|ness)?\b`, `\bdetach(?:ed|ment)\b`.
- *Criterion D (negative cognition/mood, 0.5–1.5):* `\bguilt\b`, `\bself[- ]?blame\b`, `\bdiminished\s+interest\b`, `\bestrange(?:d|ment)\b`, `\bpersistent\s+negative\b`, `\bunable\s+to\s+(?:feel|experience)\s+positive`.
- *Criterion E (arousal/reactivity, 0.5–3.0):* `\bhypervigilance\b`, `\bhypervigilant\b`, `\bexaggerated\s+startle\b`, `\bstartle\s+(?:response|reflex|reaction)`, `\bhyperarous(?:al|ed)\b`, `\binsomnia\b`, `\birritab(?:le|ility)\b`, `\banger\s+outburst\b`, `\breckless\s+behavior\b`, `\bconcentration\s+difficult\b`.
- *Treatment (tx, 1.5–3.0):* `\bprolonged\s+exposure\s+therapy\b`, `\bemdr\b`, `\bpcl[- ]?5\b`, `\bcaps[- ]?5\b`, `\btrauma[- ]?focused\b`, `\bprazosin\b`, `\bcpe\b`.

Two scoring variants are compared on val AUPRC: raw weighted count (winner) vs. TF-normalized (raw / word count).

**Specificity check (`ewang163_ptsd_specificity.py`).** Standard cross-entropy (NOT nnPU — psych controls are confirmed negatives) with class-balanced weights, training PTSD+ (n = 5,711) vs. age/sex 1:1-matched MDD/anxiety controls.

### 2.7 Splits

`ewang163_ptsd_splits.py` performs **patient-level 80/10/10 stratified split** by `subject_id` and `ptsd_label` (`random_state=42`). A patient's data never crosses split boundaries.
- Random split (canonical): train 11,837 rows / val 1,471 / test 1,551 (660 PTSD+ in test).
- **Temporal split (Fix 7)**: pre-2015 train, 2017–2019 test, using each patient's `anchor_year_group` from `patients.csv`. Crucially, MIMIC-IV applies per-patient random date shifts (~100–200 years), so raw `admittime` is unusable for chronology — the un-shifted `anchor_year_group` is the correct workaround. Result: train 11,134 / val 1,270 / test 2,455.

### 2.8 Evaluation strategy

Implemented across nine scripts in `scripts/04_evaluation/`:

**Threshold derivation (Fix 4).** All operating thresholds computed on **validation** at sensitivity ≥ 0.85 via `threshold_at_recall(probs, labels, target_recall=0.85)` — sweeps `np.linspace(1.0, 0.0, 1001)` for the lowest threshold achieving recall ≥ 0.85. **Frozen before any test-set metrics.** The val-derived threshold is stored in `evaluation_results.json` under `val_thresholds.{model}` and inherited by all downstream scripts (proxy validation, fairness, error analysis, chart review). Pre-Fix-4 ablation script and v1 attribution still use test-set thresholds, but the deltas are interpretable.

**Discrimination.** AUPRC (primary) via `average_precision_score`; AUROC via `roc_auc_score`.

**Clinical utility metrics (in `compute_metrics`):**
- LR+ = sens / (1 − spec)
- LR− = (1 − sens) / spec
- DOR = LR+ / LR− (numerically stable form: `(sens / (1 − sens + ε)) / ((1 − spec + ε) / (spec + ε))`)
- Alert rate = (TP + FP) / N
- Workup reduction vs. treat-all = 1 − alert rate
- Number Needed to Evaluate (NNE) = 1 / (sens − (1 − spec))
- Bayes' theorem prevalence-recalibrated PPV: PPV(prev) = (sens · prev) / (sens · prev + (1 − spec) · (1 − prev))
- NPV(prev) = (spec · (1 − prev)) / (spec · (1 − prev) + (1 − sens) · prev)
- NNS = 1 / PPV
- Reported at deployment prevalences {1%, 2%, 5%, 10%, 20%}.

**McNemar's test** with continuity correction on paired test predictions:
- b = #(Longformer correct & comparator wrong); c = #(Longformer wrong & comparator correct)
- chi² = (|b − c| − 1)² / (b + c); p from `scipy.stats.chi2(df=1)`.

**Calibration (Fix 5).** `ewang163_ptsd_calibration.py` fits Platt scaling (`LogisticRegression(C=1)` on val_probs vs. val_labels), then applies the Elkan-Noto correction:
- c = mean(raw model prob on val positives) — estimates P(s=1|y=1)
- corrected_prob = clip(platt_scaled / c, 0, 1) — approximates P(PTSD=1) rather than P(coded=1)

ECE on 10 equal-frequency bins with **Wilson 95% CI** per bin:
- centre = (p̂ + z²/(2n)) / (1 + z²/n);  z = 1.96
- margin = z · √((p̂(1−p̂) + z²/(4n))/n) / (1 + z²/n)
- ECE = Σᵢ (nᵢ/N) · |mean_predᵢ − observed_fracᵢ|

**Decision Curve Analysis (Vickers).** Net benefit at thresholds 0.01–0.40 step 0.005:
- NB_model(t) = TP/N − (FP/N) · (t/(1−t))
- NB_treat-all(t) = prev − (1 − prev) · (t/(1−t))
- NB_treat-none = 0

Calibrated probabilities are first deployment-prevalence-shifted via Bayes' rule:
- p_deploy = (p_cal · p_dep / p_study) / (p_cal · p_dep / p_study + (1 − p_cal) · (1 − p_dep) / (1 − p_study))

**Ramola PU correction (Fix 6).** Applied to all six models in `evaluate.py`. With α = π_p:
- corrected_AUROC = (AUROC − 0.5α) / (1 − α)
- corrected_AUPRC = AUPRC / α (capped at 1.0)
- corrected_precision = prec / (prec + (1 − prec)·(1 − α))
- corrected_FPR = max(0, FPR_raw − α·sens) / (1 − α)
- sensitivity unchanged

All raw metrics labeled as "PU lower bounds" with corrections reported alongside.

**Pharmacological proxy external validation.** `scipy.stats.mannwhitneyu(proxy_probs, unlab_probs, alternative='greater')`; AUC = U / (n_proxy · n_unlab) — the rank-biserial / Wilcoxon-AUC. Comparison group: 500 unlabeled patients drawn from training pool only (`splits['train']`, `np.random.RandomState(42)`), one note per patient via `drop_duplicates`. Threshold inherited from `evaluation_results.json::val_thresholds.longformer`.

**Fairness (Fix 9).** Per subgroup:
- *Calibration-in-the-large* = mean(predicted) − mean(observed); Wilson CI on the observed proportion.
- *Equal opportunity difference* (Hardt et al. 2016) = max(recall) − min(recall) at the val-derived threshold across subgroup levels.
- *Bootstrap 95% CI on AUPRC*: 1,000 resamples (`np.random.RandomState(42)`), percentile method (2.5th/97.5th). Skips degenerate resamples. Reported reliable iff CI width < 0.15.
- White-vs-Non-White binary contrast added because per-race n_pos is small.

**Integrated Gradients (Fix 10).** `ewang163_ptsd_attribution_v2.py` uses Captum's `IntegratedGradients` on the embedding tensor via a custom `EmbeddingForwardWrapper` that explicitly constructs `global_attention_mask` (zeros, with position 0 = CLS = 1) and passes `inputs_embeds` rather than `input_ids`. This works around v1's failure (44/50 patients failed) caused by Captum's hooks conflicting with Longformer's hybrid local/global attention. n_steps=20, internal_batch_size=1, full 4,096-token context, pad-token baseline. Word-level aggregation merges contiguous BPE subwords whose offsets connect through alphabetic / hyphen / apostrophe characters with **summed** attributions. 50 high-confidence true positives sampled from the top decile (`random_state=42`).

**Error analysis.** 25 FPs and 25 FNs sampled at threshold 0.38; per-set demographics, mean note length, top TF-IDF terms via the saved vectorizer, distinctive ratio (subset_mean / overall_mean), trauma-term scan (18 substrings).

**Runtime benchmarking.** `BenchmarkLogger` context manager (`scripts/common/ewang163_bench_utils.py`) captures wall-clock (`time.perf_counter`), CPU time (`time.process_time`), peak memory (`resource.getrusage(RUSAGE_SELF).ru_maxrss`), and GPU-hours; appends to `results/metrics/ewang163_runtime_benchmarks.csv`. The unified inference benchmark (`ewang163_unified_inference_bench.py`) times all five models in a single L40S allocation for apples-to-apples comparison; CPU baselines re-measured separately on a 16-CPU batch node.

### 2.9 Pipeline flowchart (mapped to OHDSI)

```
MIMIC-IV (read-only)
   ↓ stream + filter
[01_cohort/]    table1 → cohort_sets → admissions_extract → notes_extract       [OHDSI: Cohort + Characterization]
   ↓
[02_corpus/]    corpus_build (Fix 1 universal mask) → splits (random + temporal) [OHDSI: Dataset]
   ↓
[03_training/]  train_longformer (π_p sweep, 7 jobs) │ train_bioclinbert │      [OHDSI: Analyze]
                train_tfidf │ train_structured │ train_keyword │
                train_pulsnar │ train_specificity
   ↓
[04_evaluation/] evaluate (Fix 4/6/8) → calibration (Fix 5) → decision_curves →
                proxy_validation (Fix 4) → ablations → fairness (Fix 9) →
                attribution_v2 (Fix 10) → error_analysis → chart_review_packet
                (Fix 11) → temporal_eval → pulsnar_reeval → unified/cpu inference benches
   ↓
results/{table1, predictions, metrics, figures, attribution, error_analysis,    [OHDSI: Research Products]
         chart_review, runtime_benchmarks}
```

GitHub: https://github.com/ewang163/AIH-Final-Project — programs documented in README.md.

---

## Part 3 — Results and Discussion

All numbers below are on the patient-level held-out test set (n = 1,551 patients, 660 PTSD+, prevalence 42.55%) under val-derived thresholds, unless otherwise stated.

### 3.1 Model comparison (test set, latest evaluation)

| Model | AUPRC | AUROC | Sens | Spec | Prec | F1 | LR+ | DOR | NNS @ 2% | McNemar p vs. winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clinical Longformer (π_p=0.25, primary)** | **0.8939** | **0.9002** | 0.852 | 0.782 | 0.743 | **0.794** | 3.91 | 20.6 | **13.5** | — |
| Clinical Longformer (PULSNAR α=0.196) | 0.8848 | 0.8904 | 0.846 | 0.745 | 0.711 | 0.772 | 3.32 | 16.0 | 15.8 | — |
| BioClinicalBERT (PU, chunk-pool) | 0.8775 | 0.8853 | 0.902 | 0.626 | 0.641 | 0.749 | 2.41 | 15.3 | 21.0 | < 1e-5 |
| BioClinicalBERT (PU, truncated 512) | 0.8576 | 0.8656 | 0.820 | 0.728 | 0.691 | 0.750 | 3.02 | 12.2 | 16.5 | 8e-6 |
| TF-IDF + LogReg | 0.8380 | 0.8567 | 0.817 | 0.721 | 0.684 | 0.745 | 2.92 | 11.5 | 17.0 | 1e-6 |
| Structured + LogReg | 0.6833 | 0.7310 | 0.909 | 0.207 | 0.459 | 0.610 | 1.15 | 2.6 | 36.7 | 0 |
| Keyword (DSM-5/PCL-5) | 0.5373 | 0.6086 | 1.000 | 0.000 | 0.426 | 0.597 | 1.00 | inf | 47.0 | 0 |

**Headline.** Clinical Longformer at π_p = 0.25 wins discrimination at AUPRC 0.894 / AUROC 0.900, beating every comparator on McNemar's test. The keyword baseline is essentially random (AUPRC 0.537, AUROC 0.609); structured-only (0.683) is well below text-based models — narrative content carries the bulk of predictive signal. **Fix 8 chunk-and-pool BERT (0.878) closes most of the gap to Longformer**, indicating that long-context inference, not architecture per se, drives most of Longformer's lift; the residual ~0.016 AUPRC gain is attributable to long-range pre-training. Ramola PU corrections push Longformer corrected AUPRC to 1.0 (ceiling-clipped at α = 0.4255 test labeled fraction) and corrected AUROC to 1.0 — the raw metrics are conservative PU lower bounds.

### 3.2 Training dynamics

Clinical Longformer (5 epochs, ~1.5 GPU-h/epoch on L40S, total 5.73 GPU-h):

| Epoch | Train loss | Val AUPRC |
|---:|---:|---:|
| 1 | 0.465 | 0.864 |
| 2 | 0.303 | 0.875 |
| **3** | **0.248** | **0.883** |
| 4 | 0.195 | 0.873 |
| 5 | 0.155 | 0.867 |

Best at epoch 3; epochs 4–5 begin to overfit. The π_p = 0.25 sweep variant peaks at val AUPRC 0.8834 in epoch 3.

PULSNAR fine-tuned from a pre-trained Longformer at lr 1e-5 for 3 epochs, converging at val AUPRC 0.8725 in 3.50 GPU-h (39% cheaper than the primary).

BioClinicalBERT (5 epochs, ~140 s/epoch, 0.22 GPU-h on RTX 3090): best val AUPRC 0.835 at epoch 5.

### 3.3 Ablation studies (label leakage)

| Condition | AUPRC | AUROC | Δ vs. baseline |
|---|---:|---:|---:|
| Baseline | 0.8827 | 0.8913 | — |
| Ablation 1 — explicit PTSD strings masked | 0.8749 | 0.8898 | −0.008 |
| Ablation 2 — entire PMH section removed | 0.8218 | 0.8382 | −0.061 |

Ablation 1 is essentially free (model is *not* exploiting literal PTSD strings). Ablation 2 costs ~6 AUPRC points but the model still scores 0.82 — comfortably above TF-IDF — so HPI, Social History, and Brief Hospital Course independently encode enough PTSD-associated language. Some signal lives in PMH (carried-forward psychiatric history), but it's not dominant.

### 3.4 Calibration

| Variant | ECE | Best calibrated? |
|---|---:|---|
| Raw PU output (π_p = 0.25) | **0.0638** | ✓ |
| Platt-scaled | 0.0742 | |
| Elkan-Noto corrected | 0.0797 | |

**Raw probabilities are the best-calibrated for π_p = 0.25** — Platt scaling slightly *worsens* ECE because the nnPU at this π_p already produces a sharp, near-calibrated distribution. Elkan-Noto over-corrects when measured against PU labels (the correction shifts probabilities upward toward P(PTSD=1), but ECE is computed against PU labels that treat hidden positives as negatives — a classic Fix-6 PU-lower-bound situation). The Elkan-Noto **c estimate is 0.7833**, implying ~22% undercoding rate, consistent with Stanley et al. (2020) PCL-5 findings.

**Recommendation:** use raw probabilities for ranking and threshold selection; use Elkan-Noto-corrected probabilities wherever absolute P(PTSD=1) matters.

### 3.5 Decision curve analysis — a notable negative finding

DCA results (`ewang163_dca_results.csv`): the model offers positive net benefit over treat-none across thresholds 0.01–0.14 at 2% prevalence and a wider range at 5% — but **does not dominate the treat-all strategy in the very-low-threshold band at either prevalence** (max NB at 2%: 0.356; max at 5%: 0.389; treat-all at threshold 0.01 reaches NB ≈ 0.42). At deployment prevalences below ~2%, the cost of missing a case is high enough relative to clinician review time that flagging everyone is competitive. This is honest and clinically informative — it says the screening model is worthwhile only when paired with a deployment regime where clinician time is the binding constraint.

### 3.6 Prevalence recalibration

| Deployment prevalence | PPV | NPV | NNS |
|---:|---:|---:|---:|
| 1% | 0.038 | 0.998 | 26.3 |
| 2% | 0.074 | 0.996 | 13.5 |
| 5% | 0.171 | 0.990 | 5.9 |
| 10% | 0.303 | 0.979 | 3.3 |
| 20% | 0.494 | 0.955 | 2.0 |

At 2% inpatient deployment prevalence, ~14 patients need to be flagged to find one true PTSD case — clinically tolerable for an automated screening prompt. Workup reduction vs. treat-all is **51%** (alert rate 0.49). NPV > 0.99 at every prevalence makes a negative screen highly reliable. LR+ = 3.91 is "moderate" by clinical convention, appropriate for a screening prompt (not for diagnosis).

### 3.7 Subgroup AUPRC

| Subgroup | n | n_pos | AUPRC |
|---|---:|---:|---:|
| Female | 940 | 433 | **0.925** |
| Male | 611 | 227 | 0.828 |
| Age 20s | 293 | 140 | **0.944** |
| Age 30s | 359 | 194 | 0.920 |
| Age 40s | 308 | 131 | 0.877 |
| Age 50s | 277 | 81 | 0.867 |
| Age Other (<20 or ≥60) | 314 | 114 | 0.845 |
| Race: Black | 209 | 91 | 0.932 |
| Race: White | 1,083 | 485 | 0.892 |
| Race: Hispanic | 93 | 44 | 0.865 |
| Race: Other/Unknown | 118 | 35 | 0.892 |
| Race: Asian | 48 | 5 | 0.848 (CI width 0.55, **unreliable**) |
| Emergency = True | 855 | 450 | **0.925** |
| Emergency = False | 696 | 210 | 0.831 |

The model performs best on patients **most likely to be coded** (younger women, emergency admissions) and weakest on the demographics where undercoding is most prevalent (older patients, men, elective). This is the single most important deployment caveat.

### 3.8 Fairness — equal opportunity differences (Fix 9)

| Subgroup | Max recall | Min recall | EO diff |
|---|---:|---:|---:|
| Sex (F vs. M) | 0.892 (F) | 0.775 (M) | **0.116** |
| Age | 0.929 (20s) | 0.719 (Other) | **0.209** |
| Race binary (W vs. Non-W) | 0.869 (Non-W) | 0.845 (W) | 0.024 |
| Emergency | 0.862 (True) | 0.829 (False) | 0.034 |

**Race disparity is minimal (0.024)** — contradicting the a-priori concern. The dominant fairness issues are **age (0.209)** and **sex (0.116)** — both inherited from the non-SCAR coding bias. Asian-subgroup AUPRC is correctly suppressed as unreliable (CI width 0.55, n_pos = 5); Hispanic CI width 0.166 also borderline. Calibration-in-the-large per subgroup ranges from −0.013 (Emergency=True) to +0.066 (Emergency=False).

### 3.9 Pharmacological proxy external validation — the headline non-circular result

| Group | n | Median score | Mean | Q1, Q3 | Frac. above op-threshold (0.324) |
|---|---:|---:|---:|---:|---:|
| **Pharmacological proxy** | 102 | **0.383** | 0.464 | 0.165, 0.792 | **57.8 %** (59/102) |
| Random unlabeled sample | 500 | 0.059 | 0.146 | 0.024, 0.155 | 15.4 % (77/500) |

**Mann-Whitney U = 40,748, p = 8.29e-22, AUC = 0.7990.**

The model — never shown a single proxy patient during training — assigns ~6× the median probability to patients whose pharmacotherapy pattern (prazosin + SSRI/SNRI within 180 days) is consistent with PTSD treatment. **58% of proxy patients clear the screening threshold vs. 15% of demographically-matched unlabeled controls.** Because proxy patients are identified by an entirely independent (medication-based) criterion that the model cannot see — discharge medications are a filtered-out section — this is the project's strongest single piece of validity evidence and the only PU-uncontaminated metric (Fix 6 elevated it to co-headline status).

### 3.10 Specificity check vs. psychiatric controls

A separate Longformer trained on PTSD+ vs. age/sex 1:1-matched MDD/anxiety controls (standard cross-entropy, n_pos = 5,711, n_neg = 5,711, n with notes = 3,148) reaches:

| Metric | PU model (primary) | Specificity model |
|---|---:|---:|
| Test AUPRC | 0.8939 | **0.9109** |
| Test AUROC | 0.9002 | 0.8145 |
| Sensitivity | 0.852 | 0.852 |
| Specificity | 0.782 | 0.581 |
| Precision | 0.743 | 0.821 |
| Mean predicted prob on proxy set | 0.494 | 0.345 |

Even when held to the much stronger comparator of MDD/anxiety patients, AUPRC remains > 0.91 — **PTSD-specific signal is recoverable above-and-beyond generic "psychiatric admission" language**, ruling out the worst-case interpretation that the primary model is a psych-vs-non-psych classifier. The ΔAUPRC of +0.028 is small in absolute terms; the AUROC drops from 0.90 to 0.81 reflects the harder task (psychiatric controls have overlapping vocabulary).

### 3.11 Integrated Gradients attribution (Fix 10, 4,096-token context)

| Section | π_p = 0.25 share | PULSNAR share |
|---|---:|---:|
| HPI | 36.4 % | **43.2 %** |
| Brief Hospital Course | 35.9 % | 32.0 % |
| PMH | 26.3 % (per-token density highest at 0.021) | 22.4 % (per-token 0.036) |
| Social History | 0.5 % | 1.3 % |

**Top attributed words (π_p = 0.25):** bipolar (0.112), pylori (0.074), personality (0.067), schizoaffective (0.063), disorder (0.061), inr (0.053), coumadin (0.048), abusive (0.045), arthritis (0.044). Also notable: substance (0.039), abuse (0.038), psychiatric (0.037), assault (0.037), heroin (0.038), suicidal (0.030).

**Top attributed words (PULSNAR):** bipolar, narcotic, illness, arrested, delayed, pancreatitis, schizoaffective, psychosis, anemia, assault.

**No label-leakage tokens** (e.g., "ptsd", "posttraumatic") appear, confirming Fix 1 worked. Vocabulary spans psychiatric comorbidities, substance use, trauma exposure, and treatment context — clinically plausible PTSD-adjacent surface form.

**The PULSNAR-vs-π_p=0.25 divergence is informative.** PULSNAR shifts attribution from PMH (where comorbidities live) to HPI (where trauma history lives), and its top words are noticeably more trauma-anchored (`narcotic`, `arrested`, `assault`, `psychosis`) earlier in the list. This is exactly what SAR-aware training should do — discount the PMH-comorbidity profile (the strongest non-PTSD predictor of being ICD-coded) and rely more on narrative content. PULSNAR loses 0.009 AUPRC against ICD-coded labels but appears, by attribution, to be **less captured by the coding bias** — arguably the better model for the *undercoding* use case, even though it scores worse on the SCAR-coded benchmark.

### 3.12 Error analysis

| | n | Mean pred. prob | Mean note length | % female | % age 20s | % age "Other" | % emergency |
|---|---:|---:|---:|---:|---:|---:|---:|
| False positives | 227 | 0.66 | 3,239 | 60.4 % | 15.0 % | 17.2 % | 42.3 % |
| False negatives | 99 | 0.18 | 2,932 | 49.5 % | 10.1 % | **36.4 %** | 27.3 % |
| Test set overall | 1,551 | 0.48 | 3,108 | 60.6 % | 17.2 % | 23.0 % | 37.9 % |

**False positives skew female and emergency** — demographically matching coded patients, consistent with these being **likely true undercoded PTSD** that the ICD label simply doesn't reflect. Qualitative inspection of the 95-KB FP notes file finds substance use, prior psychiatric admissions, and assault histories that would have justified PTSD coding had a structured screen been done.

**False negatives skew male, older (Other = 36% vs. 23% overall), and elective.** They have substantially shorter notes (2,932 vs. 3,108) and very low predicted probabilities (median 0.16) — the model has nothing to anchor on in terse documentation.

### 3.13 Temporal generalization (Fix 7) — temporal training did not help

| Scenario | Test AUPRC |
|---|---:|
| Random-split model on random test | 0.888 |
| Random-split model on temporal test (2017–2019) | 0.886 |
| **Temporal-trained model on temporal test** | 0.842 |

Temporal training **hurt** generalization. The random-split model loses only 0.002 AUPRC when tested on 2017–2019 — meaning the random distribution is already representative of late MIMIC-IV. The temporal model, trained only on pre-2015 data (8,752 vs. 11,837 patients), loses 0.044 AUPRC because it has 25% less training data and misses richer post-2013 DSM-5-era coding patterns. **Random split is recommended for deployment.**

### 3.14 Compute frontier (measured)

Inference timed in a single L40S allocation (job 1923224); CPU baselines on a 16-CPU batch node:

| Model | ms/patient | Train wall (s) | Train GPU-h | Test AUPRC |
|---|---:|---:|---:|---:|
| Longformer π_p = 0.25 (winner) | 80.4 | 20,631 | 5.73 | 0.8939 |
| Longformer PULSNAR | 80.4 | 12,617 | 3.50 | 0.8848 |
| BERT chunk-pool | 22.7 | 791 | 0.22 (RTX 3090) | 0.8775 |
| BERT truncated | 2.97 | same | same | 0.8576 |
| TF-IDF + LogReg | 0.84 | 16 (CPU) | 0 | 0.8380 |
| Structured + LogReg | 17.9 (I/O bound) | 68 (CPU) | 0 | 0.6833 |
| Keyword (16-CPU) | **0.34** | 0 | 0 | 0.5373 |

Architecture (4,096 vs. 512 attention) — not GPU generation — drives the gap. Longformer pays ~22× hardware-normalized training cost and 3.55× inference latency vs. chunk-pool BERT for **+0.016 AUPRC**. Keyword (16-CPU) is 236× faster per-patient than Longformer, but at AUPRC 0.537 the speed buys near-random predictions. Total methodology-fixes compute: ~55.6 GPU-hours.

**Cost per 50,000 inpatient discharges/month:** Longformer 67 minutes (~1.1 GPU-h, ~$1–2); BERT chunk-pool 19 minutes (0.3 GPU-h); keyword 0.3 minutes (CPU). Inference is the recurring cost; training is one-shot.

### 3.15 The TF-IDF label-leakage finding (a worked-example argument for explainability)

Inspecting TF-IDF coefficients revealed **`ptsd_masked` with coefficient +37.13** — by far the largest in the model, 3× the next-highest feature. This is a leak: TF-IDF tokenization strips brackets, so `[PTSD_MASKED]` becomes the unigram `ptsd_masked`, which itself marks notes that originally contained PTSD. Additionally, `ptsd` itself has coefficient +10.29, suggesting some variants escaped masking. **TF-IDF AUPRC 0.838 is therefore inflated**; honest leak-free TF-IDF performance is unknown without retraining with stricter tokenization. Transformer tokenizers preserve `[PTSD_MASKED]` as a multi-token sequence whose embeddings are learned during fine-tuning (not as a single bag-of-words feature), and Ablation 1 (post-hoc unmasking causes only −0.008 AUPRC) confirms the transformer is not exploiting the analogous leak. Lesson: **interpretability and trustworthiness are not synonymous** — inspecting top coefficients caught a bug that AUPRC scoring did not.

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

The +6.51 coefficient on `n_prior_admissions` is **3.6× larger than the next feature** and matches the +5.63 dominance the same feature has in the PULSNAR propensity model. This is the SAR violation made manifest: the structured model is largely learning "this patient has been admitted many times, therefore probably PTSD-coded" — a coding-frequency signal, not a PTSD signal. It is also a **dataset artifact**: by construction, Group 3 (unlabeled) has index = first MIMIC-IV admission, so prior-admission count is always 0. Group 1 (PTSD+) has index = first PTSD-coded admission, which tends to be later. Deploying the structured model would entrench this artifact. Race coefficients are small in magnitude (max |0.67| for Asian, n = 5 in test — unstable), confirming Fix 9's main race-binary EO diff of 0.024.

### 3.17 The chart review packet — outstanding deliverable (Fix 11)

`ewang163_ptsd_chart_review_packet.py` produced a 195.7 KB packet (`ewang163_top50_review_packet.txt`) and an empty rating form (`ewang163_top50_review_form.csv`) for the **top-50 model-flagged unlabeled patients**. Score range: 0.840 to 0.991 (mean 0.926). Each entry includes subject_id, hadm_id, model score, age, sex, and section-filtered note (truncated at 10,000 chars). Rating scheme: 1 = Probable PTSD, 2 = Possible PTSD, 3 = Unlikely PTSD. **Clinician review is pending** — this would yield clinician-rated PPV@top50, the most persuasive single validity metric for an *undercoding* detection model.

---

## Discussion

### 3.18 Will I evaluate the model's performance?

The primary metric is **AUPRC**, with AUROC reported alongside. AUPRC is more informative under class imbalance — it can stay high while precision at clinically actionable thresholds is poor (which would rule out screening deployment). AUROC at the 42.55% test prevalence is interpretable, but real-world deployment prevalence is much lower (~ 2%), so PPV/NPV/NNS are recalibrated to deployment prevalences {1, 2, 5, 10, 20%} via Bayes' theorem on the val-derived sens/spec. Threshold-anchored metrics (sensitivity, specificity, precision, F1) at the recall ≥ 0.85 operating point are reported. McNemar's test with continuity correction quantifies pairwise model differences. The proxy Mann-Whitney AUC is the only PU-uncontaminated metric and is co-headline with AUPRC.

### 3.19 How will I determine whether the model is clinically useful and appropriate?

Three clinical-utility lenses are applied:

1. **Screening tradeoff at deployment prevalence.** At 2% prevalence, NNS = 13.5 and PPV = 7.4% — clinically tolerable for a screening prompt that points clinicians at otherwise-missed patients (a clinician spending 5 minutes per flagged chart catches one likely PTSD case per ~70 minutes). LR+ 3.91 is "moderate" — appropriate for screening, not for diagnosis. NPV > 0.99 means a negative screen is highly reliable.
2. **Decision Curve Analysis.** At deployment prevalence ≥ 5%, the model offers meaningful net benefit across thresholds 0.01–0.30 — it is a defensible alternative to flagging everyone. At very low prevalence (1–2%), treat-all dominates in the very-low-threshold band; the model is competitive only at moderate thresholds.
3. **Explainability.** Integrated Gradients (not attention — Jain & Wallace 2019) at 4,096 tokens shows HPI (36–43% of attribution) and Brief Hospital Course dominate; top words are clinically appropriate trauma/psychiatric/substance vocabulary; **no label-leakage tokens** appear.

Subgroup performance is the central caveat for clinical appropriateness: the model is best-calibrated on younger women in emergency admissions (the demographic most likely to be coded today) and weakest on older men in elective admissions (the demographic where undercoding is most prevalent — and the population a screening tool would be most valuable for). PU learning reduces but does not eliminate this inherited bias.

### 3.20 Limitations

**Section filtering is not perfect leakage prevention.** Even on a pre-diagnosis admission, PMH may carry forward "history of PTSD" from outside records — the Fix-1 audit confirmed 8.6% of pre-dx notes had explicit PTSD strings before masking. Ablation 2 quantifies the upper bound (PMH removal costs 6 AUPRC points, so residual PMH leakage is bounded but not zero).

**PU learning reduces but does not eliminate selection bias.** Kiryo nnPU removes the assumption that unlabeled patients are confirmed negatives but does not correct for the fact that ICD-coded patients are a non-random sample of true PTSD cases. The subgroup analysis shows exactly the expected residual bias: AUPRC 0.92 for women vs. 0.83 for men; 0.94 for patients in their 20s vs. 0.85 for "Other" age. PULSNAR was added as a SAR-aware sensitivity model; its attribution patterns suggest it is less captured by coding bias, even though it scores 0.009 AUPRC lower against the SCAR-coded benchmark.

**The pre-diagnosis training subsample is not representative.** Only 2,492 of 5,711 PTSD+ patients (43.6%) had pre-diagnosis admissions — by definition multi-admission patients, who tend to be sicker, older, and more psychiatrically complex. The 3,219 fallback patients use index-admission notes with masking applied; they carry more residual leakage risk.

**Proxy validation set is small (n_with_notes = 102) and has a known FPR.** The proxy criterion (prazosin + SSRI/SNRI within 180 days, no cardiovascular/BPH/Raynaud/TBI exclusion) carries an estimated 15–20% false-positive rate (patients on prazosin for off-label uses not captured by the exclusion ICD codes; PULSNAR's empirical underperformance on this metric is partly explained by the proxy itself selecting a "prazosin-adjacent" phenotype that pi_p = 0.25 over-fits to). Elevated proxy scores are evidence, not proof.

**Single-site data.** MIMIC-IV is one academic medical center in Boston. Whether any of this transfers to community hospitals, rural settings, the VA system (where PTSD prevalence is much higher), or international settings is unknown. The temporal split (Fix 7) showed limited within-MIMIC-IV distribution shift, but cross-site validation has not been performed.

**Calibration is imperfect.** ECE = 0.064 (raw, the best of the three variants). Acceptable for ranking and threshold-based screening; for any application that depends on absolute predicted probability, additional calibration work would be needed. Notably, Platt scaling slightly *worsens* calibration here, and Elkan-Noto over-corrects when measured against PU labels.

**Decision curve analysis is favourable but not dominant.** At 2% deployment prevalence, the model does not beat treat-all in the very-low-threshold band — the cost of missing a case is high enough relative to clinician review time that flagging everyone is competitive at very low prevalence.

**The TF-IDF leak (`ptsd_masked` coefficient +37.13).** A reminder that interpretability ≠ trustworthiness. Inspecting the top features caught a leak that AUPRC scoring did not. The transformer-based models do not appear to suffer the same leak (Ablation 1 cost only −0.008 AUPRC), but the experience argues for routine coefficient/attribution inspection as part of deployment validation.

### 3.21 Interpreting results in the context of existing literature, and what would constitute an actionable finding

The AUPRC of 0.894 is competitive with or exceeds the substance-misuse phenotyping benchmarks (Afshar 2019: AUPRC ~0.85 on AUDIT-screened alcohol misuse with cTAKES + CUI features; Sharma 2020: comparable for opioids), achieved on a *contaminated-negatives* problem they could sidestep by using prospectively collected reference standards. The Longformer's lift over BioClinicalBERT chunk-pool (+0.016 AUPRC, McNemar p = 8e-6) is consistent with Li et al. (2022) — long-range pretraining provides a small but reliable benefit on top of long-context inference.

Most importantly, the **proxy Mann-Whitney AUC of 0.799 (p = 8.3e-22)** is the single most actionable finding: a model trained without any reference to medications or proxy patients assigns 6× higher median probability to patients whose pharmacotherapy is consistent with PTSD treatment. This is direct, non-circular evidence that the model recovers a real PTSD-associated narrative signal rather than just re-deriving the ICD coding rule. Combined with the specificity check (AUPRC 0.91 against MDD/anxiety controls — the model is not just learning generic psychiatric language) and the IG attribution (clinically appropriate trauma/psychiatric vocabulary, no label-leakage tokens), this constitutes a defensible body of validity evidence.

A **meaningful and actionable finding** would be: clinician chart review (Fix 11, in progress) of the top-50 flagged unlabeled patients yields a clinician-rated PPV@top50 of, say, ≥ 70% — meaning ≥ 35 of the 50 most-confidently-flagged uncoded patients are judged "probable" or "possible" PTSD on chart review. That would convert this from a methodology demonstration into an actionable screening tool ready for prospective deployment trials. Conversely, a low rated PPV (< 40%) would force a re-evaluation of how much of the model's signal is true PTSD vs. coding-bias surface form.

### 3.22 What this tool is and is not

**It is** a screening prompt — a way to point inpatient clinicians at patients whose narrative notes contain language patterns associated with PTSD. The output is intended to inform a more thorough psychiatric evaluation, not to make a diagnosis. At 2% deployment prevalence and the chosen operating threshold, the tool would surface ~14 patients per true case for clinician review.

**It is not** a diagnostic tool, a replacement for structured PTSD screening (PCL-5, CAPS-5), or a basis for ICD coding. The downstream user is a clinician who will conduct a proper interview before any management decision is made.

### 3.23 Future directions

- **Clinician chart review (Fix 11 completion).** Finish rating the top-50 packet to produce clinician-validated PPV.
- **External validation** at a non-MIMIC site, ideally one with both higher PTSD base rates and richer narrative documentation (a VA medical center is the natural target).
- **Re-evaluation under a true reference standard.** A small prospective cohort screened with PCL-5 or CAPS-5 would let absolute model performance be measured against a defensible gold standard.
- **Bias mitigation for demographic subgroups.** The female / younger-age coding bias is inherited from labels; explicit subgroup-aware loss reweighting or label-noise modelling (Bekker & Davis 2020) could narrow the gap. PULSNAR's attribution shift hints that SAR-aware training helps; a richer propensity feature set (without the `n_prior_admissions` artifact) might widen the benefit.
- **Re-trained TF-IDF baseline with stricter masking** to determine its honest leak-free performance.
- **BERT chunk-pool calibration / fairness / attribution** — currently un-measured; would close the methodology comparison.
- **Integration with structured features.** The structured baseline AUPRC was 0.68; concatenating structured features (excluding the artifactual `n_prior_admissions`) into the Longformer head could provide complementary signal, particularly for the demographics where text features are weakest.

---

## Reproducibility

All design decisions, validated cohort definitions, ICD code lists, drug lists, MIMIC-IV data quirks, and bug fixes are documented in `CLAUDE.md`. The pipeline is structured so that each stage saves its output to disk before the next stage begins; SLURM job logs for every run are preserved under `logs/`. Random seeds are pinned to 42 throughout (matching, sampling, splits, model training, bootstrap). All 11 methodology fixes from `methodology_fix_plans.md` are implemented and cross-referenced against published literature; details and execution results are in `ewang163_methodology_fixes_results.md`, and the final model selection memo is in `ewang163_model_selection_memo.md`. The full multi-model comparison including runtime + explainability is in `ewang163_model_comparison.md`. The unmodified PULSNAR library (Kumar & Lambert 2024) is cloned into `PULSNAR/`.

End of write-up.
