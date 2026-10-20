# 01 — Product Requirements Document (PRD)

**Project:** NeuroSight AI — Longitudinal MRI Dementia Analysis
**Version:** 1.0 · 2026-09-14 · Status: approved for engineering planning
**Source assets:** Local OASIS-2 dataset, baseline pipeline code, reports, legacy upgrade proposal (superseded — see `00-PLAN-REVIEW.md`).

---

## 1. Problem

Dementia diagnosis today relies on cognitive testing (MMSE, CDR) and subjective radiologist review. MRI-based decision support that uses **longitudinal** scans (change over time) is more informative than single-scan classification, but no usable, honest, end-to-end system exists for the OASIS-2 data this team already holds. The prior student project reached ~72% held-out accuracy with hardcoded paths, silent failures, no backend, and a report template with no numbers in it.

## 2. Users and jobs-to-be-done

| User | Job | Success looks like |
|------|-----|--------------------|
| ML researcher (primary) | Train, compare, and reproduce models on OASIS-2 without touching code paths | One command trains the full ablation; MLflow has every run; results identical across reruns |
| Clinician / demo viewer (secondary) | Upload a patient's visits, get a calibrated status + risk read with evidence | Upload → results in <60 s, Grad-CAM evidence, explicit uncertainty and disclaimer |
| Course evaluator (tertiary) | Read the report and trust the numbers | Every table has real values, CIs, input-mode labels, and a model card |

**Explicit non-user:** this is research/decision-support software. It is **not** a medical device, makes no diagnosis, and must never be marketed as one (US FDA clearance out of scope; disclaimer required on every result surface).

## 3. Product scope

### In scope (v1.0)
1. **Reproducible research pipeline** on the full local OASIS-2 copy (150 subjects, 373 sessions across `OAS2_RAW_PART1/2`): acquisition of the missing demographics CSV, schema-validated loading, deterministic preprocessing, patient-level splits, 5-fold CV + held-out test, 11-variant ablation (ported from existing code), MLflow tracking.
2. **Two learning tasks:**
   - **T1 Current status (primary):** binary Non-demented vs Demented at final visit; CDR-derived labels, Group-verified; Converted patients split into pre/post sequences at the conversion boundary.
   - **T2 Conversion risk (secondary):** for pre-conversion/non-demented sequences, predict conversion risk score from the visit sequence — delivers the "is the disease progressing / will it decline" questions.
3. **Leakage-controlled input modes** for every model: `mri_only` (headline), `demo_only` (age/sex/education/SES), `mri+demo`, `mri+clinical` (adds MMSE/CDR — reported separately, never the headline claim).
4. **Explainability:** Grad-CAM per visit/slice + temporal-attention weights (which visits drove the call).
5. **Inference service:** FastAPI backend with background-job analysis of multi-visit uploads, calibrated probabilities (temperature scaling), model-version pinning, SQLite session store, React dashboard (Analyze wizard, Patient timeline, About/model card).
6. **Trust artifacts:** MODEL_CARD.md, DATA.md (provenance/license/citation), final PDF report regenerated from real run outputs (CLAIM/TRIPOD+AI-aligned), presentation refresh.
7. **Engineering hygiene:** package structure, config system, logging, CI (lint+type+tests), pre-commit, Docker deployment, tests at every layer.

### Out of scope (v1.0)
- DICOM/PACS integration, multi-scanner harmonization, skull-stripping/registration beyond what's specified.
- External validation on ADNI (documented as future work; hooks provided).
- Multi-user auth with accounts/roles (single shared API key for demo), cloud deployment, GPU cluster training.
- Mobile UI, i18n.

## 4. Functional requirements

**FR-1 Data** — The pipeline must run from a config that points at `data/raw/OAS2_RAW_PART1|PART2` and a demographics CSV; it must schema-validate the CSV (columns `Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF`; 373 rows; group counts within documented tolerance), match sessions to raw folders, and emit a `data_audit.json` (matched/missing per subject). Missing session → warn + skip; CSV row without raw data → recorded, never silently dropped.

**FR-2 Preprocessing** — Deterministic: shape fix → z-score → 32 middle slices (strategy configurable: middle/uniform/entropy) → 128×128 resize; 3D path to (64,128,128). Any load failure **raises** (no zero-tensor substitution). mpr-series policy configurable (`first` default, `middle`, `mean`), logged per session. Train-time augmentation: ±2 slice shift, Gaussian noise σ=0.02, h-flip p=0.3, intensity jitter ±5%, visit-dropout p=0.3 (≥2 visits kept).

**FR-3 Splits** — Patient-level 70/15/15 stratified by final label, seed 42, plus 5-fold stratified CV on train+val. A pytest leakage test must assert zero subject overlap across splits. Held-out test is touched exactly once per model, at final evaluation.

**FR-4 Models** — All 11 ablation variants (CNN-avg, CNN-last, CNN+LSTM, +SpatialAttn, +TemporalAttn, +SE, +DualAttn, CNN+Transformer, 3D-CNN, 3D-CNN+LSTM, ResNet18+LSTM-pretrained) must train via one registry with the selected input mode; factory-constructed from config; every forward pass accepts variable visit counts via masks.

**FR-5 Training** — Focal loss or weighted CE (config), label smoothing option, gradient clip 1.0, Adam/AdamW, ReduceLROnPlateau, early stopping (patience 7), class-weighted sampling option, TensorBoard + MLflow logging of params/metrics/artifacts, deterministic seeds + cudnn flags from config, checkpointing (best + last).

**FR-6 Evaluation** — Held-out test: accuracy with bootstrap 95% CI (1000 resamples), balanced accuracy, AUC-ROC, AP, F1 (binary + macro), precision, recall/sensitivity, specificity, MCC, Brier, ECE before/after temperature scaling, confusion matrix, ROC/PR/calibration plots. CV: mean±std per metric. Model comparison: McNemar's test. T2: AUC-ROC and AUC-PR for conversion prediction with explicit small-n caveat. **Every metric table labels its input mode.**

**FR-7 Explainability** — Grad-CAM heatmaps (per visit, key slices) + overlay PNGs; temporal attention bar charts per patient; population-level attention-by-position chart grouped by class.

**FR-8 Inference API** — `POST /api/v1/analyses` (multipart upload of ≥1 visit's `.img`+`.hdr` pairs, optional demographics form) returns `analysis_id` and processes in a background worker; `GET /api/v1/analyses/{id}` returns queued/running/succeeded/failed + full result when done; results include predicted class, calibrated probabilities, confidence band, risk level, progression trend (T1 across provided visits), conversion risk (T2), Grad-CAM images (base64) when requested, model version. Session store (SQLite) keeps history per pseudo-subject-id. Uploads validated (extension, pairing, size ≤ 512 MB/study, header sanity via nibabel open).

**FR-9 Frontend** — Three routes (Analyze, Patient, About) implementing the wizard in `03-APP-FLOW.md`, with the design system in `04-UI-UX-BRIEF.md`, polling for job status, and the medical disclaimer visible on every result surface.

**FR-10 Reports** — `python -m neurosight.reporting.all` regenerates: ablation tables/plots, final PDF report and PPTX **with real numbers pulled from run artifacts** (no hardcoded `--`), MODEL_CARD.md metrics section auto-filled from the champion run's `metrics.json`.

## 5. Non-functional requirements

| ID | Requirement |
|----|-------------|
| NFR-1 Reproducibility | Two clean runs of `train --model 3_cnn_lstm --seed 42` produce identical best-val-acc on CPU-deterministic settings; every run reproducible from config + MLflow metadata |
| NFR-2 Performance | Training the 11-variant ablation completes overnight on RTX 4050 6 GB (batch sizes per existing `config_3d.py` tuning); single-visit CPU inference < 10 s; 3-visit study analysis < 60 s |
| NFR-3 Honesty | No result table without CIs; no MRI headline claim from a model that saw MMSE/CDR; benchmark comparisons cite dataset variant (OASIS-1 vs OASIS-2) |
| NFR-4 Privacy | Uploaded scans stored only under `data/uploads/` (never webroot/logs), auto-purged after 7 days, never written to logs; demo runs local-only |
| NFR-5 Quality gates | ruff + black clean; mypy on `neurosight/` core packages; pytest ≥ 80% coverage on data/model/eval packages; CI green before merge |
| NFR-6 Portability | `docker compose up` serves frontend+backend; native Windows dev flow documented (no Docker required for training) |
| NFR-7 Compliance posture | Disclaimer + "research only, not a medical device" on all surfaces; OASIS data-use agreement + citation (Marcus et al. 2010) in DATA.md; MODEL_CARD.md per CHAI-informed template |

## 6. Success criteria (measurable, honest)

| # | Criterion | Target |
|---|-----------|--------|
| S1 | T1 `mri_only` held-out test accuracy | ≥ 75% with bootstrap 95% CI reported (credible band for rigorous OASIS-2 longitudinal: 60–86%) |
| S2 | T1 `mri_only` held-out test AUC-ROC | ≥ 0.80 |
| S3 | CV stability | 5-fold accuracy mean ≥ 72% with std ≤ 8 points, all folds reported |
| S4 | Calibration | ECE ≤ 0.10 after temperature scaling; calibration plot saved |
| S5 | Leakage control | `mri_only` vs `mri+clinical` gap quantified and reported; `demo_only` baseline (logistic regression) documented |
| S6 | T2 conversion head | AUC-ROC + AUC-PR reported on pre-conversion sequences with n stated; no target (exploratory) |
| S7 | Reproducibility | NFR-1 determinism check passes; ablation rerun from configs alone |
| S8 | Product | Upload→result E2E demo works locally via docker compose; all NFR-5 gates green |
| S9 | Trust artifacts | MODEL_CARD.md, DATA.md, real-numbered final_report.pdf + presentation.pptx exist and match run artifacts |

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CSV acquired but schema differs from Kaggle mirror | Medium | Schema validator with column aliases (`oasis_longitudinal.csv` == `oasis_longitudinal_demographics.csv`); fixture test with both header variants |
| Tiny test set (~22 sequences) → noisy point estimates | Certain | Bootstrap CIs mandatory; report CV as primary evidence; never claim single-decimal precision |
| Converted-split inflates patient count (same subject in train & test pre/post) | High | Split at **subject** level before sequence splitting — leakage test must cover this exact case |
| GPU OOM on 3D models | Medium | Keep batch 2 + grad-accum 4 tuning; config guard that checks VRAM and reduces batch |
| 99%-paper benchmark pressure | Medium | Benchmark table in report distinguishes OASIS-1/leaky vs rigorous longitudinal setups (research-backed) |
| Upload abuse (huge/zip-bomb files) | Low | Size caps, extension+header validation, per-IP rate limit |

## 8. Release plan

- **M-demo** (end of Engineering Plan): local end-to-end demo with champion model + full trust artifacts.
- **v1.1 candidates (documented, not built):** ADNI external validation, ONNX runtime serving, multi-user auth, DICOM ingest.
