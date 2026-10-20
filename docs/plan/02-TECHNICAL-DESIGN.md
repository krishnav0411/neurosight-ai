# 02 — Technical Design

**Project:** NeuroSight AI · v1.0 · 2026-09-14
**Inputs:** `01-PRD.md` (requirements), `00-PLAN-REVIEW.md` (defects driving these decisions), existing baseline code in `src/` (ported, not rewritten).

---

## 1. Architecture overview

```
                         ┌────────────────────────────────────────────────┐
                         │                neurosight-ai/               │
   OASIS-2 (data/raw        │                                                │
   folder) ─────────────►│  configs/ (YAML) ──► src/neurosight/           │
   + demographics CSV    │                       data/      models/       │
                         │                       training/  evaluation/   │
                         │                       inference/ reporting/    │
                         │                       utils/                   │
                         │  backend/ (FastAPI + worker + SQLite)          │
                         │  frontend/ (React 18 + TS + Vite)              │
                         │  mlruns/ · models/ · reports/ · tests/         │
                         └────────────────────────────────────────────────┘
```

Two planes share one package:
- **Research plane** (CLI): `python -m neurosight.data.audit` → `preprocess` → `train` (CV/ablation) → `evaluate` → `explain` → `reporting.all`.
- **Serving plane** (FastAPI): loads a pinned TorchScript/`.pth` checkpoint + the same preprocessing object; background worker executes analyses; SQLite stores results.

The preprocessing/labeling code is **shared verbatim** between planes — a single source of truth is the whole point of the design (the old repo's breakage came from forked generations).

## 2. Key decisions and trade-offs

| # | Decision | Alternatives considered | Rationale |
|---|----------|------------------------|-----------|
| D1 | **Evolve + port** existing July/root code into a clean package; do not re-derive | Full rewrite (old plan); keep patching in place | Ablation registry, CV, eval suite, Grad-CAM, report generators already work; rewrite risks reintroducing bugs; in-place patching can't fix the forked-generations mess |
| D2 | **Binary primary + conversion head** instead of 3-class softmax | 3-class (old plan); ordinal regression | ~31 Converted subjects can't sustain a class; binary matches label derivation (CDR) and the existing working loader; conversion head delivers the progression promise without a degenerate class |
| D3 | **Subject-level split before Converted sequence splitting** | Split sequences (old plan implicitly) | Pre/post sequences of the same subject must never straddle splits — leakage the old loader risks |
| D4 | **Input modes as config, one headline** (`mri_only`) | Fuse MMSE silently (old plan) | MMSE/CDR define the label → leakage; modes make the leak measurable instead of hidden |
| D5 | **Pydantic v2 + YAML configs** (flat: `data/model/train/serving.yaml`) | Hydra, OmegaConf | Matches old plan's promise, enough for this scale, typed validation, no new paradigm |
| D6 | **Background-job inference** (job id + poll) | Synchronous endpoint (old plan) | A 3-visit study takes 10–60 s to preprocess+infer; sync requests time out; worker model also rate-limits GPU use |
| D7 | **TorchScript export for serving** (optional per model; `.pth` fallback) | Raw `.pth` only | Scripted module pins architecture+weights together; guards the "app loads incompatible checkpoint" failure seen in the old repo (`app.py` importing nonexistent `preprocess_scan`) |
| D8 | **SQLite via SQLModel** for sessions | Postgres, JSON files | Zero-ops demo; typed models; trivially swappable later |
| D9 | **MLflow file-store** (`mlruns/`) + optional server in compose | W&B, TensorBoard only | Offline-friendly; TensorBoard kept for curves; matches old plan intent |
| D10 | **Bootstrap CIs + McNemar + CV as primary evidence** | Single split point estimates (old plan) | n_test ≈ 22 sequences; point estimates lie; standards (TRIPOD+AI/CLAIM) require uncertainty |

## 3. Component design

### 3.1 Package layout

```
neurosight-ai/
├── configs/                 data.yaml · model.yaml · training.yaml · serving.yaml
├── data/                    raw/ (local or mapped directory) · metadata/ · processed/ · uploads/
├── src/neurosight/
│   ├── config.py            Pydantic schemas: DataConfig, ModelConfig, TrainingConfig, ServingConfig; load_config()
│   ├── data/
│   │   ├── audit.py         CSV schema validation, session↔CSV matching, data_audit.json
│   │   ├── loader.py        OASIS2Loader (ported; subject-level splits; Converted handling; series policy)
│   │   ├── preprocessing.py fix_shape, z_score, select_slices, resize, augment (ported; exceptions raise)
│   │   ├── dataset.py       OASIS2D/3D datasets, collate fns, weighted sampler (ported)
│   │   └── cache.py         processed-tensor cache (replaces 1.08 GB monolithic dataset_cache.pt)
│   ├── models/
│   │   ├── registry.py      ABLATION_VARIANTS + build_model(key, config) — ported from ablation.py
│   │   ├── cnn_lstm.py attention.py se_cnn.py cnn3d.py cnn3d_lstm.py transformer.py ensemble.py   (ported)
│   │   ├── heads.py         StatusHead (T1) · ConversionHead (T2, shared encoder)
│   │   └── export.py        torch.jit.script/export helpers
│   ├── training/
│   │   ├── losses.py        FocalLoss, class weights (ported)
│   │   ├── trainer.py       Trainer (ported; +determinism flags, MLflow hooks)
│   │   ├── cv.py            5-fold CV driver (ported from train_cv.py)
│   │   ├── search.py        bounded hyperparameter search (ported)
│   │   └── metrics.py       MetricsTracker
│   ├── evaluation/
│   │   ├── evaluator.py     compute_metrics, bootstrap_ci, McNemar (ported from evaluate.py)
│   │   ├── calibration.py   temperature scaling on val, ECE
│   │   ├── gradcam.py       GradCAM, saliency, overlays (ported)
│   │   └── attention_viz.py temporal-attention plots (ported concept from old plan)
│   ├── inference/
│   │   ├── predictor.py     loads pinned model+preprocessor; predict_study(visits)→StudyResult
│   │   └── validation.py    file/pair/header checks
│   ├── api/  (backend/)     main.py · deps.py(auth) · routes/analyses.py, health.py · jobs.py · store.py
│   ├── reporting/           report_pdf.py · report_pptx.py · model_card.py (ported generators, data-driven)
│   └── utils/               logging.py seed.py io.py device.py
├── scripts/                 thin CLI entry points (click)
├── tests/                   unit/ · integration/ · fixtures/ (synthetic 2-subject mini-OASIS)
├── frontend/                React 18 + TS + Vite + Tailwind + shadcn/ui
├── reports/                 outputs (figures/, csv/, final_report.pdf, presentation.pptx)
├── models/                  checkpoints + export/ (TorchScript) + manifest.json
├── .github/workflows/ci.yml · Dockerfile.backend · Dockerfile.frontend · docker-compose.yml
└── pyproject.toml · requirements.lock · .pre-commit-config.yaml · MODEL_CARD.md · DATA.md · README.md
```

### 3.2 Data model (pipeline)

- `SubjectRecord`: `subject_id`, `sessions: [SessionRecord]`, `final_label ∈ {0,1}`, `group ∈ {Nondemented, Demented, Converted}`, `conversion_visit: int|None`.
- `SessionRecord`: `visit_num`, `mri_id`, `series_paths` (per policy), `cdr`, `mmse`, `meta_vec(8)`, `mr_delay`.
- Sequence construction (T1): Converted subject → **two** records (pre-conversion labeled 0, post labeled 1) *after* subject-level split. Patients with <2 resolvable visits are excluded and counted in `data_audit.json`.
- Metadata vector (8-d, unchanged from working code): `[M/F, Age/100, EDUC/20, SES/5, MMSE/30, eTIV/2000, nWBV, ASF/2]` — but **mode-masked** at the head: `mri_only`→zeros+flag, `demo_only`→only demo dims, `mri+clinical`→all. The mask is part of the checkpoint manifest so serving can't drift from training.

### 3.3 Model design

- Per-slice CNN encoders → mean-pool over slices → per-visit vector → temporal encoder (LSTM/Transformer/3D variants) → head.
- **StatusHead (T1):** masked temporal context (+masked metadata branch when enabled) → MLP → 2 logits; temperature-scaled at serving.
- **ConversionHead (T2):** shares the frozen-or-finetuned temporal encoder; binary head trained only on pre-conversion/non-demented sequences; output = conversion risk. Class imbalance handled with focal loss + AUC-PR reporting (exploratory per PRD S6).
- Param budgets unchanged (0.53–1.4 M for customs; ResNet18+LSTM ≈ 11 M) — sized for the data and the 6 GB GPU.

### 3.4 Training protocol

1. `data.audit` → `data_audit.json` (hard gate; fail on CSV schema mismatch).
2. `preprocess` → per-session `.npz` cache under `data/processed/` (filename = `subject_visit_series.pt`, checksum in manifest).
3. Split (subject-level, stratified, seed 42) → CV folds on train+val; test untouched.
4. For each registry variant × selected input mode: train (early stop on val AUC **not** accuracy — imbalanced), log to MLflow, save best/last checkpoints.
5. Champion selection rule (pre-registered): highest **val AUC** among variants, tie-broken by lower ECE and fewer params.
6. Final: fit temperature scaling on val; single test-set evaluation; McNemar vs runner-up; generate all reports.

### 3.5 Serving design

- FastAPI (pydantic v2) + uvicorn worker + `BackgroundTasks`/threadpool executor; job states `queued→running→succeeded|failed` in SQLite; SSE endpoint optional over polling.
- `POST /api/v1/analyses` enforces: API key (`X-API-Key`, dev-shared-secret), multipart caps (512 MB/study, ≤8 visits, ≤4 series/visit), `.img/.hdr` pairing, nibabel header open (shape ∈ expected range), demographics form optional per mode.
- Response schema (v1): `analysis_id, status, subject_ref, model_version, input_mode, predicted_class, class_labels, probabilities (calibrated), confidence_band, risk_level, progression_trend, conversion_risk, gradcam: [base64 png per visit] | null, created_at, completed_at`.
- Retention: uploads purged after 7 days (background task at startup + daily); results rows kept (no PHI beyond pseudo-id).
- CORS: frontend origin only; docs disabled when `serving.public=false`.

## 4. Constraints

- **Hardware:** dev box = Windows 11, RTX 4050 6 GB → 2D batch 4, 3D batch 2 + grad-accum 4 (inherited from `config_3d.py`); CPU inference acceptable for demo (models ≤ 1.4 M params).
- **Dataset fixed** at OASIS-2 local copy; no internet needed at train time; CSV is the only external acquisition (one-time, task M0).
- **Windows paths:** `pathlib.Path` everywhere; `NUM_WORKERS=0` (Windows multiprocessing constraint, already learned in old configs); junction (not symlink) for `data/raw`.
- **Licensing:** OASIS data = research-use agreement; no redistribution of scans in repo (`.gitignore data/`), citation required (Marcus et al. 2010).

## 5. Technology choices (final)

| Layer | Choice | Version floor |
|-------|--------|---------------|
| Language | Python | 3.11+ (3.13 in use) |
| DL | PyTorch + torchvision | 2.x |
| Imaging | NiBabel, SciPy, OpenCV-headless | 5 / 1.10 / 4.8 |
| ML/serving | FastAPI, pydantic v2, uvicorn, SQLModel, python-multipart | latest stable |
| Experiment | MLflow, TensorBoard | 2.x |
| Config | PyYAML + Pydantic settings | 2.x |
| Quality | pytest, ruff, black, mypy, pre-commit | latest |
| Frontend | React 18, TypeScript, Vite 5, Tailwind 3, shadcn/ui, react-query, recharts, axios | LTS |
| Deploy | Docker + compose (backend, frontend/nginx, optional mlflow profile) | compose v2 |
| CI | GitHub Actions (ubuntu): ruff+black → mypy → pytest (CPU, fixtures) → frontend build | — |

## 6. Test strategy (summary — detail in 06)

- **Unit:** preprocessing (shape/normalization/slice-selection invariants), metadata masking per mode, collate/mask correctness, metrics vs sklearn reference values, config validation errors.
- **Property/integration:** no-leakage split test (incl. Converted straddle), trainer smoke test (2 epochs on synthetic 6-subject fixture reaches overfit sanity), evaluator on fixture, Grad-CAM shape/grad-flow, predictor E2E on fixture study.
- **API:** auth 401, validation 422/413, job lifecycle with mocked predictor, retention task.
- **CI fixture:** `tests/fixtures/` mini-OASIS (2 subjects × 2 visits × synthetic 64×64×32 volumes + mini CSV) so no real data is needed in CI.
