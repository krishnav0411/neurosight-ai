# 05 — Backend Design

**Project:** NeuroSight AI · v1.0 · 2026-09-14
**Covers:** data model, storage, API surface, access rules, background jobs, events, and the inference runtime. Stack rationale in `02-TECHNICAL-DESIGN.md` §3.5.

---

## 1. Runtime shape

```
uvicorn backend.main:app  (single process)
├── FastAPI app
│   ├── middleware: CORS (frontend origin only), request-id, structured access log (no bodies)
│   ├── routes: /health · /api/v1/analyses · /api/v1/patients · /api/v1/model-card
│   └── startup: load model manifest → warm predictor → purge expired uploads
├── AnalysisWorker (threadpool, max_concurrency=1 on GPU / 2 on CPU)
│   └── runs neurosight.inference.predictor on queued analyses
└── SQLite (SQLModel, WAL mode) at backend/data/neurosight.db
```

Model artifacts are read from `models/manifest.json`:

```json
{ "active_model": "3_cnn_lstm", "input_mode": "mri_only",
  "checkpoint": "models/export/3_cnn_lstm_ts.pt", "fallback_pth": "models/best_3_cnn_lstm.pth",
  "temperature": 1.07, "metrics_digest": {"test_acc": 0.76, "test_auc": 0.83, "ece": 0.06},
  "trained_at": "...", "git_sha": "...", "data_audit_sha": "..." }
```

`model_version` in every response = `f"{active_model}@{git_sha[:8]}[{input_mode}]"`. The API refuses to start if the manifest is missing or the checkpoint fails a forward-pass probe on a synthetic tensor.

## 2. Data model (SQLite via SQLModel)

```python
class AnalysisJob(SQLModel, table=True):
    id: str = Field(primary_key=True)            # uuid4 hex
    subject_ref: str | None                      # user-supplied pseudonym or client UUID
    status: str                                  # queued|running|succeeded|failed|cancelled
    stage: str | None                            # preprocessing|inference|gradcam
    input_mode: str                              # mri_only|demo_only|mri_demo|mri_clinical
    visit_count: int
    demographics_json: str | None                # masked per input_mode
    include_gradcam: bool
    error_message: str | None
    result_json: str | None                      # StudyResult payload, JSON
    model_version: str | None
    created_at: datetime; started_at: datetime | None; completed_at: datetime | None
    upload_dir: str                              # data/uploads/<id>/, purged post-run +7d

class PurgeLog(SQLModel, table=True):            # audit of deletions
    id: int; job_id: str; purged_at: datetime; bytes_freed: int
```

No table stores raw scans or patient identifiers; `subject_ref` is whatever opaque string the client sends. Results are retained indefinitely (research record); uploads are transient.

## 3. API surface (v1)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | none | `{status:"ok", model_version, uptime_s, device}` — 200 always when up |
| POST | `/api/v1/analyses` | key | multipart upload → 201 `{analysis_id, status:"queued", poll_url}` |
| GET | `/api/v1/analyses/{id}` | key | job status + result when succeeded |
| DELETE | `/api/v1/analyses/{id}` | key | cancel (queued/running) or delete record (terminal) |
| GET | `/api/v1/analyses` | key | list (paginated, newest first) |
| GET | `/api/v1/patients` | key | distinct `subject_ref` list with last-analysis summary |
| GET | `/api/v1/patients/{ref}/history` | key | that subject's analyses, chronological |
| GET | `/api/v1/model-card` | none | MODEL_CARD.md content + metrics digest JSON |
| GET | `/api/v1/analyses/{id}/events` | key | optional SSE stream of stage changes (fallback: poll) |

### 3.1 `POST /api/v1/analyses` contract

Multipart form:
- `files[]` — `.img`/`.hdr` (and `.nii` accepted). Required: ≥1 complete pair per visit.
- `visit_labels[]` — optional explicit grouping (else parsed from `OAS2_XXXX_MRY` / `mpr-N` patterns).
- `subject_ref` — optional string ≤ 64 chars.
- `demographics` — optional JSON string (Age, Sex, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF).
- `input_mode` — optional override; validated against model manifest capability.
- `include_gradcam` — bool, default true.

Validation pipeline (fail fast, 422 with per-file reasons):
1. Content-type/extension allow-list; total bytes ≤ 512 MB; ≤ 8 visits; ≤ 4 series/visit.
2. Every `.img` has its `.hdr` twin (same stem); ≥ 1 complete visit.
3. nibabel header probe per pair: opens, ndim ∈ {3,4}, voxel dims within 20% of (176,208,176)-family; reject otherwise ("not an OASIS-format volume").
4. If `input_mode` requires demographics (demo modes) → form must carry required fields; else ignored (logged).

### 3.2 Result payload (`result_json`, mirrored in `GET` response)

```json
{ "predicted_class": 1, "class_labels": ["Non-Demented","Demented"],
  "probabilities": {"Non-Demented": 0.13, "Demented": 0.87},
  "raw_probabilities": {"Non-Demented": 0.18, "Demented": 0.82},
  "confidence": 0.87, "confidence_band": "adequate|low",
  "risk_level": "High", "input_mode": "mri_only", "model_version": "...",
  "longitudinal": { "trend": "Declining", "per_visit_probability": [0.21, 0.55, 0.87],
                    "visit_labels": ["MR1","MR2","MR3"] },
  "conversion_risk": { "score": 0.64, "available": true },
  "gradcam": [ { "visit": "MR1", "images": { "original": "<b64>",
                "heatmap": "<b64>", "overlay": "<b64>" }, "slices": [10,16,24] } ],
  "caveats": ["small_test_set", "research_only"] }
```

Grad-CAM slices fixed at `[10,16,24]` (brain-core slices used in report figures); images pre-resized to 512 px JPEG-quality-85 PNG to keep payloads < 2 MB.

## 4. Access rules & security

- **Auth:** static shared API key via `SERVING_API_KEY` env (`.env`, gitignored); constant-time compare; 401 with `WWW-Authenticate` on miss; `/health` and `/model-card` exempt.
- **Rate limit:** 10 analyses/min/IP (slowapi) — protects the single GPU worker.
- **Upload safety:** size caps above; files written under `data/uploads/<job_id>/` with randomized stems; **no user-supplied filename ever used in a path**; uploads never logged.
- **PHI posture:** store-only-what's-needed (demographics form masked to the active input mode before persistence); 7-day upload purge at startup + daily; README's privacy section documents local-only operation.
- **Headers:** `X-Content-Type-Options: nosniff`; docs UI (`/docs`) disabled when `SERVING_PUBLIC=false` (default).

## 5. Background jobs & events

- Worker claims `queued` jobs FIFO, sets `running` + `stage`, executes predictor, writes `result_json`, sets `succeeded` (or `failed` + `error_message` with a human message; stack traces go to the log file only).
- Cancellation: cooperative — worker checks job status between visits; a cancelled job's upload dir is purged immediately.
- Events (for SSE/poll parity): `job.created`, `job.stage_changed`, `job.succeeded`, `job.failed`, `job.cancelled` — emitted to an in-process pub/sub; SSE handler subscribes per job id. Polling is the primary client path; SSE is an optimization, not a dependency.
- Retention sweeper: on startup + every 24 h → purge upload dirs of terminal jobs older than 7 days → `PurgeLog` row.

## 6. Inference runtime (`neurosight.inference.predictor`)

1. Load manifest → TorchScript module if present else `.pth` + registry rebuild (arch pinned by manifest `model_key` + `input_mode`; the manifest's metadata mask is applied to demographics before the model sees them).
2. Group uploads into visits (per §3.1), apply the **shared** `preprocessing` pipeline (train/serve parity — same code path, `augment=False`, mode `eval` transforms).
3. Build study tensor `(1, T, S, 1, H, W)` + mask + metadata vector (zeros per mask) → forward → softmax → **temperature scaling** (from manifest) → calibrated probabilities.
4. Longitudinal fields: per-visit Demented probability from the same forward (temporal attention weights kept for the UI strip); trend label thresholds: |Δ| ≤ 0.05 Stable, ≤ 0.15 Declining, > 0.15 Rapidly Declining (largest consecutive rise).
5. Conversion risk: ConversionHead forward when `T` ≥ 2 and manifest includes it; `available:false` otherwise — never a fabricated number.
6. Grad-CAM (requested): per visit on slices [10,16,24], overlays via the ported `gradcam.py` helpers.
7. Everything inside `torch.inference_mode()`; device from manifest (`cuda` if available else `cpu`); per-visit timing logged.

## 7. Error taxonomy (HTTP → UI copy)

| Condition | HTTP | `detail` |
|-----------|------|----------|
| Missing/invalid API key | 401 | "Invalid or missing API key." |
| Rate limit | 429 | "Too many analyses — retry in N s." |
| Unpaired/incompatible files | 422 | per-file list, e.g. "mpr-2.nifti.img has no matching .hdr" |
| Payload too large | 413 | "Study exceeds 512 MB." |
| Unknown job id | 404 | "Analysis not found." |
| Predictor crash | 500 | "Analysis failed during inference. The error has been logged." (job → failed) |

## 8. Configuration & deployment

- `configs/serving.yaml`: host/port, worker concurrency, upload caps, retention days, CORS origins, public flag; secrets via env only (`.env.example` committed, `.env` ignored).
- `docker-compose.yml` services: `backend` (python:3.11-slim + `libgomp1`, models/ + configs/ baked, data/ volume), `frontend` (nginx, `/api` proxy → backend:8000), optional `mlflow` profile (5000). Healthchecks on both; `depends_on: condition: service_healthy`.
- Native (no Docker) dev flow documented: `uvicorn backend.main:app --reload` + `npm run dev` — matches Windows dev box where training happens.
