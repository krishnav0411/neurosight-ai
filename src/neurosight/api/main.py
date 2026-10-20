"""main.py — FastAPI serving plane (plan doc 05; milestone M6).

    uvicorn neurosight.api.main:app --config-dir configs

Startup: load model manifest -> warm predictor (forward probe) -> start the
FIFO worker + retention sweeper. /health and /model-card are unauthenticated;
everything else requires the X-API-Key header (constant-time compare).
"""

from __future__ import annotations

import hmac
import json
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from neurosight.api.jobs import AnalysisWorker
from neurosight.api.store import AnalysisJob, JobStore, result_to_dict
from neurosight.api.validation import UploadValidationError, validate_and_persist
from neurosight.config import load_config
from neurosight.inference.predictor import NeuroSightPredictor
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

_STATE: dict = {}


def create_app(
    config_dir: str = "configs", predictor: NeuroSightPredictor | None = None
) -> FastAPI:
    cfg = load_config(config_dir).serving
    store = JobStore(cfg)
    if predictor is None:
        predictor = NeuroSightPredictor.from_manifest_file(
            Path(cfg.models_dir) / "manifest.json",
            models_dir=cfg.models_dir,
            num_slices=32,
            img_size=128,
        )
    worker = AnalysisWorker(store, predictor, cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        store.purge_expired_uploads(cfg.uploads_dir, cfg.retention_days)
        worker.start()
        logger.info("NeuroSight API ready: %s", predictor.model_version)
        yield
        worker.stop()

    # env override for hosted deploys: NEUROSIGHT_CORS_ORIGINS="https://a.vercel.app,https://b.vercel.app"
    cors_env = os.environ.get("NEUROSIGHT_CORS_ORIGINS")
    cors_origins = (
        [o.strip() for o in cors_env.split(",") if o.strip()] if cors_env else cfg.cors_origins
    )
    app = FastAPI(
        title="NeuroSight AI API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if cfg.public else None,
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _STATE.update({"cfg": cfg, "store": store, "worker": worker, "predictor": predictor})

    # ── middleware: request id + rate limit on analysis creation only ───────
    rate_bucket: dict[str, deque[float]] = defaultdict(deque)

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        client = request.client.host if request.client else "anon"
        now = time.time()
        # only analysis CREATION is capped: read-only polling (job status, SSE)
        # and /health must never 429 — the dashboard exhausts the window alone
        limited = request.method == "POST" and request.url.path == "/api/v1/analyses"
        if limited:
            window = rate_bucket[client]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= 10:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many analyses — retry shortly."},
                )
            window.append(now)
        # bound the bucket: drop windows that have fully drained
        for key in [k for k, w in rate_bucket.items() if not w]:
            del rate_bucket[key]
        return await call_next(request)

    app.state.rate_bucket = rate_bucket  # exposed for tests/observability

    # ── auth ────────────────────────────────────────────────────────────────
    def require_api_key(request: Request) -> None:
        expected = os.environ.get(cfg.api_key_env, "")
        if not expected:
            raise HTTPException(500, detail=f"server misconfigured: {cfg.api_key_env} not set")
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(401, detail="Invalid or missing API key.")

    # ── unauthenticated ─────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_version": predictor.model_version,
            "uptime_s": round(time.time() - _START[0], 1),
            "device": str(predictor.device),
        }

    @app.get("/api/v1/model-card")
    def model_card():
        path = Path("MODEL_CARD.md")
        content = (
            path.read_text(encoding="utf-8")
            if path.exists()
            else ("MODEL_CARD.md not yet generated — see docs/plan/06-ENGINEERING-PLAN.md T8.2.")
        )
        return {"content": content, "metrics_digest": predictor.manifest.get("metrics_digest")}

    # ── analyses ────────────────────────────────────────────────────────────
    @app.post("/api/v1/analyses", status_code=201, dependencies=[Depends(require_api_key)])
    async def create_analysis(
        files: list[UploadFile] = File(...),
        subject_ref: str | None = Form(None),
        demographics: str | None = Form(None),
        visit_labels_json: str | None = Form(None),
        include_gradcam: bool = Form(True),
    ):
        job_id = uuid.uuid4().hex
        staging_dir = cfg.uploads_dir / f".tmp_staging_{job_id}"
        staged_files: list[tuple[str, bytes | Path]] = []
        max_bytes = cfg.max_upload_mb * 1024 * 1024
        total_read = 0
        chunk_size = 64 * 1024
        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                staged_path = staging_dir / f"{uuid.uuid4().hex}.tmp"
                with open(staged_path, "wb") as out_f:
                    while True:
                        chunk = await f.read(chunk_size)
                        if not chunk:
                            break
                        total_read += len(chunk)
                        if total_read > max_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Upload payload exceeds maximum allowed size of "
                                    f"{cfg.max_upload_mb} MB"
                                ),
                            )
                        out_f.write(chunk)
                staged_files.append((f.filename or "unnamed", staged_path))
        except Exception:
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        finally:
            for f in files:
                try:
                    await f.close()
                except Exception:
                    pass
        # visit_labels_json: JSON LIST parallel to files (browsers strip folder
        # paths, so the client supplies the visit for each file explicitly);
        # malformed JSON is a client error -> 422, never a 500
        try:
            explicit = json.loads(visit_labels_json) if visit_labels_json else None
            demo = json.loads(demographics) if demographics else None
        except json.JSONDecodeError as exc:
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
            raise HTTPException(422, detail=f"malformed JSON form field: {exc}") from exc
        if explicit is not None and (
            not isinstance(explicit, list) or not all(isinstance(x, str) for x in explicit)
        ):
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
            raise HTTPException(422, detail="visit_labels_json must be a JSON list of labels")
        if demo is not None and not isinstance(demo, dict):
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
            raise HTTPException(422, detail="demographics must be a JSON object")

        try:
            upload = validate_and_persist(staged_files, cfg, job_id, explicit)
        except UploadValidationError as exc:
            raise HTTPException(422, detail={"problems": exc.problems}) from exc
        finally:
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)

        manifest = {"visits": upload.visits, "visit_labels": upload.visit_labels}
        manifest_path = cfg.uploads_dir / job_id / "visit_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        if subject_ref and len(subject_ref) > 64:
            raise HTTPException(422, detail="subject_ref must be <= 64 characters")
        job = AnalysisJob(
            id=job_id,
            subject_ref=subject_ref,
            input_mode=predictor.input_mode,
            visit_count=len(upload.visits),
            visit_labels_json=json.dumps(upload.visit_labels),
            demographics_json=json.dumps(demo) if demo else None,
            include_gradcam=include_gradcam,
            upload_dir=str(cfg.uploads_dir / job_id),
        )
        store.create(job)
        worker.notify()
        return {
            "analysis_id": job_id,
            "status": "queued",
            "visit_count": len(upload.visits),
            "visit_labels": upload.visit_labels,
            "poll_url": f"/api/v1/analyses/{job_id}",
        }

    @app.get("/api/v1/analyses", dependencies=[Depends(require_api_key)])
    def list_analyses(limit: int = 50):
        return [result_to_dict(j) for j in store.list_jobs(min(limit, 200))]

    @app.get("/api/v1/analyses/{job_id}", dependencies=[Depends(require_api_key)])
    def get_analysis(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, detail="Analysis not found.")
        return result_to_dict(job)

    @app.delete("/api/v1/analyses/{job_id}", dependencies=[Depends(require_api_key)])
    def delete_analysis(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, detail="Analysis not found.")
        if job.status in ("queued", "running"):
            worker.cancel(job_id)
            store.update(job_id, status="cancelled", completed_at=datetime.now(UTC))
        else:
            store.update(job_id, status="cancelled")
        return {"analysis_id": job_id, "status": "cancelled"}

    @app.get("/api/v1/patients", dependencies=[Depends(require_api_key)])
    def patients():
        return {"patients": store.patients()}

    @app.get("/api/v1/patients/{subject_ref}/history", dependencies=[Depends(require_api_key)])
    def patient_history(subject_ref: str):
        return [result_to_dict(j) for j in store.history(subject_ref)]

    @app.get("/api/v1/analyses/{job_id}/events", dependencies=[Depends(require_api_key)])
    async def events(job_id: str):
        import anyio

        async def stream():
            last_status, last_stage = None, None
            for _ in range(600):  # ~5 min max lifetime for the stream
                job = store.get(job_id)
                if job is None:
                    yield "event: error\ndata: job not found\n\n"
                    return
                if (job.status, job.stage) != (last_status, last_stage):
                    payload = json.dumps({"status": job.status, "stage": job.stage})
                    yield f"event: job\ndata: {payload}\n\n"
                    last_status, last_stage = job.status, job.stage
                if job.status in ("succeeded", "failed", "cancelled"):
                    return
                await anyio.sleep(0.5)
            yield "event: timeout\ndata: {}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


_START = [time.time()]
