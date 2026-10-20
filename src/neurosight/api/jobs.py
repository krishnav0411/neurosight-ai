"""jobs.py — background analysis worker (plan doc 05 §5).

FIFO claims from the store, cooperative cancellation between visits, staged
status updates. Concurrency is 1 on GPU by default (config).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from neurosight.api.store import JobStore
from neurosight.config import ServingConfig
from neurosight.inference.predictor import NeuroSightPredictor, StudyCancelledError
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisWorker:
    """Single-threaded FIFO worker; poll interval keeps SQLite contention low."""

    def __init__(self, store: JobStore, predictor: NeuroSightPredictor, cfg: ServingConfig):
        self.store = store
        self.predictor = predictor
        self.cfg = cfg
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None
        self._cancelled: set[str] = set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="analysis-worker", daemon=True)
        self._thread.start()
        logger.info("analysis worker started (concurrency=%s)", self.cfg.worker_concurrency)

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()

    def notify(self) -> None:
        """Wake the worker immediately (called after a job is enqueued)."""
        self._wakeup.set()

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    def _loop(self) -> None:
        last_sweep = 0.0
        while not self._stop.is_set():
            try:
                job = self.store.claim_next_queued()
                if job is None:
                    self._wakeup.wait(timeout=1.0)
                    self._wakeup.clear()
                else:
                    self._execute(job.id)
                if time.time() - last_sweep > 86400:
                    self.store.purge_expired_uploads(self.cfg.uploads_dir, self.cfg.retention_days)
                    last_sweep = time.time()
            except Exception as exc:
                logger.exception("worker loop error: %s", exc)
                time.sleep(1.0)

    def _execute(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return

        # Visit file groups were persisted at upload time by the route via
        # upload manifest; the job row carries the upload dir + labels.
        try:
            manifest_path = Path(job.upload_dir) / "visit_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            visits: list[list[str]] = manifest["visits"]
            labels: list[str] = manifest["visit_labels"]
        except Exception as exc:
            self.store.update(
                job_id,
                status="failed",
                completed_at=datetime.now(UTC),
                error_message=f"upload manifest unreadable: {exc}",
            )
            return

        try:
            if job_id in self._cancelled:
                raise CancelledError()
            self.store.update(job_id, stage="inference")
            result = self.predictor.predict_study(
                visits=visits,
                demographics=json.loads(job.demographics_json) if job.demographics_json else None,
                visit_labels=labels,
                include_gradcam=job.include_gradcam,
                # prompt cancellation: checked between visits INSIDE inference
                should_cancel=lambda: job_id in self._cancelled,
            )
            if job_id in self._cancelled:
                raise CancelledError()
            self.store.update(
                job_id,
                status="succeeded",
                stage=None,
                result_json=json.dumps(result),
                model_version=result["model_version"],
                completed_at=datetime.now(UTC),
            )
            logger.info("job %s succeeded (%s)", job_id, result["model_version"])
        except (CancelledError, StudyCancelledError):
            self._cancelled.discard(job_id)
            self.store.update(job_id, status="cancelled", completed_at=datetime.now(UTC))
            logger.info("job %s cancelled", job_id)
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            self.store.update(
                job_id,
                status="failed",
                error_message=f"analysis failed: {exc}",
                completed_at=datetime.now(UTC),
            )


class CancelledError(Exception):
    """Internal: cooperative cancellation signal."""
