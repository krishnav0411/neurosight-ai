"""store.py — SQLite job/session store via SQLModel (plan doc 05 §2).

No table stores raw scans or patient identifiers: ``subject_ref`` is whatever
opaque string the client sends, and uploads live on disk under the job id.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine, select

from neurosight.config import ServingConfig
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisJob(SQLModel, table=True):
    id: str = Field(primary_key=True)
    subject_ref: str | None = None
    status: str = "queued"  # queued|running|succeeded|failed|cancelled
    stage: str | None = None  # preprocessing|inference|gradcam
    input_mode: str = "mri_only"
    visit_count: int = 0
    visit_labels_json: str | None = None
    demographics_json: str | None = None
    include_gradcam: bool = True
    error_message: str | None = None
    result_json: str | None = None
    model_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    upload_dir: str = ""


class PurgeLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: str
    purged_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    bytes_freed: int = 0


class JobStore:
    """Thin data-access layer over SQLite (WAL) for analysis jobs."""

    def __init__(self, cfg: ServingConfig):
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{cfg.db_path}", connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(self.engine)

    def create(self, job: AnalysisJob) -> None:
        with Session(self.engine) as session:
            session.add(job)
            session.commit()

    def get(self, job_id: str) -> AnalysisJob | None:
        with Session(self.engine) as session:
            return session.get(AnalysisJob, job_id)

    def update(self, job_id: str, **fields) -> None:
        with Session(self.engine) as session:
            job = session.get(AnalysisJob, job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            session.add(job)
            session.commit()

    def claim_next_queued(self) -> AnalysisJob | None:
        """Atomically claim the oldest queued job (FIFO)."""
        with Session(self.engine) as session:
            job = session.exec(
                select(AnalysisJob)
                .where(AnalysisJob.status == "queued")
                .order_by(AnalysisJob.created_at)  # type: ignore[arg-type]
            ).first()
            if job is None:
                return None
            job.status = "running"
            job.started_at = datetime.now(UTC)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def list_jobs(self, limit: int = 100) -> list[AnalysisJob]:
        with Session(self.engine) as session:
            jobs = session.exec(
                select(AnalysisJob)
                .order_by(AnalysisJob.created_at.desc())  # type: ignore[arg-type,attr-defined]
                .limit(limit)  # type: ignore[attr-defined]
            ).all()
            return list(jobs)

    def history(self, subject_ref: str) -> list[AnalysisJob]:
        with Session(self.engine) as session:
            jobs = session.exec(
                select(AnalysisJob)
                .where(AnalysisJob.subject_ref == subject_ref)  # type: ignore[arg-type]
                .order_by(AnalysisJob.created_at.desc())  # type: ignore[arg-type,attr-defined]
            ).all()
            return list(jobs)

    def patients(self) -> list[str]:
        with Session(self.engine) as session:
            rows = session.exec(
                select(AnalysisJob.subject_ref).distinct()  # type: ignore[arg-type]
            ).all()
            return sorted(r for r in rows if r)

    def purge_expired_uploads(self, uploads_dir: Path, retention_days: int) -> int:
        """Delete upload dirs of terminal jobs older than retention; log it."""
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        freed = 0
        with Session(self.engine) as session:
            jobs = session.exec(select(AnalysisJob)).all()
            for job in jobs:
                if job.status in ("succeeded", "failed", "cancelled") and job.upload_dir:
                    created = job.completed_at or job.created_at
                    if created is not None:
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=UTC)
                        if created < cutoff and Path(job.upload_dir).exists():
                            job_bytes = sum(
                                f.stat().st_size
                                for f in Path(job.upload_dir).rglob("*")
                                if f.is_file()
                            )
                            import shutil

                            shutil.rmtree(Path(job.upload_dir), ignore_errors=True)
                            session.add(PurgeLog(job_id=job.id, bytes_freed=job_bytes))
                            freed += job_bytes
            session.commit()
        if freed:
            logger.info("purged expired uploads: %d bytes freed", freed)
        return freed


def result_to_dict(job: AnalysisJob) -> dict:
    """Job row -> API response dict (result parsed when present)."""
    payload: dict = {
        "analysis_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "subject_ref": job.subject_ref,
        "input_mode": job.input_mode,
        "visit_count": job.visit_count,
        "visit_labels": json.loads(job.visit_labels_json) if job.visit_labels_json else None,
        "model_version": job.model_version,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error_message,
    }
    if job.result_json:
        payload["result"] = json.loads(job.result_json)
    return payload


__all__ = ["AnalysisJob", "JobStore", "PurgeLog", "result_to_dict"]
