"""Tests for the FastAPI serving plane (plan T6.2, T6.3).

These run a REAL predictor over real (tiny) NIfTI uploads — no mocks. The
predictor uses a freshly trained-on-random-weights checkpoint so the flow is
fully exercised: upload -> validate -> queue -> worker -> result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurosight.api.main import create_app
from neurosight.models.registry import build_model
from neurosight.utils.io import save_checkpoint

DATA_YAML = (
    'data:\n  csv_path: "D:/x.csv"\n  raw_paths: []\n'
    '  processed_dir: "D:/p"\n  uploads_dir: "D:/u"\n'
)
YAML = {
    "data.yaml": DATA_YAML,
    "model.yaml": 'model:\n  model_key: "3_cnn_lstm"\n  input_mode: "mri_only"\n',
    "training.yaml": "training:\n  epochs: 1\n",
    "serving.yaml": None,  # written per-test with tmp paths
}


@pytest.fixture()
def api(tmp_path: Path, monkeypatch) -> TestClient:
    """App wired to a tmp SQLite/uploads dir and a real random-weight model."""
    monkeypatch.setenv("NEUROSIGHT_API_KEY", "test-key-123")

    serving = tmp_path / "configs" / "serving.yaml"
    serving.parent.mkdir(parents=True, exist_ok=True)
    # forward slashes: backslashes break double-quoted YAML scalars
    up = (tmp_path / "uploads").as_posix()
    md = (tmp_path / "models").as_posix()
    db = (tmp_path / "jobs.db").as_posix()
    serving.write_text(
        "serving:\n"
        '  host: "127.0.0.1"\n'
        "  port: 8000\n"
        '  api_key_env: "NEUROSIGHT_API_KEY"\n'
        f'  uploads_dir: "{up}"\n'
        f'  models_dir: "{md}"\n'
        f'  db_path: "{db}"\n',
        encoding="utf-8",
    )
    for name, content in YAML.items():
        if content:
            (tmp_path / "configs" / name).write_text(content, encoding="utf-8")

    # a real (random-weight) checkpoint + manifest
    from neurosight.config import ModelConfig

    model = build_model("3_cnn_lstm", ModelConfig())
    ckpt = tmp_path / "models" / "best_3_cnn_lstm_final.pth"
    save_checkpoint(ckpt, model)
    (tmp_path / "models" / "manifest.json").write_text(
        json.dumps(
            {
                "active_model": "3_cnn_lstm",
                "input_mode": "mri_only",
                "checkpoint": str(ckpt),
                "temperature": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    app = create_app(str(tmp_path / "configs"))
    with TestClient(app) as client:
        client.headers.update({"X-API-Key": "test-key-123"})
        yield client


def _tiny_nifti_bytes(seed: int) -> bytes:
    import io

    import nibabel as nib

    vol = np.random.RandomState(seed).rand(32, 32, 48).astype(np.float32) * 100
    buf = io.BytesIO()
    holder = nib.FileHolder(fileobj=buf)
    img = nib.Nifti1Image(vol, np.eye(4), file_map={"image": holder})
    img.to_file_map()
    return buf.getvalue()


def test_cors_env_override(tmp_path: Path, monkeypatch) -> None:
    """NEUROSIGHT_CORS_ORIGINS replaces the YAML origins (hosted deploys)."""
    from fastapi.testclient import TestClient

    from neurosight.api.main import create_app

    monkeypatch.setenv("NEUROSIGHT_API_KEY", "k")
    monkeypatch.setenv("NEUROSIGHT_CORS_ORIGINS", "https://demo.example.app")
    cfg_dir = tmp_path / "cfg2"
    cfg_dir.mkdir()
    for name, content in YAML.items():
        if content:
            (cfg_dir / name).write_text(content, encoding="utf-8")
    up, md, db = (
        (tmp_path / "u2").as_posix(),
        (tmp_path / "models").as_posix(),
        (tmp_path / "j2.db").as_posix(),
    )
    (cfg_dir / "serving.yaml").write_text(
        "serving:\n"
        '  api_key_env: "NEUROSIGHT_API_KEY"\n'
        f'  uploads_dir: "{up}"\n'
        f'  models_dir: "{md}"\n'
        f'  db_path: "{db}"\n',
        encoding="utf-8",
    )
    # the app probes a real model at startup — provide a random-weight member
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    from neurosight.config import ModelConfig
    from neurosight.models.registry import build_model
    from neurosight.utils.io import save_checkpoint

    ckpt = tmp_path / "models" / "best_7_dual_attn_final_7_dual_attn_mri_only.pth"
    save_checkpoint(ckpt, build_model("7_dual_attn", ModelConfig()))
    (tmp_path / "models" / "manifest.json").write_text(
        json.dumps(
            {
                "active_model": "7_dual_attn",
                "input_mode": "mri_only",
                "checkpoint": str(ckpt),
                "temperature": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    app = create_app(str(cfg_dir))
    with TestClient(app) as client:
        r = client.options(
            "/api/v1/patients",
            headers={"Origin": "https://demo.example.app", "Access-Control-Request-Method": "GET"},
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "https://demo.example.app"
        other = client.options(
            "/api/v1/patients",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        assert other.headers.get("access-control-allow-origin") != "https://evil.example"


def test_health_no_auth(api: TestClient) -> None:
    # bare client: the fixture's X-API-Key header must not mask an auth regression
    bare = TestClient(api.app, raise_server_exceptions=False)
    r = bare.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_card_endpoint(api: TestClient) -> None:
    r = api.get("/api/v1/model-card")
    assert r.status_code == 200
    assert "content" in r.json()


def test_api_key_required(api: TestClient) -> None:
    # fixture client has the right key; a raw client must be rejected
    bare = TestClient(api.app, raise_server_exceptions=False)
    r = bare.get("/api/v1/analyses")
    assert r.status_code == 401
    r2 = bare.get("/api/v1/analyses", headers={"X-API-Key": "wrong"})
    assert r2.status_code == 401


def test_upload_and_full_job_lifecycle(api: TestClient, tmp_path: Path) -> None:
    files = [
        ("files", ("OAS2_9999_MR1/RAW/mpr-1.nii", _tiny_nifti_bytes(1))),
        ("files", ("OAS2_9999_MR2/RAW/mpr-1.nii", _tiny_nifti_bytes(2))),
    ]
    r = api.post(
        "/api/v1/analyses",
        files=files,
        data={
            "subject_ref": "test-subject",
            "visit_labels_json": '["OAS2_9999_MR1", "OAS2_9999_MR2"]',
        },
    )
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["visit_count"] == 2
    assert set(job["visit_labels"]) == {"OAS2_9999_MR1", "OAS2_9999_MR2"}

    # wait for the worker (real CNN on CPU: seconds)
    deadline = time.time() + 180
    payload = None
    while time.time() < deadline:
        payload = api.get(f"/api/v1/analyses/{job['analysis_id']}").json()
        if payload["status"] in ("succeeded", "failed"):
            break
        time.sleep(1.0)
    assert payload is not None and payload["status"] == "succeeded", payload
    result = payload["result"]
    assert result["predicted_class"] in (0, 1)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3
    assert result["model_version"].startswith("3_cnn_lstm@")
    assert result["caveats"] == ["research_only", "small_test_set"]


def test_upload_img_hdr_pair_succeeds(api: TestClient, tmp_path: Path) -> None:
    """Regression: an .img/.hdr pair must survive upload validation — the old
    code randomized each filename BEFORE pairing, so real Analyze pairs were
    always rejected as 'incomplete pair' (the bug in the first live demo)."""

    import nibabel as nib

    vol = np.random.RandomState(9).rand(32, 32, 48).astype(np.float32) * 100
    analyze_img = nib.AnalyzeImage(vol, np.eye(4))
    tmp_pair = tmp_path / "real.img"
    nib.save(analyze_img, tmp_pair)  # writes real.img + real.hdr (Analyze pair)
    img = (tmp_path / "real.img").read_bytes()
    hdr = (tmp_path / "real.hdr").read_bytes()
    r = api.post(
        "/api/v1/analyses",
        files=[
            ("files", ("OAS2_7777_MR1/RAW/mpr-1.nifti.img", img)),
            ("files", ("OAS2_7777_MR1/RAW/mpr-1.nifti.hdr", hdr)),
        ],
        data={
            "visit_labels_json": '["OAS2_7777_MR1", "OAS2_7777_MR1"]',
            "include_gradcam": "false",
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["visit_count"] == 1


def test_upload_unpaired_img_rejected(api: TestClient) -> None:
    r = api.post(
        "/api/v1/analyses",
        files=[("files", ("OAS2_0001_MR1/RAW/mpr-1.nifti.img", b"garbage"))],
    )
    assert r.status_code == 422
    assert any("incomplete pair" in p for p in r.json()["detail"]["problems"])


def test_upload_bad_extension_rejected(api: TestClient) -> None:
    r = api.post("/api/v1/analyses", files=[("files", ("scan.txt", b"hello"))])
    assert r.status_code == 422
    assert any("unsupported extension" in p for p in r.json()["detail"]["problems"])


def test_patient_history(api: TestClient) -> None:
    files = [("files", ("OAS2_8888_MR1/RAW/mpr-1.nii", _tiny_nifti_bytes(3)))]
    r = api.post(
        "/api/v1/analyses",
        files=files,
        data={
            "subject_ref": "hist-subject",
            "include_gradcam": "false",
            "visit_labels_json": '["OAS2_8888_MR1"]',
        },
    )
    assert r.status_code == 201
    deadline = time.time() + 180
    while time.time() < deadline:
        state = api.get(f"/api/v1/analyses/{r.json()['analysis_id']}").json()["status"]
        if state in ("succeeded", "failed"):
            break
        time.sleep(1.0)
    history = api.get("/api/v1/patients/hist-subject/history").json()
    assert len(history) == 1
    patients = api.get("/api/v1/patients").json()["patients"]
    assert "hist-subject" in patients


def test_delete_unknown_job_404(api: TestClient) -> None:
    assert api.delete("/api/v1/analyses/deadbeef").status_code == 404


# ── rate limiting: only analysis creation is capped (CodeRabbit) ─────────────


def test_rate_limit_caps_only_analysis_creation(api: TestClient) -> None:
    """POST /api/v1/analyses is capped at 10/min; read-only polling endpoints
    (dashboard job status + SSE) must never 429."""
    statuses = []
    for _i in range(11):
        r = api.post(
            "/api/v1/analyses",
            files=[("files", ("scan.txt", b"not an mri"))],  # fast 422, no worker run
        )
        statuses.append(r.status_code)
    assert all(s == 422 for s in statuses[:10]), statuses
    assert statuses[-1] == 429, "the 11th creation inside the window must be limited"

    # reads are never limited, even with the creation window exhausted
    for _ in range(12):
        assert api.get("/api/v1/analyses").status_code == 200
    assert api.get("/health").status_code == 200


def test_rate_bucket_evicts_empty_windows(api: TestClient) -> None:
    """The per-client bucket must not grow without bound: fully drained windows
    are dropped by the next request."""
    r = api.post("/api/v1/analyses", files=[("files", ("scan.txt", b"not an mri"))])
    assert r.status_code == 422
    bucket = api.app.state.rate_bucket  # type: ignore[attr-defined]
    assert len(bucket) == 1
    next(iter(bucket.values())).clear()  # simulate the 60 s window draining
    api.get("/health")  # any request sweeps empty windows
    assert len(bucket) == 0


# ── malformed JSON form fields must 422, not 500 (CodeRabbit) ────────────────


def test_malformed_visit_labels_json_422(api: TestClient) -> None:
    r = api.post(
        "/api/v1/analyses",
        files=[("files", ("OAS2_0001_MR1/RAW/mpr-1.nii", _tiny_nifti_bytes(51)))],
        data={"visit_labels_json": "{not json"},
    )
    assert r.status_code == 422, r.text
    assert "malformed JSON form field" in r.json()["detail"]


def test_malformed_demographics_json_422(api: TestClient) -> None:
    r = api.post(
        "/api/v1/analyses",
        files=[("files", ("OAS2_0001_MR1/RAW/mpr-1.nii", _tiny_nifti_bytes(52)))],
        data={"demographics": "[not an object"},
    )
    assert r.status_code == 422, r.text
    assert "malformed JSON form field" in r.json()["detail"]


# ── rejected uploads must not linger on disk (CodeRabbit, retention) ─────────


def test_rejected_upload_files_are_deleted(tmp_path: Path) -> None:
    """A post-write rejection leaves NO upload directory behind: no AnalysisJob
    row is created, so the retention sweeper could never purge those scans."""
    from neurosight.api.validation import UploadValidationError, validate_and_persist
    from neurosight.config import ServingConfig

    cfg = ServingConfig(uploads_dir=tmp_path / "uploads")
    files = [("scan.nii", b"garbage that nibabel cannot open")]
    with pytest.raises(UploadValidationError):
        validate_and_persist(files, cfg, "rejected-job")
    assert not (tmp_path / "uploads" / "rejected-job").exists()


def test_visit_cap_rejection_removes_persisted_files(tmp_path: Path) -> None:
    from neurosight.api.validation import UploadValidationError, validate_and_persist
    from neurosight.config import ServingConfig

    cfg = ServingConfig(uploads_dir=tmp_path / "uploads", max_visits=2)
    files = [(f"OAS2_9{v:03d}_MR1.nii", _tiny_nifti_bytes(60 + v)) for v in range(3)]
    with pytest.raises(UploadValidationError, match="exceed"):
        validate_and_persist(files, cfg, "capped-job")
    assert not (tmp_path / "uploads" / "capped-job").exists()


# ── predictor: cancellation between visits + honest Grad-CAM loop ────────────


def _make_predictor(tmp_path: Path):
    from neurosight.config import ModelConfig
    from neurosight.inference.predictor import NeuroSightPredictor
    from neurosight.models.registry import build_model
    from neurosight.utils.io import save_checkpoint

    model = build_model("3_cnn_lstm", ModelConfig())
    ckpt_dir = tmp_path / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "best_3_cnn_lstm_final.pth"
    save_checkpoint(ckpt, model)
    manifest = ckpt_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "active_model": "3_cnn_lstm",
                "input_mode": "mri_only",
                "checkpoint": str(ckpt),
                "temperature": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return NeuroSightPredictor.from_manifest_file(
        manifest, models_dir=ckpt_dir, num_slices=32, img_size=128
    )


def _write_tiny_nifti(path: Path, seed: int) -> str:
    import nibabel as nib

    vol = np.random.RandomState(seed).rand(32, 32, 48).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(vol, np.eye(4)), path)
    return str(path)


def test_predict_study_gradcam_iterates_visits_not_labels(tmp_path: Path) -> None:
    """Extra caller-supplied labels must not IndexError the Grad-CAM loop, and
    missing labels fall back to MR<n> — the loop indexes visits defensively."""
    from neurosight.inference.predictor import NeuroSightPredictor  # noqa: F401

    predictor = _make_predictor(tmp_path)
    scan = _write_tiny_nifti(tmp_path / "a.nii", 61)
    result = predictor.predict_study(
        [[scan]], visit_labels=["MR1", "MR2", "MR3"], include_gradcam=True
    )
    assert len(result["gradcam"]) == 1  # one entry PER VISIT, not per label
    assert result["gradcam"][0]["visit"] == "MR1"


def test_predict_study_raises_study_cancelled_between_visits(tmp_path: Path) -> None:
    from neurosight.inference.predictor import StudyCancelledError

    predictor = _make_predictor(tmp_path)
    v1 = _write_tiny_nifti(tmp_path / "a.nii", 62)
    v2 = _write_tiny_nifti(tmp_path / "b.nii", 63)
    with pytest.raises(StudyCancelledError):
        predictor.predict_study([[v1], [v2]], should_cancel=lambda: True)
    # a never-cancelled callback completes normally
    result = predictor.predict_study([[v1], [v2]], should_cancel=lambda: False)
    assert result["visit_count"] == 2


def test_worker_maps_mid_inference_cancellation_to_cancelled(tmp_path: Path) -> None:
    """predict_study receives a cancellation callback; StudyCancelled raised
    mid-inference lands as job status 'cancelled', not 'failed'."""
    from neurosight.api.jobs import AnalysisWorker
    from neurosight.api.store import AnalysisJob, JobStore
    from neurosight.config import ServingConfig
    from neurosight.inference.predictor import StudyCancelledError

    cfg = ServingConfig(
        uploads_dir=tmp_path / "uploads", models_dir=tmp_path / "models", db_path=tmp_path / "db"
    )
    store = JobStore(cfg)
    job_id = "canceljob"
    upload_dir = tmp_path / "uploads" / job_id
    upload_dir.mkdir(parents=True)
    (upload_dir / "visit_manifest.json").write_text(
        json.dumps({"visits": [["a.nii"]], "visit_labels": ["MR1"]}), encoding="utf-8"
    )
    store.create(
        AnalysisJob(id=job_id, input_mode="mri_only", visit_count=1, upload_dir=str(upload_dir))
    )

    class StubPredictor:
        def predict_study(self, **kwargs):
            callback = kwargs.get("should_cancel")
            assert callback is not None, "worker must pass a cancellation callback"
            # simulate the DELETE arriving mid-inference (worker pre-checks
            # cannot see it yet): the callback must reflect the cancel set
            worker.cancel(job_id)
            assert callback()
            raise StudyCancelledError("cancelled between visits")

    worker = AnalysisWorker(store, StubPredictor(), cfg)  # type: ignore[arg-type]
    worker._execute(job_id)
    job = store.get(job_id)
    assert job is not None and job.status == "cancelled", "mid-inference cancel must not fail"


def test_create_app_with_repo_manifest_loads_e5_ensemble() -> None:
    """Regression test: create_app() with actual repo configs and models/manifest.json
    must load all 5 ensemble members and report neurosight-e5 version."""
    from neurosight.api.main import _STATE

    repo_root = Path(__file__).resolve().parent.parent
    manifest_file = repo_root / "models" / "manifest.json"
    if not manifest_file.exists():
        pytest.skip("models/manifest.json not present in repository")
    app = create_app(str(repo_root / "configs"))
    assert app is not None
    predictor = _STATE.get("predictor")
    assert predictor is not None, "predictor must be stored in app state"
    assert len(predictor.models) == 5, f"expected 5 ensemble models, got {len(predictor.models)}"
    assert predictor.model_version.startswith("neurosight-e5@")
    assert abs(predictor.temperature - 0.657303) < 1e-3
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_version"].startswith("neurosight-e5@")


def test_e5_synthetic_two_visit_inference_and_gradcam(tmp_path: Path, monkeypatch) -> None:
    """Full end-to-end integration test with real E5 checkpoints:
    1. Loads the 5-member ensemble from models/manifest.json.
    2. Runs synthetic two-visit NIfTI study with Grad-CAM.
    3. Validates model version, calibrated probabilities, longitudinal trend,
       and Grad-CAM base64 slice payloads for both visits.
    """
    monkeypatch.setenv("NEUROSIGHT_API_KEY", "test-e5-key")
    repo_root = Path(__file__).resolve().parent.parent
    manifest_file = repo_root / "models" / "manifest.json"
    if not manifest_file.exists():
        pytest.skip("models/manifest.json not present in repository")

    # Set up isolated configs directory pointing to real models
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    up = (tmp_path / "uploads").as_posix()
    md = (repo_root / "models").as_posix()
    db = (tmp_path / "jobs.db").as_posix()
    (cfg_dir / "serving.yaml").write_text(
        "serving:\n"
        '  host: "127.0.0.1"\n'
        "  port: 8000\n"
        '  api_key_env: "NEUROSIGHT_API_KEY"\n'
        f'  uploads_dir: "{up}"\n'
        f'  models_dir: "{md}"\n'
        f'  db_path: "{db}"\n',
        encoding="utf-8",
    )
    for name, content in YAML.items():
        if content:
            (cfg_dir / name).write_text(content, encoding="utf-8")

    app = create_app(str(cfg_dir))
    with TestClient(app) as client:
        client.headers.update({"X-API-Key": "test-e5-key"})

        # Verify /health endpoint
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_version"].startswith("neurosight-e5@")

        # Create two synthetic NIfTI files
        v1_bytes = _tiny_nifti_bytes(42)
        v2_bytes = _tiny_nifti_bytes(43)

        res = client.post(
            "/api/v1/analyses",
            files=[
                ("files", ("sub-01_ses-01_mpr.nii", v1_bytes, "application/octet-stream")),
                ("files", ("sub-01_ses-02_mpr.nii", v2_bytes, "application/octet-stream")),
            ],
            data={
                "visit_labels_json": json.dumps(["MR1", "MR2"]),
                "include_gradcam": "true",
            },
        )
        assert res.status_code == 201
        job_id = res.json()["analysis_id"]

        # Poll worker until job succeeds
        deadline = time.time() + 30.0
        record = None
        while time.time() < deadline:
            record = client.get(f"/api/v1/analyses/{job_id}").json()
            if record["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.3)

        assert record is not None
        assert record["status"] == "succeeded", f"job failed: {record.get('error')}"
        result = record["result"]
        assert result["model_version"].startswith("neurosight-e5@")
        assert result["visit_count"] == 2
        assert "Non-Demented" in result["probabilities"]
        assert "Demented" in result["probabilities"]
        # Probabilities must sum to 1.0 within floating point tolerance
        p_sum = sum(result["probabilities"].values())
        assert abs(p_sum - 1.0) < 1e-5
        # Longitudinal per-visit probabilities
        assert len(result["longitudinal"]["per_visit_probability"]) == 2
        # Grad-CAM checks: 2 visits, key slices 10, 16, 24
        assert "gradcam" in result
        assert len(result["gradcam"]) == 2
        assert result["gradcam"][0]["visit"] == "MR1"
        assert result["gradcam"][1]["visit"] == "MR2"
        for g_visit in result["gradcam"]:
            assert set(g_visit["slices"]) == {"10", "16", "24"}
            for s in ("10", "16", "24"):
                assert len(g_visit["images"][s]) > 100  # valid base64 PNG string


def test_upload_exceeding_max_size_returns_413_and_cleans_staging(
    tmp_path: Path, monkeypatch
) -> None:
    """Uploads exceeding max_upload_mb are rejected with 413 and clean up staging."""
    monkeypatch.setenv("NEUROSIGHT_API_KEY", "test-key-123")
    serving = tmp_path / "configs" / "serving.yaml"
    serving.parent.mkdir(parents=True, exist_ok=True)
    up = (tmp_path / "uploads").as_posix()
    md = (tmp_path / "models").as_posix()
    db = (tmp_path / "jobs.db").as_posix()
    serving.write_text(
        "serving:\n"
        '  host: "127.0.0.1"\n'
        "  port: 8000\n"
        '  api_key_env: "NEUROSIGHT_API_KEY"\n'
        "  max_upload_mb: 1\n"
        f'  uploads_dir: "{up}"\n'
        f'  models_dir: "{md}"\n'
        f'  db_path: "{db}"\n',
        encoding="utf-8",
    )
    for name, file_content in YAML.items():
        if file_content:
            (tmp_path / "configs" / name).write_text(file_content, encoding="utf-8")

    from neurosight.config import ModelConfig
    from neurosight.models.registry import build_model

    ckpt_dir = tmp_path / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    m = build_model("3_cnn_lstm", ModelConfig())
    save_checkpoint(ckpt_dir / "best_3_cnn_lstm_final_3_cnn_lstm_mri_only.pth", m)
    (ckpt_dir / "manifest.json").write_text(
        json.dumps(
            {
                "active_model": "3_cnn_lstm",
                "checkpoint": "best_3_cnn_lstm_final_3_cnn_lstm_mri_only.pth",
                "temperature": 1.0,
            }
        ),
        encoding="utf-8",
    )

    app = create_app(config_dir=str(tmp_path / "configs"))
    client = TestClient(app, headers={"X-API-Key": "test-key-123"})

    # Send a 1.2 MB file (> 1 MB limit)
    oversized = b"0" * (1024 * 1024 + 200 * 1024)
    r = client.post("/api/v1/analyses", files=[("files", ("scan.nii", oversized))])
    assert r.status_code == 413
    assert "exceeds maximum allowed size" in r.json()["detail"]
    # Verify staging folders are cleaned up
    assert list((tmp_path / "uploads").glob(".tmp_staging_*")) == []


def test_upload_multi_file_streaming_success(tmp_path: Path, monkeypatch) -> None:
    """Multi-file streaming uploads stage directly to disk and clean up staging."""
    monkeypatch.setenv("NEUROSIGHT_API_KEY", "test-key-123")
    serving = tmp_path / "configs" / "serving.yaml"
    serving.parent.mkdir(parents=True, exist_ok=True)
    up = (tmp_path / "uploads").as_posix()
    md = (tmp_path / "models").as_posix()
    db = (tmp_path / "jobs.db").as_posix()
    serving.write_text(
        "serving:\n"
        '  host: "127.0.0.1"\n'
        "  port: 8000\n"
        '  api_key_env: "NEUROSIGHT_API_KEY"\n'
        "  max_upload_mb: 20\n"
        f'  uploads_dir: "{up}"\n'
        f'  models_dir: "{md}"\n'
        f'  db_path: "{db}"\n',
        encoding="utf-8",
    )
    for name, file_content in YAML.items():
        if file_content:
            (tmp_path / "configs" / name).write_text(file_content, encoding="utf-8")

    from neurosight.config import ModelConfig
    from neurosight.models.registry import build_model

    ckpt_dir = tmp_path / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    m = build_model("3_cnn_lstm", ModelConfig())
    save_checkpoint(ckpt_dir / "best_3_cnn_lstm_final_3_cnn_lstm_mri_only.pth", m)
    (ckpt_dir / "manifest.json").write_text(
        json.dumps(
            {
                "active_model": "3_cnn_lstm",
                "checkpoint": "best_3_cnn_lstm_final_3_cnn_lstm_mri_only.pth",
                "temperature": 1.0,
            }
        ),
        encoding="utf-8",
    )

    app = create_app(config_dir=str(tmp_path / "configs"))
    client = TestClient(app, headers={"X-API-Key": "test-key-123"})

    files = [
        ("files", ("scan_v1.nii", _tiny_nifti_bytes(101))),
        ("files", ("scan_v2.nii", _tiny_nifti_bytes(102))),
    ]
    data = {
        "visit_labels_json": json.dumps(["MR1", "MR2"]),
        "subject_ref": "SUBJ_STREAM_01",
    }
    r = client.post("/api/v1/analyses", files=files, data=data)
    assert r.status_code == 201
    resp = r.json()
    job_id = resp["analysis_id"]
    assert resp["visit_count"] == 2

    # Verify staging directory is cleaned up and persisted files exist
    assert list((tmp_path / "uploads").glob(".tmp_staging_*")) == []
    job_dir = tmp_path / "uploads" / job_id
    assert job_dir.exists()
    assert len(list(job_dir.glob("*.nii"))) == 2
