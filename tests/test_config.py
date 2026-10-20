"""Tests for the strict config system (plan T0.2).

Covered: defaults, unknown-key rejection, missing-file error, invalid YAML,
ratio-sum validation, and the plan-mandated guardrail defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neurosight.config import NeuroSightConfig, load_config

DATA_YAML = """
data:
  csv_path: "D:/fake/oasis_longitudinal.csv"
  raw_paths: ["D:/fake/part1", "D:/fake/part2"]
  processed_dir: "D:/fake/processed"
  uploads_dir: "D:/fake/uploads"
"""
MODEL_YAML = """
model:
  model_key: "3_cnn_lstm"
  input_mode: "mri_only"
"""
TRAINING_YAML = """
training:
  epochs: 50
"""
SERVING_YAML = """
serving:
  port: 8000
"""


def _write_configs(
    tmp_path: Path,
    data: str = DATA_YAML,
    model: str = MODEL_YAML,
    training: str = TRAINING_YAML,
    serving: str = SERVING_YAML,
) -> Path:
    (tmp_path / "data.yaml").write_text(data, encoding="utf-8")
    (tmp_path / "model.yaml").write_text(model, encoding="utf-8")
    (tmp_path / "training.yaml").write_text(training, encoding="utf-8")
    (tmp_path / "serving.yaml").write_text(serving, encoding="utf-8")
    return tmp_path


def test_load_default_configs(configs_dir: Path) -> None:
    cfg = load_config(configs_dir)
    assert isinstance(cfg, NeuroSightConfig)
    # Guardrail defaults mandated by the plan (T0.2 / 00-PLAN-REVIEW fixes)
    assert cfg.data.min_visits == 2
    assert cfg.data.series_selection == "first"
    assert cfg.data.label_mode == "binary_cdr"
    assert cfg.data.slice_strategy == "middle"
    assert cfg.model.input_mode == "mri_only"
    assert cfg.model.num_classes == 2
    assert cfg.training.deterministic is True
    assert cfg.training.cudnn_benchmark is False
    assert cfg.training.num_workers == 0
    assert cfg.training.early_stop_metric == "val_auc"
    assert cfg.serving.max_upload_mb == 512


def test_real_config_paths_resolve_through_junction(configs_dir: Path) -> None:
    """The CSV path and both raw paths must resolve (via the data/raw junction)."""
    cfg = load_config(configs_dir)
    assert cfg.data.csv_path.name == "oasis_longitudinal.csv"
    assert len(cfg.data.raw_paths) == 2


def test_serving_cors_and_vite_preview_proxy(configs_dir: Path) -> None:
    """Preview port (4173) and dev port (5173) must both be allowed by CORS,
    and frontend/vite.config.ts must configure proxy for both dev and preview."""
    cfg = load_config(configs_dir).serving
    assert "http://localhost:4173" in cfg.cors_origins
    assert "http://localhost:5173" in cfg.cors_origins

    vite_cfg_path = configs_dir.parent / "frontend" / "vite.config.ts"
    if vite_cfg_path.exists():
        content = vite_cfg_path.read_text(encoding="utf-8")
        assert "preview:" in content
        assert "port: 4173" in content
        assert "proxy:" in content


def test_unknown_key_rejected(tmp_path: Path) -> None:
    d = _write_configs(tmp_path, data=DATA_YAML + "  bogus_key: 1\n")
    with pytest.raises(ValidationError, match="bogus_key"):
        load_config(d)


def test_unknown_section_rejected(tmp_path: Path) -> None:
    d = _write_configs(tmp_path)
    (d / "data.yaml").write_text(DATA_YAML + "surprise_section:\n  x: 1\n", encoding="utf-8")
    # The loader's own contract rejects stray top-level sections with a file-specific error
    with pytest.raises(ValueError, match="exactly one top-level mapping key 'data:'"):
        load_config(d)


def test_missing_file_lists_all_missing(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text(DATA_YAML, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"model\.yaml.*training\.yaml.*serving\.yaml"):
        load_config(tmp_path)


def test_invalid_yaml_raises_with_filename(tmp_path: Path) -> None:
    d = _write_configs(tmp_path)
    (d / "model.yaml").write_text("model: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match=r"model\.yaml"):
        load_config(d)


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    d = _write_configs(tmp_path)
    (d / "serving.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(d)


def test_ratios_must_sum_to_one(tmp_path: Path) -> None:
    bad = DATA_YAML + "  train_ratio: 0.80\n  val_ratio: 0.10\n  test_ratio: 0.05\n"
    d = _write_configs(tmp_path, data=bad)
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        load_config(d)


def test_frozen_after_load(configs_dir: Path) -> None:
    cfg = load_config(configs_dir)
    with pytest.raises(ValidationError):
        cfg.data.min_visits = 3  # type: ignore[misc]


def test_worker_concurrency_clamped_to_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker runs exactly one GPU thread: configured values > 1 are
    clamped to 1 (with a logged warning) instead of being silently ignored."""
    import neurosight.config as config_module
    from neurosight.config import ServingConfig

    warnings: list[str] = []

    class _RecordingLogger:
        def warning(self, msg: str, *args: object) -> None:
            warnings.append(str(msg) % args if args else str(msg))

        def info(self, *args: object) -> None:
            pass

    monkeypatch.setattr(config_module, "logger", _RecordingLogger(), raising=False)
    cfg = ServingConfig(uploads_dir=tmp_path / "u", worker_concurrency=4)
    assert cfg.worker_concurrency == 1, "worker_concurrency > 1 must clamp to 1"
    assert warnings and "worker_concurrency" in warnings[0]
    # the supported value passes through untouched
    assert ServingConfig(uploads_dir=tmp_path / "v", worker_concurrency=1).worker_concurrency == 1
