"""config.py — strict Pydantic configuration schemas + YAML loader for NeuroSight AI.

Every pipeline behavior is config-driven: no module may hardcode a filesystem
path, hyperparameter, or mode. Unknown YAML keys are rejected (``extra="forbid"``)
so a renamed field can never silently fall back to a default, and schemas are
frozen so a loaded config cannot be mutated mid-run.

Usage:
    cfg = load_config("configs")   # -> NeuroSightConfig(.data, .model, .training, .serving)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from neurosight.utils.logging import get_logger

InputMode = Literal["mri_only", "demo_only", "mri_demo", "mri_clinical"]

logger = get_logger(__name__)


class _Strict(BaseModel):
    """Base for all config sections: reject unknown keys, freeze after load."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    """Dataset location, preprocessing shapes, labeling, and split policy.

    Paths are validated for existence by the data audit (M1), not here — the
    same schema must load in CI where no real data exists.
    """

    csv_path: Path
    raw_paths: list[Path]
    processed_dir: Path
    uploads_dir: Path

    min_visits: int = Field(default=2, ge=1)
    num_slices: int = Field(default=32, ge=1)
    slice_size: int = Field(default=128, ge=16)
    vol_depth: int = Field(default=64, ge=1)
    slice_strategy: Literal["middle", "uniform", "entropy"] = "middle"
    # Policy for the 1-4 mpr-N series inside each session folder (plan defect A3)
    series_selection: Literal["first", "middle", "mean"] = "first"
    # binary_cdr: CDR==0 -> 0 else 1, Group-verified, Converted split pre/post
    label_mode: Literal["binary_cdr", "per_visit", "majority"] = "binary_cdr"

    train_ratio: float = Field(default=0.70, gt=0, lt=1)
    val_ratio: float = Field(default=0.15, gt=0, lt=1)
    test_ratio: float = Field(default=0.15, gt=0, lt=1)
    random_seed: int = 42

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> DataConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train/val/test ratios must sum to 1.0 (got {total:.6f}: "
                f"{self.train_ratio}/{self.val_ratio}/{self.test_ratio})"
            )
        return self


class ModelConfig(_Strict):
    """Model selection and architecture hyperparameters.

    ``model_key`` is a registry key validated by ``neurosight.models.registry``
    at build time (M3); it is kept as a plain string here so the config layer
    does not depend on the model layer.
    """

    model_key: str = "3_cnn_lstm"
    input_mode: InputMode = "mri_only"
    lstm_hidden: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.5, ge=0, lt=1)
    d_model: int = Field(default=64, ge=1)
    nhead: int = Field(default=4, ge=1)
    num_layers: int = Field(default=1, ge=1)
    num_classes: int = Field(default=2, ge=2)
    use_conversion_head: bool = True
    freeze_backbone: bool = True


class TrainingConfig(_Strict):
    """Optimization, scheduling, determinism, and experiment tracking.

    Determinism defaults follow plan NFR-1: seeds + ``cudnn.deterministic``.
    ``num_workers=0`` is deliberate on Windows (subprocess spawn cost and the
    memory constraint learned in the predecessor project's configs).
    """

    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=4, ge=1)
    batch_size_3d: int = Field(default=2, ge=1)
    grad_accum_3d: int = Field(default=4, ge=1)

    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    patience: int = Field(default=7, ge=1)
    scheduler_patience: int = Field(default=3, ge=1)
    scheduler_factor: float = Field(default=0.5, gt=0, le=1)
    min_lr: float = Field(default=1e-6, gt=0)
    clip_norm: float = Field(default=1.0, gt=0)

    use_focal: bool = False
    label_smoothing: float = Field(default=0.0, ge=0, le=1)
    use_sampler: bool = False

    n_folds: int = Field(default=5, ge=2)
    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False
    num_workers: int = Field(default=0, ge=0)

    early_stop_metric: Literal["val_auc", "val_acc"] = "val_auc"

    checkpoint_dir: Path = Path("models")
    experiment_name: str = "neurosight_baseline"
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"


class ServingConfig(_Strict):
    """FastAPI serving plane (M6). Secrets come from the environment only."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key_env: str = "NEUROSIGHT_API_KEY"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:4173",
        "http://localhost:80",
    ]

    max_upload_mb: int = Field(default=512, ge=1)
    max_visits: int = Field(default=8, ge=1)
    max_series_per_visit: int = Field(default=4, ge=1)
    retention_days: int = Field(default=7, ge=1)
    worker_concurrency: int = Field(default=1, ge=1)
    public: bool = False  # when False, /docs is disabled

    models_dir: Path = Path("models")
    uploads_dir: Path = Path("data/uploads")
    db_path: Path = Path("data/neurosight.db")

    @field_validator("worker_concurrency")
    @classmethod
    def _single_worker_only(cls, v: int) -> int:
        """The FIFO worker runs exactly one GPU thread: configured values > 1
        are clamped to 1 (with a warning) instead of being silently ignored."""
        if v > 1:
            logger.warning(
                "worker_concurrency=%d requested but only 1 worker is supported — clamping to 1", v
            )
            return 1
        return v


class NeuroSightConfig(_Strict):
    """Root bundle returned by :func:`load_config`."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    serving: ServingConfig


_CONFIG_FILES = {
    "data": "data.yaml",
    "model": "model.yaml",
    "training": "training.yaml",
    "serving": "serving.yaml",
}


def load_config(config_dir: str | Path) -> NeuroSightConfig:
    """Load and validate the four YAML config files from ``config_dir``.

    Raises:
        FileNotFoundError: if any of the four files is missing (names all of them).
        ValueError: if a file is not parseable YAML.
        pydantic.ValidationError: on unknown keys or invalid values.
    """
    directory = Path(config_dir)
    missing = [name for name in _CONFIG_FILES.values() if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"config directory '{directory}' is missing required files: {', '.join(missing)}"
        )

    payloads: dict[str, dict] = {}
    for key, filename in _CONFIG_FILES.items():
        path = directory / filename
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {path}: {exc}") from exc
        if not isinstance(parsed, dict) or set(parsed.keys()) != {key}:
            raise ValueError(f"{path} must contain exactly one top-level mapping key '{key}:'")
        payloads[key] = parsed[key]

    try:
        return NeuroSightConfig.model_validate(payloads)
    except ValidationError:
        # Re-raised as-is: pydantic's message already names the file section,
        # the offending field, and the reason.
        raise


__all__ = [
    "DataConfig",
    "InputMode",
    "ModelConfig",
    "NeuroSightConfig",
    "ServingConfig",
    "TrainingConfig",
    "load_config",
]
