"""io.py — filesystem helpers and full-training-state checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (parents included) if missing; return it as a Path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save full training state so a run can be resumed or evaluated later."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    torch.save(payload, resolved)
    logger.debug("Checkpoint saved: %s (epoch=%s)", resolved, epoch)
    return resolved


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`; restore into
    ``model``/``optimizer`` when given. Returns the full payload dict."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    payload = torch.load(resolved, map_location=map_location, weights_only=True)
    if model is not None and payload.get("model_state") is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    logger.debug("Checkpoint loaded: %s (epoch=%s)", resolved, payload.get("epoch"))
    return payload


__all__ = ["ensure_dir", "load_checkpoint", "save_checkpoint"]
