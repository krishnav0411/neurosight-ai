"""device.py — compute-device selection with explicit logging."""

from __future__ import annotations

import torch

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Prefer CUDA, fall back to CPU; always log which is used."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "Using CUDA device: %s (%.1f GB VRAM)", props.name, props.total_memory / 1024**3
        )
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available — using CPU")
    return device


__all__ = ["get_device"]
