"""seed.py — reproducibility: python/numpy/torch seeds plus cudnn determinism flags."""

from __future__ import annotations

import random

import numpy as np
import torch

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch and CUDA; pin cudnn flags when ``deterministic``.

    Called once at every entry point (training scripts, evaluator, predictor)
    before any tensor op. ``deterministic=False`` trades reproducibility for
    speed and must be a visible config choice (``training.deterministic``), never
    an accident.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    logger.info("Seed set to %s (deterministic=%s)", seed, deterministic)


__all__ = ["set_seed"]
