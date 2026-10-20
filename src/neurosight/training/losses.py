"""losses.py — Focal loss and class-weight helpers (ported from the predecessor)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (F is the universal PyTorch alias)

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt).

    Down-weights easy examples so learning focuses on hard cases — the default
    gamma=2 is the literature standard. Alpha is a per-class weight tensor.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha.to(inputs.device).gather(0, targets) * focal
        return focal.mean() if self.reduction == "mean" else focal.sum()


def get_class_weights(labels: list[int] | np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1.

    Returns 1.0 for classes absent from ``labels`` so a missing class never
    produces inf weights.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.where(counts > 0, len(labels) / (num_classes * np.maximum(counts, 1)), 1.0)
    weights = weights / weights.mean()
    logger.info("class weights: %s (counts %s)", np.round(weights, 3).tolist(), counts.tolist())
    return torch.tensor(weights, dtype=torch.float32)


__all__ = ["FocalLoss", "get_class_weights"]
