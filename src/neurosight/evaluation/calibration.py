"""calibration.py — temperature scaling + ECE (plan T5.2).

A clinical decision-support score must be *calibrated*: "87% confidence" has to
mean the model is right ~87% of the time at that confidence. Temperature scaling
(single learnable scalar on validation logits) is the standard post-hoc fix.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import torch
import torch.nn as nn

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE: sum over bins of |accuracy - confidence| weighted by bin mass.

    ``y_prob`` is the positive-class probability.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(pairwise(bins)):
        # the first bin is closed on the left: p = 0.0 is a legitimate model
        # output and must join [0, 0.1] instead of silently dropping out
        lower = (y_prob >= lo) if i == 0 else (y_prob > lo)
        in_bin = lower & (y_prob <= hi)
        if not in_bin.any():
            continue
        conf = y_prob[in_bin].mean()
        acc = y_true[in_bin].mean()
        ece += in_bin.mean() * abs(acc - conf)
    return float(ece)


MIN_TEMPERATURE = 0.05  # effective floor used by forward/apply_probs AND the fit


class TemperatureScaler(nn.Module):
    """Single temperature parameter fit on validation logits (LBFGS, NLL loss)."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    @property
    def effective_temperature(self) -> float:
        """The temperature actually applied (the raw parameter floored)."""
        return max(float(self.temperature.item()), MIN_TEMPERATURE)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=MIN_TEMPERATURE)

    @torch.no_grad()
    def apply_probs(self, probs: np.ndarray) -> np.ndarray:
        """Re-calibrate positive-class probabilities from raw ones.

        Works on the softmax ratio: p1' = (p1^(1/T)) / (p0^(1/T) + p1^(1/T)).
        """
        t = self.effective_temperature
        p1 = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1 - 1e-9)
        p0 = 1 - p1
        adj = p1 ** (1 / t) / (p0 ** (1 / t) + p1 ** (1 / t))
        return adj


def fit_temperature(
    val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 100
) -> TemperatureScaler:
    """Fit the temperature on held-out (validation) logits — never on test.

    The fit is clamped to ``MIN_TEMPERATURE`` afterwards: ``forward`` already
    treats T < 0.05 as 0.05, so the NLL is FLAT below the floor and an
    unbounded LBFGS overshoots into that flat zone leaving a raw parameter
    below zero (a real run reported T = -0.143 / -0.365 — prediction-inverting
    nonsense). Pinning the parameter to the floor keeps every reported value
    equal to the temperature that is actually applied.
    """
    scaler = TemperatureScaler()
    logits = val_logits.detach()
    labels = val_labels.detach()
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scaler.temperature.clamp_(min=MIN_TEMPERATURE)
    logger.info("temperature fitted: %.4f", scaler.effective_temperature)
    return scaler


__all__ = ["MIN_TEMPERATURE", "TemperatureScaler", "expected_calibration_error", "fit_temperature"]
