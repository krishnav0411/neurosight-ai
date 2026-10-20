"""metrics.py — epoch-level metric accumulation for training and validation."""

from __future__ import annotations

import numpy as np
import torch


class MetricsTracker:
    """Accumulate logits/labels over an epoch; compute loss, accuracy, AUC.

    AUC needs both classes present in the epoch's labels — otherwise it is
    recorded as 0.0 (and the trainer must not model-select on it that epoch).
    """

    def __init__(self) -> None:
        self._logits: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._loss_sum = 0.0
        self._n = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float) -> None:
        self._logits.append(logits.detach().cpu().numpy())
        self._labels.append(labels.detach().cpu().numpy())
        self._loss_sum += loss * labels.size(0)
        self._n += labels.size(0)

    def compute(self) -> dict[str, float]:
        if self._n == 0:
            return {"loss": 0.0, "accuracy": 0.0, "auc": 0.0}
        logits = np.concatenate(self._logits)
        labels = np.concatenate(self._labels)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        preds = logits.argmax(axis=1)
        metrics = {
            "loss": self._loss_sum / self._n,
            "accuracy": float((preds == labels).mean()),
        }
        if len(np.unique(labels)) > 1 and probs.shape[1] == 2:
            from sklearn.metrics import roc_auc_score

            try:
                metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))
            except ValueError:  # degenerate probabilities
                metrics["auc"] = 0.0
        else:
            metrics["auc"] = 0.0
        return metrics

    def reset(self) -> None:
        self._logits.clear()
        self._labels.clear()
        self._loss_sum = 0.0
        self._n = 0


__all__ = ["MetricsTracker"]
