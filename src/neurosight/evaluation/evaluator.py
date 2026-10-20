"""evaluator.py — held-out-test evaluation with uncertainty (ported from the
predecessor's evaluate.py; CIs, McNemar, and per-mode labeling are mandatory).

Every metrics dict carries ``input_mode`` and ``n_test``: no number may ever be
reported without its experimental context (plan FR-6, NFR-3).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

CLASS_LABELS = ["Non-Demented", "Demented"]


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any metric; returns (mean, lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
    scores = sorted(scores)
    lo = float(np.percentile(scores, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(scores, (1 + alpha) / 2 * 100))
    return float(np.mean(scores)), lo, hi


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> tuple[float, float]:
    """McNemar's test on paired predictions; returns (chi2, p)."""
    b = int(np.sum((pred1 == y_true) & (pred2 != y_true)))
    c = int(np.sum((pred1 != y_true) & (pred2 == y_true)))
    chi2 = (b - c) ** 2 / (b + c + 1e-10)
    p = float(1 - stats.chi2.cdf(chi2, 1))
    return chi2, p


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_params: int = 0,
    input_mode: str = "mri_only",
    ece: float | None = None,
) -> dict:
    """Full binary-classification metric suite with bootstrap CIs.

    ``y_prob`` is the calibrated Demented probability.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    m: dict = {
        "model": model_name,
        "input_mode": input_mode,
        "n_test": len(y_true),
        "parameters": int(n_params),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    _acc_mean, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy_score)
    m["accuracy_ci"] = [acc_lo, acc_hi]

    if len(np.unique(y_true)) > 1:
        m["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        m["avg_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        m["auc_roc"] = 0.0
        m["avg_precision"] = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    m["confusion_matrix"] = cm.tolist()
    m["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    m["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    m["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    m["recall"] = m["sensitivity"]
    m["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    m["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    m["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    if ece is not None:
        m["ece"] = float(ece)
    return m


def collect_predictions(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a model over a loader; returns (labels, preds, demented_probs)."""
    model.eval()
    all_labels: list[float] = []
    all_probs: list[float] = []
    with torch.no_grad():
        for scans, meta, labels, mask in loader:
            logits = model(scans.to(device), meta.to(device), mask.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    return y_true, (y_prob >= 0.5).astype(int), y_prob


def save_evaluation_report(all_metrics: dict[str, dict], save_path: str | Path) -> str:
    """Human-readable comparison table; every row labeled with its input mode."""
    lines = ["=" * 78, "  NeuroSight AI — Evaluation Report", "=" * 78, ""]
    header = (
        f"  {'Model':<24} {'Mode':<13} {'Acc':<8} {'95% CI':<16} "
        f"{'AUC':<8} {'F1':<8} {'Sens':<7} {'Spec':<7} {'ECE':<6}"
    )
    lines.append(header)
    lines.append("  " + "─" * 88)
    for name, m in sorted(
        all_metrics.items(), key=lambda kv: kv[1].get("accuracy", 0), reverse=True
    ):
        ci = m.get("accuracy_ci", [0, 0])
        lines.append(
            f"  {name:<24} {m.get('input_mode', '?'):<13} {m.get('accuracy', 0) * 100:>6.1f}%  "
            f"[{ci[0] * 100:.1f},{ci[1] * 100:.1f}] {m.get('auc_roc', 0):<8.4f} "
            f"{m.get('f1', 0):<8.4f} {m.get('sensitivity', 0):<7.3f} "
            f"{m.get('specificity', 0):<7.3f} {m.get('ece', float('nan')):<6.3f}"
        )
    lines.append("")
    for name, m in all_metrics.items():
        cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
        lines.append(
            f"  [{name}] n={m.get('n_test')} mode={m.get('input_mode')} "
            f"balanced_acc={m.get('balanced_accuracy', 0):.3f} mcc={m.get('mcc', 0):.3f} "
            f"TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}"
        )
    lines.append("")
    lines.append("  Small test sets (n≈22) give wide CIs — treat point estimates with caution.")
    report = "\n".join(lines)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    logger.info("evaluation report written: %s", path)
    return report


def save_metrics_json(all_metrics: dict[str, dict], save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")


__all__ = [
    "CLASS_LABELS",
    "bootstrap_ci",
    "collect_predictions",
    "compute_metrics",
    "mcnemar_test",
    "save_evaluation_report",
    "save_metrics_json",
]
