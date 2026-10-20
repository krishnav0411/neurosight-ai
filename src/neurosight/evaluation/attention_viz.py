"""attention_viz.py — temporal attention visualisation (plan T5.3).

Shows WHICH visits drove the prediction (per patient) and whether the model
weights later visits more for demented patients (population view).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


def plot_attention_weights(
    weights: np.ndarray,
    subject_id: str,
    pred_label: str,
    true_label: str | None,
    out_dir: str | Path,
    visit_labels: list[str] | None = None,
) -> Path:
    """Bar chart of attention weight per visit for one patient."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.max() if w.max() > 0 else 1.0)
    labels = visit_labels or [f"MR{i + 1}" for i in range(len(w))]
    if true_label is None:
        bar_color = "#2563eb"  # ground truth unknown — neutral blue
    else:
        bar_color = "#16a34a" if pred_label == true_label else "#dc2626"
    colors = [bar_color] * len(w)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, w, color=colors)
    ax.set_ylabel("Attention (norm.)")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f"{subject_id}: pred={pred_label}" + (f" true={true_label}" if true_label else ""),
        fontsize=10,
    )
    fig.tight_layout()
    path = out / f"attention_{subject_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("attention plot saved: %s", path)
    return path


def plot_population_attention(
    all_weights: np.ndarray, labels: np.ndarray, out_dir: str | Path
) -> Path:
    """Average attention by visit position, grouped by true class."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_weights = np.atleast_2d(np.asarray(all_weights, dtype=np.float64))
    max_t = all_weights.shape[1]

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name, color in ((0, "Non-Demented", "#16a34a"), (1, "Demented", "#dc2626")):
        rows = all_weights[np.asarray(labels) == label]
        if len(rows) == 0:
            continue
        mean = rows.mean(axis=0)
        mean = mean / (mean.max() if mean.max() > 0 else 1.0)
        ax.plot(range(1, max_t + 1), mean, "o-", label=name, color=color)
    ax.set_xlabel("Visit position")
    ax.set_ylabel("Mean attention (norm.)")
    ax.set_title("Population temporal attention by class", fontsize=11)
    ax.legend()
    fig.tight_layout()
    path = out / "attention_population.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("population attention plot saved: %s", path)
    return path


__all__ = ["plot_attention_weights", "plot_population_attention"]
