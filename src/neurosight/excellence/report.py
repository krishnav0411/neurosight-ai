"""report.py — per-fold metric summaries and the system comparison table (plan 07, T4 support).

Turns the fold-prediction dicts produced by
:mod:`neurosight.excellence.ensemble` into reportable numbers: per-fold AUC,
ECE raw (mean-softmax ensemble) and ECE temperature-scaled, plus mean and std.
The scaled ECE is OUT OF SAMPLE: within each fold, the val observations are
split deterministically — even indices calibrate the temperature, odd indices
score the scaled ECE — so the temperature is never evaluated on the rows it
was fitted on (in-sample scaled ECE is optimistically biased). Both ECEs are
reported side by side because temperature and ensembling interact (Rahaman,
NeurIPS 2021).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from neurosight.evaluation.calibration import (
    MIN_TEMPERATURE,
    expected_calibration_error,
    fit_temperature,
)
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

# a fold needs at least 2 calibration + 2 evaluation rows for the even/odd
# split to support an out-of-sample temperature fit; smaller folds fall back
# to raw ECE only (ece_scaled: null)
MIN_SPLIT_OBSERVATIONS = 4


def probs_to_logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Positive-class probabilities -> (n, 2) logit matrix for the temperature fit.

    Inverse softmax under the two-class assumption: ``[log(1-p), log(p)]``.
    """
    clipped = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.stack([np.log1p(-clipped), np.log(clipped)], axis=1)


def evaluate_fold_predictions(per_fold: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Per-fold AUC + ECE raw/scaled and their mean±std for one system.

    Temperature is fitted per fold (plan guardrail 5) — never pooled across
    folds, never on test data — and OUT OF SAMPLE: each fold's val observations
    are split deterministically into even-index rows (temperature calibration)
    and odd-index rows (scaled-ECE evaluation), so the scaled ECE is never
    computed on the observations the temperature was fitted on. Folds with
    fewer than :data:`MIN_SPLIT_OBSERVATIONS` observations cannot support the
    split honestly: they report the raw ECE only, with ``ece_scaled`` and
    ``temperature`` null. ``ece_calib_n`` / ``ece_eval_n`` are the sizes of the
    two halves of the deterministic split (reported but unused in the fallback).
    """
    if not per_fold:
        raise ValueError("no fold predictions to summarize")

    folds: dict[str, dict[str, Any]] = {}
    aucs: list[float] = []
    eces_raw: list[float] = []
    eces_scaled: list[float] = []
    for fold in sorted(per_fold):
        y_true = np.asarray(per_fold[fold]["y_true"], dtype=np.int64)
        y_prob = np.asarray(per_fold[fold]["y_prob"], dtype=np.float64)
        auc = float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else 0.0

        # deterministic out-of-sample split: even indices calibrate the
        # temperature, odd indices score the scaled ECE
        cal_true, cal_prob = y_true[0::2], y_prob[0::2]
        eval_true, eval_prob = y_true[1::2], y_prob[1::2]

        ece_raw = expected_calibration_error(y_true, y_prob)
        ece_scaled: float | None = None
        temperature: float | None = None
        if len(cal_true) >= 2 and len(eval_true) >= 2:
            logits = torch.from_numpy(probs_to_logits(cal_prob))
            scaler = fit_temperature(logits, torch.from_numpy(cal_true))
            y_prob_scaled = scaler.apply_probs(eval_prob)

            ece_scaled = expected_calibration_error(eval_true, y_prob_scaled)
            # the EFFECTIVE temperature apply_probs used (fit_temperature clamps
            # the parameter to the same floor; the belt-and-braces max() here
            # keeps the report honest even for hand-constructed scalers)
            temperature = max(float(scaler.temperature.item()), MIN_TEMPERATURE)

        folds[str(fold)] = {
            "n": len(y_true),
            "auc": auc,
            "ece_raw": float(ece_raw),
            "ece_scaled": ece_scaled,
            "temperature": temperature,
            "ece_calib_n": int(cal_true.size),
            "ece_eval_n": int(eval_true.size),
        }
        aucs.append(auc)
        eces_raw.append(ece_raw)
        if ece_scaled is not None:
            eces_scaled.append(ece_scaled)
        logger.info(
            "fold %s: auc=%.4f ece_raw=%.4f ece_scaled=%s (T=%s) calib_n=%d eval_n=%d",
            fold,
            auc,
            ece_raw,
            "n/a" if ece_scaled is None else f"{ece_scaled:.4f}",
            "n/a" if temperature is None else f"{temperature:.3f}",
            cal_true.size,
            eval_true.size,
        )

    return {
        "folds": folds,
        "n_folds": len(folds),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_ece_raw": float(np.mean(eces_raw)),
        "mean_ece_scaled": float(np.mean(eces_scaled)) if eces_scaled else None,
    }


def format_comparison_table(
    summaries: Mapping[str, Mapping[str, Any]],
    baseline_auc: float,
    baseline_std: float,
) -> str:
    """ASCII comparison table: system | mean AUC | std | folds | ECE raw/scaled vs S0."""
    lines = [
        "=" * 86,
        "  NeuroSight-E5 — same-fold CV system comparison (S0 = single champion baseline)",
        "=" * 86,
        f"  {'System':<8} {'MeanAUC':>8} {'Std':>7} {'Folds':>6} "
        f"{'ECEraw':>8} {'ECEscaled':>10} {'dAUCvsS0':>9}",
        "  " + "-" * 72,
        f"  {'S0':<8} {baseline_auc:>8.4f} {baseline_std:>7.4f} "
        f"{'-':>6} {'-':>8} {'-':>10} {0.0:>9.4f}",
    ]
    for system in sorted(summaries):
        s = summaries[system]
        scaled = s.get("mean_ece_scaled")
        scaled_cell = "-" if scaled is None else f"{float(scaled):>10.4f}"
        lines.append(
            f"  {system:<8} {float(s['mean_auc']):>8.4f} {float(s['std_auc']):>7.4f} "
            f"{int(s['n_folds']):>6} {float(s['mean_ece_raw']):>8.4f} "
            f"{scaled_cell:>10} {float(s['mean_auc']) - baseline_auc:>9.4f}"
        )
    lines += [
        "",
        "  Selection: max mean CV AUC, tie -> ECE (plan guardrail 2).",
        "  Held-out test set untouched — one bake-off remains (plan guardrail 1).",
    ]
    return "\n".join(lines)


__all__ = ["evaluate_fold_predictions", "format_comparison_table", "probs_to_logits"]
