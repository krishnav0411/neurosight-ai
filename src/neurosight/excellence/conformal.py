"""conformal.py — split-conformal prediction sets (plan 07, T3).

A point prediction hides its own uncertainty; a prediction set does not. Split
conformal calibration (Vovk; Angelopoulos & Bates 2022) turns any held-out
score distribution into a set-valued predictor with a finite-sample guarantee:
with threshold q̂ taken as the ``k``-th smallest nonconformity score of a
calibration set, ``k = min(ceil((n+1)·coverage), n)``, a new exchangeable point
is covered with probability at least ``coverage``.

Here the nonconformity score of class *c* is ``1 - p(c)``, so class *c* joins
the set iff ``1 - p(c) <= q̂``. With two classes the set is empty, a singleton,
or both classes — never a fabricated certainty.

Fitted per fold on validation only (plan guardrail 5); the test set sees it
exactly once, at the bake-off.
"""

from __future__ import annotations

import math

import numpy as np

NON_DEMENTED = "Non-Demented"
DEMENTED = "Demented"


def conformal_rank(n: int, coverage: float = 0.9) -> int:
    """Corrected rank of the threshold: ``k = min(ceil((n+1)·coverage), n)``.

    The threshold is the ``k``-th smallest calibration score. The clamp handles
    the case where the finite-sample correction exceeds the sample size
    (n=6, coverage=0.9 -> ceil(6.3)=7 > 6): the old level formula
    ``ceil((n+1)·c)/n`` could then exceed 1 and relied on a post-hoc clamp.

    Raises:
        ValueError: on ``n < 1`` or coverage outside (0, 1].
    """
    if n < 1:
        raise ValueError(f"calibration set must have at least 1 sample, got n={n}")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    return int(min(math.ceil((n + 1) * coverage), n))


def conformal_threshold(n: int, coverage: float = 0.9) -> float:
    """Quantile level selecting the corrected rank: ``conformal_rank(n, c) / n``.

    Always in (0, 1] by construction — the rank is clamped to ``n`` before the
    division (see :func:`conformal_rank`).

    Raises:
        ValueError: on ``n < 1`` or coverage outside (0, 1].
    """
    return conformal_rank(n, coverage) / n


def fit_conformal(val_probs: np.ndarray, val_true: np.ndarray, coverage: float = 0.9) -> float:
    """Fit the conformal threshold on a calibration (validation) set.

    Nonconformity score of the true class is ``1 - p(true class)``; the
    returned threshold is the :func:`conformal_rank`-th smallest score —
    ``k = min(ceil((n+1)·coverage), n)``.

    Raises:
        ValueError: on empty or length-mismatched inputs.
    """
    probs = np.asarray(val_probs, dtype=np.float64)
    labels = np.asarray(val_true)
    if probs.size == 0 or labels.size == 0:
        raise ValueError("calibration inputs must be non-empty")
    if probs.shape != labels.shape:
        raise ValueError(f"val_probs/val_true length mismatch: {probs.shape} vs {labels.shape}")
    p_true = np.where(labels == 1, probs, 1.0 - probs)
    scores = 1.0 - p_true
    k = conformal_rank(int(scores.size), coverage)
    return float(np.sort(scores)[k - 1])


def split_half_coverage(
    val_probs: np.ndarray,
    val_true: np.ndarray,
    coverage: float = 0.9,
) -> dict[str, float | int | None]:
    """Out-of-sample empirical conformal coverage from one fold's val predictions.

    Coverage measured on the same rows the threshold was fit on is
    *calibration-set* coverage — optimistically biased, not an estimate for
    unseen data. The deterministic split-half fixes that: even-index rows
    calibrate (via :func:`fit_conformal`), odd-index rows are scored, so the
    threshold never sees the rows it is evaluated on.

    Folds too small to support an honest split (fewer than 2 rows in either
    half) report ``coverage=None`` rather than an in-sample number.

    Returns:
        ``{"coverage": float | None, "calib_n": int, "eval_n": int}``.

    Raises:
        ValueError: on length-mismatched inputs.
    """
    probs = np.asarray(val_probs, dtype=np.float64)
    labels = np.asarray(val_true)
    if probs.shape != labels.shape:
        raise ValueError(f"val_probs/val_true length mismatch: {probs.shape} vs {labels.shape}")
    cal_p, cal_y = probs[0::2], labels[0::2]
    eval_p, eval_y = probs[1::2], labels[1::2]
    if cal_p.size < 2 or eval_p.size < 2:
        return {"coverage": None, "calib_n": int(cal_p.size), "eval_n": int(eval_p.size)}
    threshold = fit_conformal(cal_p, cal_y, coverage)
    p_true = np.where(eval_y == 1, eval_p, 1.0 - eval_p)
    covered = (1.0 - p_true) <= threshold
    return {
        "coverage": float(covered.mean()),
        "calib_n": int(cal_p.size),
        "eval_n": int(eval_p.size),
    }


def prediction_set(p_demented: float, threshold: float) -> list[str]:
    """Prediction set for one case; a fixed-order subset of both class names.

    Class *c* is in the set iff its nonconformity score ``1 - p(c)`` clears the
    threshold. With p = ``p_demented``: Non-Demented has p(ND) = 1 - p, so its
    score is p; Demented's score is 1 - p. Below-half thresholds can produce an
    empty set — that is the method reporting genuine ambiguity, not a bug.
    """
    p = float(p_demented)
    members: list[str] = []
    if p <= threshold:  # score(Non-Demented) = 1 - (1 - p) = p
        members.append(NON_DEMENTED)
    if 1.0 - p <= threshold:  # score(Demented) = 1 - p
        members.append(DEMENTED)
    return members


__all__ = [
    "DEMENTED",
    "NON_DEMENTED",
    "conformal_rank",
    "conformal_threshold",
    "fit_conformal",
    "prediction_set",
    "split_half_coverage",
]
