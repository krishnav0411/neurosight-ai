"""excellence — Model-Excellence harness (docs/plan/07-MODEL-EXCELLENCE-PLAN.md).

Offline evaluation systems built on the existing CV checkpoints. Ensembling is
SAME-FOLD only: fold i is scored by models trained blind to fold i's val (its
own per-variant fold-i checkpoints) — cross-fold members would score training
data and are banned.

- T1 :mod:`neurosight.excellence.ensemble` — per-variant same-fold CV
  predictions + cross-variant averaging.
- T2 :mod:`neurosight.excellence.tta` — test-time augmentation (identity,
  h-flip, slice-shift ±2).
- T3 :mod:`neurosight.excellence.conformal` — split-conformal prediction sets.
- report helpers in :mod:`neurosight.excellence.report` — per-fold metric
  summaries and the S0-vs-system comparison table.

Selection happens on CV only; the held-out test set is touched exactly once by
the declared bake-off (plan guardrail 1).
"""

from __future__ import annotations

from neurosight.excellence.conformal import (
    conformal_rank,
    conformal_threshold,
    fit_conformal,
    prediction_set,
    split_half_coverage,
)
from neurosight.excellence.ensemble import (
    average_variant_probs,
    collect_fold_checkpoints,
    samefold_cross_variant,
    samefold_fold_predictions,
)
from neurosight.excellence.tta import (
    shift_slices_zerofill,
    tta_batches,
    tta_collect_predictions,
)

__all__ = [
    "average_variant_probs",
    "collect_fold_checkpoints",
    "conformal_rank",
    "conformal_threshold",
    "fit_conformal",
    "prediction_set",
    "samefold_cross_variant",
    "samefold_fold_predictions",
    "shift_slices_zerofill",
    "split_half_coverage",
    "tta_batches",
    "tta_collect_predictions",
]
