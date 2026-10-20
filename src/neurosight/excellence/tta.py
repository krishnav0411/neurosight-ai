"""tta.py — test-time augmentation with mean-softmax aggregation (plan 07, T2).

Only medically-plausible transforms (plan S3): the identity, a left-right
flip (head orientation is the only plausible anatomical mirror), and small
slice shifts of ±2 — the same 32-slice stack resampled a slice either way.
The shifts are ZERO-FILLED and NON-CIRCULAR: slices pushed past the stack
boundary are discarded and the vacated edge is zero, matching the
preprocessing zero-padding contract. A circular roll would wrap superior
anatomy onto the inferior boundary — anatomically impossible, so it is out.
Anything affine-in-3D or intensity-warped is deliberately absent.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

SLICE_SHIFT = 2  # slices to shift (zero-filled) along the slice axis (dim=2); ±2 per plan S3


def shift_slices_zerofill(scans: torch.Tensor, shift: int) -> torch.Tensor:
    """Zero-filled, NON-circular shift along the slice axis (dim=2).

    ``shift > 0`` moves slice content toward higher indices: the first
    ``shift`` slices become zero and the last ``shift`` slices are discarded
    (mirror image for ``shift < 0``). Unlike ``torch.roll`` nothing wraps
    around — superior anatomy can never reappear at the inferior boundary —
    and the zero fill matches the preprocessing zero-padding contract.
    """
    if shift == 0:
        return scans.clone()
    shifted = torch.zeros_like(scans)
    if shift > 0:
        shifted[:, :, shift:] = scans[:, :, :-shift]
    else:
        shifted[:, :, :shift] = scans[:, :, -shift:]
    return shifted


def tta_batches(scans: torch.Tensor) -> list[torch.Tensor]:
    """The four TTA views of a (B, T, S, H, W) scan batch, in fixed order.

    1. identity
    2. horizontal flip — ``scans.flip(-1)`` (mirrors W, the left-right axis)
    3. slice shift +SLICE_SHIFT on the slice axis (dim=2) — zero-filled, no wrap
    4. slice shift -SLICE_SHIFT on the slice axis (dim=2) — zero-filled, no wrap
    """
    if scans.dim() != 5:
        raise ValueError(
            f"expected (B, T, S, H, W) scans, got {scans.dim()} dims (shape {tuple(scans.shape)})"
        )
    return [
        scans,
        scans.flip(-1),
        shift_slices_zerofill(scans, SLICE_SHIFT),
        shift_slices_zerofill(scans, -SLICE_SHIFT),
    ]


def tta_collect_predictions(
    model: nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect predictions with TTA: mean softmax across the four views.

    Same contract as
    :func:`neurosight.evaluation.evaluator.collect_predictions` — returns
    ``(y_true, y_pred, y_prob)`` where ``y_prob`` is the averaged positive-class
    probability. Batch inference only: ``model.eval()`` + ``torch.no_grad()``.
    """
    model.eval()
    all_labels: list[float] = []
    all_probs: list[float] = []
    with torch.no_grad():
        for scans, meta, labels, mask in loader:
            scans = scans.to(device)
            meta = meta.to(device)
            mask = mask.to(device)
            prob_sum: torch.Tensor | None = None
            for view in tta_batches(scans):
                logits = model(view, meta, mask)
                probs = torch.softmax(logits, dim=1)
                prob_sum = probs if prob_sum is None else prob_sum + probs
            assert prob_sum is not None  # tta_batches always yields the 4 members
            mean_probs = prob_sum / 4.0
            all_probs.extend(mean_probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    logger.debug("TTA collected %d predictions (4 passes each)", len(y_true))
    return y_true, (y_prob >= 0.5).astype(int), y_prob


__all__ = ["SLICE_SHIFT", "shift_slices_zerofill", "tta_batches", "tta_collect_predictions"]
