"""modes.py — leakage-controlled input modes (plan defect B1, PRD FR-3).

The 8-dim metadata vector layout is inherited from the predecessor project:

    0: sex (M/F)      1: age/100     2: educ/20    3: ses/5
    4: mmse/30        5: eTIV/2000   6: nWBV       7: ASF/2

MMSE (dim 4) is a *diagnosis-defining* clinical score — feeding it silently and
reporting the accuracy as "MRI-based" is label leakage (00-PLAN-REVIEW B1).
Every experiment therefore runs in exactly one mode; the mask is part of the
run config and the exported checkpoint manifest so serving can never drift
from training. Headline results come from ``mri_only`` only.
"""

from __future__ import annotations

import numpy as np

# Dimension groups
DEMO_DIMS = [0, 1, 2, 3]  # sex, age, education, SES
MRI_DERIVED_DIMS = [5, 6, 7]  # eTIV, nWBV, ASF (derived from the MRI itself)
CLINICAL_DIMS = [4]  # MMSE — label leakage outside `mri_clinical`

INPUT_MODE_MASKS: dict[str, list[int]] = {
    "mri_only": [],  # image tensor only, no metadata
    "demo_only": DEMO_DIMS,  # demographics baseline (no image used by tabular model)
    "mri_demo": DEMO_DIMS + MRI_DERIVED_DIMS,  # image + demographics/MRI-derived
    "mri_clinical": DEMO_DIMS + MRI_DERIVED_DIMS + CLINICAL_DIMS,  # experimental, leak-prone
}

METADATA_DIM = 8


def metadata_mask(input_mode: str) -> np.ndarray:
    """Return an (8,) float32 0/1 mask selecting the metadata dims for ``input_mode``."""
    if input_mode not in INPUT_MODE_MASKS:
        raise ValueError(
            f"unknown input_mode '{input_mode}'; expected one of {sorted(INPUT_MODE_MASKS)}"
        )
    mask = np.zeros(METADATA_DIM, dtype=np.float32)
    mask[INPUT_MODE_MASKS[input_mode]] = 1.0
    return mask


def apply_metadata_mask(meta: np.ndarray, input_mode: str) -> np.ndarray:
    """Zero out the masked dims of a (..., 8) metadata array (copy, input untouched)."""
    return meta * metadata_mask(input_mode)


__all__ = [
    "CLINICAL_DIMS",
    "DEMO_DIMS",
    "INPUT_MODE_MASKS",
    "METADATA_DIM",
    "MRI_DERIVED_DIMS",
    "apply_metadata_mask",
    "metadata_mask",
]
