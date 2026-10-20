"""preprocessing.py — deterministic, fail-loud preprocessing for OASIS-2 MRI volumes.

Ported from the predecessor project with the plan's fixes:

- **Fail loud**: a volume that cannot be interpreted raises ``VolumeLoadError`` —
  the old code's ``except: zeros`` silent-garbage path is gone (plan defect B6).
- **Deterministic augmentation**: stochastic ops take an explicit
  ``np.random.RandomState`` seeded by the caller (per dataset item), so the same
  sample+seed always yields the same augmentation.

Pipeline: shape fix → z-score → slice selection → resize → optional augment.
2D path yields (S, H, W); 3D path yields (1, D, H, W).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import zoom as scipy_zoom

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SLICE_STRATEGY = "middle"


class VolumeLoadError(ValueError):
    """Raised when a volume cannot be interpreted as a 3D MRI scan."""


def fix_shape(volume: np.ndarray) -> np.ndarray:
    """Squeeze 4D→3D and order axes to (depth, H, W).

    nibabel exposes Analyze 7.5 volumes as (W, H, D); the heuristic transposes
    when the last axis is the largest. Raises :class:`VolumeLoadError` for
    anything that is not a plausible 3D volume.
    """
    if volume.ndim == 4:
        volume = volume.squeeze()
    if volume.ndim != 3:
        raise VolumeLoadError(f"expected a 3D volume (or squeezable 4D), got shape {volume.shape}")
    if min(volume.shape) < 16:
        raise VolumeLoadError(f"implausible volume dimensions {volume.shape}")
    # Heuristic from the working predecessor code: depth is the largest axis.
    if volume.shape[2] > volume.shape[0]:
        volume = volume.transpose(2, 0, 1)
    return volume


def z_score_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score over the whole volume; near-constant volumes pass through."""
    std = volume.std()
    if std < eps:
        logger.warning("volume std %.2e < eps — returning unnormalized copy", std)
        return volume.copy()
    return (volume - volume.mean()) / (std + eps)


def compute_slice_entropy(slice_2d: np.ndarray) -> float:
    """Shannon entropy of a slice's 64-bin histogram — higher = more tissue signal."""
    hist, _ = np.histogram(slice_2d, bins=64, range=(slice_2d.min(), slice_2d.max()))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def select_slices(
    volume: np.ndarray,
    num_slices: int = 32,
    strategy: str = DEFAULT_SLICE_STRATEGY,
) -> np.ndarray:
    """Select ``num_slices`` slices from a (D, H, W) volume.

    middle: centre-most block (zero-padded if the volume is shallower);
    uniform: evenly spaced indices; entropy: top-k by tissue entropy.
    """
    depth = volume.shape[0]
    if strategy == "middle":
        if depth >= num_slices:
            start = (depth - num_slices) // 2
            return volume[start : start + num_slices]
        pad = np.zeros((num_slices - depth, *volume.shape[1:]), dtype=volume.dtype)
        return np.concatenate([volume, pad], axis=0)
    if strategy == "uniform":
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        return volume[indices]
    if strategy == "entropy":
        if depth <= num_slices:
            if depth == num_slices:
                return volume
            # honour the num_slices contract like "middle" does: shallow
            # volumes are zero-padded, never returned short
            pad = np.zeros((num_slices - depth, *volume.shape[1:]), dtype=volume.dtype)
            return np.concatenate([volume, pad], axis=0)
        entropies = [compute_slice_entropy(volume[i]) for i in range(depth)]
        indices = np.sort(np.argsort(entropies)[-num_slices:])
        return volume[indices]
    raise ValueError(f"unknown slice strategy: {strategy}")


def resize_2d_slices(volume: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize a (S, H, W) stack to (S, size, size) with bilinear interpolation."""
    out = np.zeros((volume.shape[0], size, size), dtype=np.float32)
    for i in range(volume.shape[0]):
        out[i] = cv2.resize(volume[i], (size, size), interpolation=cv2.INTER_LINEAR)
    return out


def resize_3d_volume(
    volume: np.ndarray, target: tuple[int, int, int] = (64, 128, 128)
) -> np.ndarray:
    """Trilinear resize of a (D, H, W) volume to ``target``."""
    factors = (
        target[0] / volume.shape[0],
        target[1] / volume.shape[1],
        target[2] / volume.shape[2],
    )
    return scipy_zoom(volume, factors, order=1).astype(np.float32)


def augment_2d(
    volume: np.ndarray,
    rng: np.random.RandomState,
    slice_shift: int = 2,
    noise_std: float = 0.02,
    flip_prob: float = 0.3,
    intensity_jitter: float = 0.05,
) -> np.ndarray:
    """2D-stack augmentation driven by a caller-seeded RNG (deterministic replay)."""
    aug = volume.copy()
    if rng.rand() < 0.5:
        aug = np.roll(aug, rng.randint(-slice_shift, slice_shift + 1), axis=0)
    if rng.rand() < 0.5:
        aug = aug + rng.randn(*aug.shape).astype(np.float32) * noise_std
    if rng.rand() < flip_prob:
        aug = np.flip(aug, axis=2).copy()
    if rng.rand() < 0.5:
        aug = aug * (1.0 + rng.uniform(-intensity_jitter, intensity_jitter))
    return aug.astype(np.float32)


def augment_3d(
    volume: np.ndarray,
    rng: np.random.RandomState,
    noise_std: float = 0.02,
    flip_prob: float = 0.3,
) -> np.ndarray:
    """3D-volume augmentation driven by a caller-seeded RNG."""
    aug = volume.copy()
    if rng.rand() < 0.5:
        aug = aug + rng.randn(*aug.shape).astype(np.float32) * noise_std
    if rng.rand() < flip_prob:
        aug = np.flip(aug, axis=2).copy()
    if rng.rand() < 0.5:
        aug = aug * (1.0 + rng.uniform(-0.05, 0.05))
    return aug.astype(np.float32)


def preprocess_2d(
    scan: np.ndarray,
    num_slices: int = 32,
    img_size: int = 128,
    slice_strategy: str = DEFAULT_SLICE_STRATEGY,
    augment: bool = False,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Full 2D pipeline: (S, H, W) float32. Raises VolumeLoadError on bad input."""
    vol = fix_shape(np.asarray(scan, dtype=np.float32))
    vol = z_score_normalize(vol)
    vol = select_slices(vol, num_slices=num_slices, strategy=slice_strategy)
    vol = resize_2d_slices(vol, size=img_size)
    if augment:
        if rng is None:
            raise ValueError("augment=True requires an explicit seeded RandomState")
        vol = augment_2d(vol, rng)
    return vol


def preprocess_3d(
    scan: np.ndarray,
    target: tuple[int, int, int] = (64, 128, 128),
    augment: bool = False,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Full 3D pipeline: (1, D, H, W) float32. Raises VolumeLoadError on bad input."""
    vol = fix_shape(np.asarray(scan, dtype=np.float32))
    vol = z_score_normalize(vol)
    vol = resize_3d_volume(vol, target)
    if augment:
        if rng is None:
            raise ValueError("augment=True requires an explicit seeded RandomState")
        vol = augment_3d(vol, rng)
    return vol[np.newaxis, :, :, :]


__all__ = [
    "VolumeLoadError",
    "augment_2d",
    "augment_3d",
    "compute_slice_entropy",
    "fix_shape",
    "preprocess_2d",
    "preprocess_3d",
    "resize_2d_slices",
    "resize_3d_volume",
    "select_slices",
    "z_score_normalize",
]
