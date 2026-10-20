"""Tests for preprocessing (plan T2.1): shapes, normalization, strategies,
deterministic augmentation, and the fail-loud contract."""

from __future__ import annotations

import numpy as np
import pytest

from neurosight.data.preprocessing import (
    VolumeLoadError,
    augment_2d,
    fix_shape,
    preprocess_2d,
    preprocess_3d,
    select_slices,
    z_score_normalize,
)


@pytest.fixture()
def volume() -> np.ndarray:
    """Synthetic (W=64, H=64, D=128) volume — depth-last, like the raw Analyze data."""
    rng = np.random.RandomState(0)
    base = rng.rand(64, 64, 128).astype(np.float32)
    return base * 100 + 500


def test_fix_shape_transposes_depth_last(volume: np.ndarray) -> None:
    out = fix_shape(volume)
    assert out.shape == (128, 64, 64)  # (D, H, W)


def test_fix_shape_squeezes_4d(volume: np.ndarray) -> None:
    out = fix_shape(volume[..., np.newaxis])  # (64, 64, 128, 1)
    assert out.ndim == 3 and out.shape == (128, 64, 64)


def test_fix_shape_rejects_2d() -> None:
    with pytest.raises(VolumeLoadError, match="3D"):
        fix_shape(np.zeros((64, 64), dtype=np.float32))


def test_fix_shape_rejects_tiny_volume() -> None:
    with pytest.raises(VolumeLoadError, match="implausible"):
        fix_shape(np.zeros((4, 4, 8), dtype=np.float32))


def test_z_score_normalize() -> None:
    vol = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    out = z_score_normalize(vol)
    assert out.mean() == pytest.approx(0.0, abs=1e-6)
    assert out.std() == pytest.approx(1.0, abs=1e-6)


def test_z_score_constant_volume_passthrough() -> None:
    vol = np.full((32, 8, 8), 7.0, dtype=np.float32)
    out = z_score_normalize(vol)
    assert np.allclose(out, vol)  # no NaN explosion on flat volumes


def test_select_slices_strategies() -> None:
    vol = np.arange(64 * 4 * 4, dtype=np.float32).reshape(64, 4, 4)
    middle = select_slices(vol, 32, "middle")
    assert middle.shape[0] == 32 and middle[0, 0, 0] == vol[16, 0, 0]
    uniform = select_slices(vol, 32, "uniform")
    assert uniform.shape[0] == 32 and uniform[0, 0, 0] == 0.0
    entropy = select_slices(vol, 32, "entropy")
    assert entropy.shape[0] == 32
    # shallow volume zero-pads to num_slices
    shallow = select_slices(np.zeros((10, 4, 4), dtype=np.float32), 32, "middle")
    assert shallow.shape == (32, 4, 4)
    with pytest.raises(ValueError, match="unknown slice strategy"):
        select_slices(vol, 32, "bogus")


def test_select_slices_entropy_pads_shallow_volumes() -> None:
    """The entropy strategy must honour the num_slices contract like middle
    does: shallow volumes are zero-padded, exact-depth volumes pass through."""
    vol = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    out = select_slices(vol, 10, "entropy")
    assert out.shape == (10, 4, 4)
    assert np.array_equal(out[:6], vol)  # original content preserved
    assert np.all(out[6:] == 0)  # zero pad, never garbage
    exact = select_slices(vol, 6, "entropy")
    assert np.array_equal(exact, vol)


def test_preprocess_2d_shape_and_stats(volume: np.ndarray) -> None:
    out = preprocess_2d(volume, num_slices=32, img_size=128)
    assert out.shape == (32, 128, 128)
    assert out.dtype == np.float32
    # slices come from a z-scored volume: mean ~0, well-bounded
    assert abs(out.mean()) < 1.0


def test_preprocess_2d_shallow_volume_pads(volume: np.ndarray) -> None:
    shallow = volume[:, :, :20]  # depth 20 < 32 slices
    out = preprocess_2d(shallow.transpose(2, 0, 1).astype(np.float32), num_slices=32)
    assert out.shape == (32, 128, 128)


def test_preprocess_3d_shape(volume: np.ndarray) -> None:
    out = preprocess_3d(volume, target=(64, 128, 128))
    assert out.shape == (1, 64, 128, 128)


def test_augmentation_deterministic_per_rng(volume: np.ndarray) -> None:
    stack = volume.transpose(2, 0, 1).astype(np.float32)[:32]
    a = augment_2d(stack, np.random.RandomState(123))
    b = augment_2d(stack, np.random.RandomState(123))
    c = augment_2d(stack, np.random.RandomState(124))
    assert np.array_equal(a, b)  # same seed -> identical
    assert not np.array_equal(a, c)  # different seed -> (virtually surely) different


def test_preprocess_augment_requires_rng(volume: np.ndarray) -> None:
    with pytest.raises(ValueError, match="seeded RandomState"):
        preprocess_2d(volume, augment=True, rng=None)
