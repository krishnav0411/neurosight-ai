"""Tests for datasets, collate, sampler, and cache (plan T2.2).

Synthetic studies are real (tiny) NIfTI files written by nibabel, laid out with
the same ``<MRI_ID>/RAW/mpr-N`` convention the loader produces.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from neurosight.data.cache import SessionCache, build_session_cache
from neurosight.data.dataset import (
    Oasis2DDataset,
    Oasis3DDataset,
    collate_2d,
    collate_3d_seq,
    collate_3d_single,
    make_dataloaders,
    make_weighted_sampler,
)
from neurosight.data.loader import SequenceRecord

W, H, D = 32, 32, 48  # raw-ish shape; fix_shape transposes to depth-first


def _2d_loader_stub(study: list[SequenceRecord]):
    """Minimal loader stand-in for build_session_cache over the study fixture."""

    class LoaderStub:
        config = type("C", (), {"series_selection": "first"})()

        @staticmethod
        def iter_sessions():
            for rec in study:
                for paths in rec.visit_paths:
                    yield Path(paths[0]).parent.parent.name, rec.subject_id, paths

    return LoaderStub()


@pytest.fixture()
def study(tmp_path: Path) -> list[SequenceRecord]:
    """Two subjects: N01 (3 visits) and D01 (2 visits), real tiny NIfTI volumes."""
    records = []
    for subject, n_visits, label in (("N01", 3, 0), ("D01", 2, 1)):
        paths_per_visit, metas, delays = [], [], []
        for v in range(1, n_visits + 1):
            mri_id = f"{subject}_MR{v}"
            raw = tmp_path / mri_id / "RAW"
            raw.mkdir(parents=True, exist_ok=True)
            vol = np.random.RandomState(hash(mri_id) % 2**31).rand(W, H, D).astype(np.float32) * 100
            path = raw / "mpr-1.nii"
            nib.save(nib.Nifti1Image(vol, np.eye(4)), path)
            paths_per_visit.append([str(path)])
            metas.append(np.linspace(0.1, 0.8, 8, dtype=np.float32))
            delays.append(365 * (v - 1))
        records.append(
            SequenceRecord(
                subject_id=subject,
                kind="full",
                visit_paths=paths_per_visit,
                visit_meta=metas,
                visit_labels=[0] * n_visits if label == 0 else [1] * n_visits,
                visit_delays=delays,
                label=label,
            )
        )
    return records


def test_collate_2d_variable_lengths() -> None:
    batch = [
        (torch.randn(2, 4, 8, 8), torch.randn(2, 8), torch.tensor(0)),
        (torch.randn(3, 4, 8, 8), torch.randn(3, 8), torch.tensor(1)),
        (torch.randn(1, 4, 8, 8), torch.randn(1, 8), torch.tensor(0)),
    ]
    scans, metas, labels, mask = collate_2d(batch)
    assert scans.shape == (3, 3, 4, 8, 8)
    assert metas.shape == (3, 3, 8)
    assert labels.tolist() == [0, 1, 0]
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1], [1, 0, 0]]
    # padded scan timesteps are exactly zero
    assert torch.all(scans[2, 1:] == 0)


def test_collate_3d_single_and_seq() -> None:
    single = [(torch.randn(1, 8, 8, 8), torch.randn(8), torch.tensor(0)) for _ in range(3)]
    vols, _metas, _labels, ones = collate_3d_single(single)
    assert vols.shape == (3, 1, 8, 8, 8) and ones.shape == (3, 1)

    seq = [
        (torch.randn(2, 1, 8, 8, 8), torch.randn(2, 8), torch.tensor(1)),
        (torch.randn(1, 1, 8, 8, 8), torch.randn(1, 8), torch.tensor(0)),
    ]
    vols, _metas, _labels, mask = collate_3d_seq(seq)
    assert vols.shape == (2, 2, 1, 8, 8, 8)
    assert mask.tolist() == [[1, 1], [1, 0]]


def test_dataset_2d_end_to_end(study: list[SequenceRecord]) -> None:
    ds = Oasis2DDataset(study, num_slices=32, img_size=64)
    assert len(ds) == 2
    scans, meta, label = ds[0]
    assert scans.shape == (3, 32, 64, 64)
    assert scans.dtype == torch.float32
    assert meta.shape == (3, 8)
    assert label.item() == 0
    # mri_only masks every metadata dim
    assert torch.all(meta == 0)
    ds_demo = Oasis2DDataset(study, input_mode="mri_demo", num_slices=32, img_size=64)
    _, meta_demo, _ = ds_demo[0]
    assert meta_demo[0, 4] == 0 and meta_demo[0, 0] != 0  # MMSE masked, demographics kept


def test_dataset_2d_deterministic(study: list[SequenceRecord]) -> None:
    ds = Oasis2DDataset(study, num_slices=32, img_size=64, augment=True, temporal_augment=True)
    a = ds[0]
    b = ds[0]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_dataset_fails_loud_on_missing_scan(study: list[SequenceRecord]) -> None:
    Path(study[0].visit_paths[0][0]).unlink()  # delete visit 1's scan
    ds = Oasis2DDataset(study, num_slices=32, img_size=64)
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_dataset_uses_cache(study: list[SequenceRecord], tmp_path: Path) -> None:
    class LoaderStub:
        config = type("C", (), {"series_selection": "first"})()

        @staticmethod
        def iter_sessions():
            for rec in study:
                for paths in rec.visit_paths:
                    mri_id = Path(paths[0]).parent.parent.name
                    yield mri_id, rec.subject_id, paths

    stats = build_session_cache(
        LoaderStub(), tmp_path / "processed", num_slices=32, img_size=64, include_3d=False
    )
    assert stats["processed"] == 5  # 3 + 2 visits across the two subjects
    cold = Oasis2DDataset(study, num_slices=32, img_size=64)
    warm = Oasis2DDataset(
        study, num_slices=32, img_size=64, cache=SessionCache(tmp_path / "processed", mode="2d")
    )
    assert np.allclose(cold[0][0].numpy(), warm[0][0].numpy(), atol=1e-5)
    assert warm.cache.hits > 0
    # parameter change -> cache miss (stale data can never be served as fresh)
    stale = Oasis2DDataset(
        study, num_slices=16, img_size=64, cache=SessionCache(tmp_path / "processed", mode="2d")
    )
    assert stale[0][0].shape == (3, 16, 64, 64)


def test_train_augmentation_applies_on_cache_hits(
    study: list[SequenceRecord], tmp_path: Path
) -> None:
    """The cache stores PRE-augmentation tensors: a train-mode cache hit must
    still be augmented (deterministically) — the old early return served every
    cached training sample unaugmented."""
    build_session_cache(_2d_loader_stub(study), tmp_path / "processed", num_slices=32, img_size=64)
    cache = SessionCache(tmp_path / "processed", mode="2d")
    warm_train = Oasis2DDataset(
        study, num_slices=32, img_size=64, augment=True, temporal_augment=True, cache=cache
    )
    cold_train = Oasis2DDataset(
        study, num_slices=32, img_size=64, augment=True, temporal_augment=True
    )
    cache.hits = 0
    for i in range(len(study)):
        assert torch.equal(warm_train[i][0], cold_train[i][0]), f"item {i} diverged"
    assert cache.hits >= len(study), "the comparison never actually hit the cache"

    # the augmentation is not a silent no-op on this fixture
    raw = Oasis2DDataset(
        study, num_slices=32, img_size=64, cache=SessionCache(tmp_path / "processed", mode="2d")
    )
    changed = sum(not torch.equal(warm_train[i][0], raw[i][0]) for i in range(len(study)))
    assert changed >= 1, "augmentation was a no-op on this fixture; the test lost its teeth"


def test_train_augmentation_applies_on_3d_cache_hits(
    study: list[SequenceRecord], tmp_path: Path
) -> None:
    build_session_cache(
        _2d_loader_stub(study),
        tmp_path / "processed",
        num_slices=32,
        img_size=64,
        vol_depth=32,
        include_3d=True,
    )
    cache = SessionCache(tmp_path / "processed", mode="3d")
    warm = Oasis3DDataset(
        study, mode="sequential", vol_depth=32, img_size=64, augment=True, cache=cache
    )
    cold = Oasis3DDataset(study, mode="sequential", vol_depth=32, img_size=64, augment=True)
    cache.hits = 0
    for i in range(len(study)):
        assert torch.equal(warm[i][0], cold[i][0]), f"item {i} diverged"
    assert cache.hits > 0, "the comparison never actually hit the cache"


def test_cache_modes_are_isolated(tmp_path: Path) -> None:
    """A 2D entry must never satisfy a 3D lookup (and vice versa) — the bug that
    silently skipped the whole 3D cache build."""
    cache_2d = SessionCache(tmp_path, mode="2d")
    cache_3d = SessionCache(tmp_path, mode="3d")
    cache_2d.put(
        "OAS2_0001_MR1",
        "OAS2_0001",
        np.zeros((32, 64, 64), dtype=np.float32),
        32,
        64,
        "middle",
        "first",
    )
    assert cache_2d.contains("OAS2_0001_MR1")
    assert not cache_3d.contains("OAS2_0001_MR1")  # mode isolation
    assert cache_2d.get("OAS2_0001_MR1", 32, 64, "middle", "first") is not None
    assert cache_3d.get("OAS2_0001_MR1", 64, 64, "middle", "first") is None


def test_dataset_3d_single_and_sequential(study: list[SequenceRecord]) -> None:
    single = Oasis3DDataset(study, mode="single", vol_depth=32, img_size=64)
    vols, _meta, label = single[0]
    assert vols.shape == (1, 32, 64, 64) and label.item() == 0  # (1, D, H, W)
    seq = Oasis3DDataset(study, mode="sequential", vol_depth=32, img_size=64)
    vols, _meta, label = seq[0]
    assert vols.shape == (3, 32, 64, 64)  # (T, D, H, W) channelless


def test_dataset_3d_cache_and_live_shapes_match(
    study: list[SequenceRecord], tmp_path: Path
) -> None:
    """The cache and live-preprocessing paths must agree exactly on shape —
    their divergence crashed 10_3d_lstm after 8.5 h of training."""
    from neurosight.data.cache import build_session_cache

    class LoaderStub:
        config = type("C", (), {"series_selection": "first"})()

        @staticmethod
        def iter_sessions():
            for rec in study:
                for paths in rec.visit_paths:
                    yield Path(paths[0]).parent.parent.name, rec.subject_id, paths

    build_session_cache(
        LoaderStub(),
        tmp_path / "processed",
        num_slices=32,
        img_size=64,
        vol_depth=32,
        include_3d=True,
    )
    cache = SessionCache(tmp_path / "processed", mode="3d")
    for mode, expected in (("single", (1, 32, 64, 64)), ("sequential", (3, 32, 64, 64))):
        cold = Oasis3DDataset(study, mode=mode, vol_depth=32, img_size=64)
        warm = Oasis3DDataset(study, mode=mode, vol_depth=32, img_size=64, cache=cache)
        assert cold[0][0].shape == warm[0][0].shape == expected, mode


def test_dataset_3d_cache_uses_configured_policy(
    study: list[SequenceRecord], tmp_path: Path
) -> None:
    """3D lookups must pass the dataset's slice_strategy/series_selection —
    the hardcoded "middle"/"first" made every non-default lookup miss, so each
    item re-ran preprocess_3d on the full volume and epochs slowed down."""
    from neurosight.data.cache import build_session_cache

    class LoaderStub:
        config = type("C", (), {"series_selection": "first"})()

        @staticmethod
        def iter_sessions():
            for rec in study:
                for paths in rec.visit_paths:
                    yield Path(paths[0]).parent.parent.name, rec.subject_id, paths

    build_session_cache(
        LoaderStub(),
        tmp_path / "processed",
        num_slices=32,
        img_size=64,
        slice_strategy="uniform",
        series_selection="first",
        vol_depth=32,
        include_3d=True,
    )
    cache = SessionCache(tmp_path / "processed", mode="3d")
    ds = Oasis3DDataset(
        study,
        mode="single",
        vol_depth=32,
        img_size=64,
        slice_strategy="uniform",
        series_selection="first",
        cache=cache,
    )
    ds[0]
    assert cache.hits > 0, "3D cache lookup ignored the configured slice strategy"


def test_make_dataloaders_3d_passes_policy(study: list[SequenceRecord], configs_dir: Path) -> None:
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    for fam in ("3d", "3d_lstm"):
        train, _val, _test = make_dataloaders(study, study[:1], None, fam, cfg.data, cfg.training)
        ds = train.dataset
        assert ds.slice_strategy == cfg.data.slice_strategy, fam
        assert ds.series_selection == cfg.data.series_selection, fam


def test_weighted_sampler_balances(study: list[SequenceRecord]) -> None:
    records = study * 5  # 10 label-0, 10 label-1... build imbalance instead
    records = [study[0]] * 9 + [study[1]]  # 9 label-0, 1 label-1
    picked = []
    for _ in range(200):  # each pass yields len(records) draws
        picked.extend(records[i].label for i in iter(make_weighted_sampler(records)))
    ones = sum(picked) / len(picked)
    assert 0.3 < ones < 0.7  # inverse-frequency pulls the rare class up


def test_weighted_sampler_empty_records() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_weighted_sampler([])


def test_weighted_sampler_maps_weights_by_label_value(study: list[SequenceRecord]) -> None:
    """Weights key on the label VALUE, not the np.unique position: a fold with
    a single class (e.g. only label 1) must not IndexError, and any label set
    must map each sample to its own class's inverse frequency."""
    sampler = make_weighted_sampler([study[1]] * 3)  # single-class fold, label 1
    # uniform weights (up to scale) — and, crucially, no IndexError from the old
    # positional weights[1] on a length-1 np.unique array
    assert torch.allclose(sampler.weights, torch.full((3,), 1.0 / 3.0, dtype=torch.float64))

    mixed = [study[0], study[1], study[1], study[1]]  # 1x label-0, 3x label-1
    sampler = make_weighted_sampler(mixed)
    assert sampler.weights[0].item() == pytest.approx(1.0)  # label 0: 1/1
    assert torch.allclose(sampler.weights[1:], torch.full((3,), 1.0 / 3.0, dtype=torch.float64))


def test_make_dataloaders_2d(study: list[SequenceRecord], configs_dir: Path) -> None:
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    train, _val, test = make_dataloaders(study, study[:1], study[:1], "2d", cfg.data, cfg.training)
    scans, _metas, _labels, mask = next(iter(train))
    assert scans.shape[2:] == (cfg.data.num_slices, cfg.data.slice_size, cfg.data.slice_size)
    assert mask.shape[0] == scans.shape[0]
    assert test is not None


def test_make_dataloaders_unknown_family(study: list[SequenceRecord], configs_dir: Path) -> None:
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    with pytest.raises(ValueError, match="unknown model family"):
        make_dataloaders(study, study[:1], None, "4d", cfg.data, cfg.training)


def test_collate_matches_dataloader_flow(study: list[SequenceRecord]) -> None:
    ds = Oasis2DDataset(study, num_slices=32, img_size=64)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_2d)
    for scans, metas, labels, mask in loader:
        assert mask.sum() <= mask.numel()
        assert scans.shape[0] == labels.shape[0] == metas.shape[0] == mask.shape[0]
