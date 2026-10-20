"""dataset.py — PyTorch datasets, collate functions, and sampler for all model families.

Ported from the predecessor project with the plan's fixes:

- **Fail loud**: unreadable scans raise — no zero-tensor substitution (B6).
- **Mode-masked metadata**: the 8-dim clinical vector is masked per ``input_mode``
  before leaving the dataset (B1 — see neurosight/data/modes.py).
- **Deterministic augmentation**: per-item RNG seeded from (dataset seed, index).
- Cache-aware: sessions resolve through :class:`neurosight.data.cache.SessionCache`
  when one is supplied.

Families:
    Oasis2DDataset       -> (T, S, H, W) scans       (2d models)
    Oasis3DDataset       -> (1, D, H, W) or (T, 1, D, H, W)  (3d / 3d_lstm)
    collate_*             -> zero-padded batches with visit masks
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from neurosight.data.cache import SessionCache
from neurosight.data.modes import apply_metadata_mask
from neurosight.data.preprocessing import preprocess_2d, preprocess_3d
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


def _item_rng(seed: int, index: int) -> np.random.RandomState:
    """Stable per-item RNG: same (seed, index) -> same augmentation forever."""
    return np.random.RandomState((seed * 1_000_003 + index) % (2**31 - 1))


def _load_volume(path: str) -> np.ndarray:
    """Load one scan file; any failure raises (fail-loud contract)."""
    image = nib.load(path)  # type: ignore[no-untyped-call]
    return np.asarray(image.get_fdata(), dtype=np.float32)  # type: ignore[attr-defined]


def _mri_id_from_paths(paths: list[str]) -> str:
    """``.../OAS2_XXXX_MRY/RAW/mpr-1.nifti.img`` -> ``OAS2_XXXX_MRY``."""
    return Path(paths[0]).parent.parent.name


class Oasis2DDataset(Dataset):
    """Sequence of visits as 2D slice stacks: (T, S, H, W) + (T, 8) meta + label."""

    def __init__(
        self,
        records: Sequence,
        input_mode: str = "mri_only",
        num_slices: int = 32,
        img_size: int = 128,
        slice_strategy: str = "middle",
        series_selection: str = "first",
        augment: bool = False,
        temporal_augment: bool = False,
        visit_drop_prob: float = 0.3,
        seed: int = 42,
        cache: SessionCache | None = None,
    ):
        self.records = list(records)
        self.input_mode = input_mode
        self.num_slices = num_slices
        self.img_size = img_size
        self.slice_strategy = slice_strategy
        self.series_selection = series_selection
        self.augment = augment
        self.temporal_augment = temporal_augment
        self.visit_drop_prob = visit_drop_prob
        self.seed = seed
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def _visit_stack(self, paths: list[str], rng: np.random.RandomState) -> np.ndarray:
        """One visit -> (S, H, W); multiple series (policy 'mean') are averaged."""
        mri_id = _mri_id_from_paths(paths)
        if self.cache is not None:
            cached = self.cache.get(
                mri_id,
                self.num_slices,
                self.img_size,
                self.slice_strategy,
                self.series_selection,
            )
            if cached is not None:
                # the cache stores the PRE-augmentation stack: a train-mode hit
                # must still be augmented (deterministically via the item rng),
                # or every cached training sample would be served unaugmented
                if not self.augment:
                    return cached
                from neurosight.data.preprocessing import augment_2d

                return augment_2d(np.asarray(cached, dtype=np.float32), rng)
        stacks = [
            preprocess_2d(
                _load_volume(p),
                self.num_slices,
                self.img_size,
                self.slice_strategy,
                self.augment,
                rng,
            )
            for p in paths
        ]
        return np.mean(stacks, axis=0) if len(stacks) > 1 else stacks[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        rng = _item_rng(self.seed, idx)

        visit_paths, visit_meta = rec.visit_paths, rec.visit_meta
        if self.temporal_augment and len(visit_paths) > 2 and rng.rand() < self.visit_drop_prob:
            drop = int(rng.randint(1, len(visit_paths)))  # never drop the first visit
            keep = [i for i in range(len(visit_paths)) if i != drop]
            visit_paths = [visit_paths[i] for i in keep]
            visit_meta = [visit_meta[i] for i in keep]

        scans = np.stack([self._visit_stack(paths, rng) for paths in visit_paths])
        meta = apply_metadata_mask(np.stack(visit_meta).astype(np.float32), self.input_mode)
        return (
            torch.from_numpy(np.ascontiguousarray(scans, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(meta)),
            torch.tensor(rec.label, dtype=torch.long),
        )


class Oasis3DDataset(Dataset):
    """3D volumes: mode 'single' (last visit) or 'sequential' (all visits)."""

    def __init__(
        self,
        records: Sequence,
        input_mode: str = "mri_only",
        mode: str = "single",
        vol_depth: int = 64,
        img_size: int = 128,
        slice_strategy: str = "middle",
        series_selection: str = "first",
        augment: bool = False,
        temporal_augment: bool = False,
        visit_drop_prob: float = 0.3,
        seed: int = 42,
        cache: SessionCache | None = None,
    ):
        if mode not in ("single", "sequential"):
            raise ValueError(f"3D dataset mode must be 'single' or 'sequential', got '{mode}'")
        self.records = list(records)
        self.input_mode = input_mode
        self.mode = mode
        self.vol_depth = vol_depth
        self.img_size = img_size
        self.slice_strategy = slice_strategy
        self.series_selection = series_selection
        self.augment = augment
        self.temporal_augment = temporal_augment
        self.visit_drop_prob = visit_drop_prob
        self.seed = seed
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def _volume(self, paths: list[str], rng: np.random.RandomState) -> np.ndarray:
        """One visit -> (D, H, W) volume, channelless, in BOTH paths.

        The 3D cache stores channelless volumes; the live preprocessing path
        must match (preprocess_3d returns (1, D, H, W), so strip the channel).
        Shape divergence here is what crashed the 10_3d_lstm overnight run.
        """
        mri_id = _mri_id_from_paths(paths)
        if self.cache is not None:
            # key on the CONFIGURED policies — the hardcoded "middle"/"first"
            # made every non-default 3D lookup miss and re-run preprocess_3d
            cached = self.cache.get(
                mri_id, self.vol_depth, self.img_size, self.slice_strategy, self.series_selection
            )
            if cached is not None:
                # cache stores the PRE-augmentation volume (see _visit_stack)
                if not self.augment:
                    return cached
                from neurosight.data.preprocessing import augment_3d

                return augment_3d(np.asarray(cached, dtype=np.float32), rng)
        vols = [
            preprocess_3d(
                _load_volume(p), (self.vol_depth, self.img_size, self.img_size), self.augment, rng
            )[0]
            for p in paths
        ]
        return np.mean(vols, axis=0) if len(vols) > 1 else vols[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        rng = _item_rng(self.seed, idx)
        visit_paths, visit_meta = rec.visit_paths, rec.visit_meta
        if self.mode == "single":
            vols = self._volume(visit_paths[-1], rng)[np.newaxis]  # (1, D, H, W)
            meta = visit_meta[-1]
            return (
                torch.from_numpy(np.ascontiguousarray(vols, dtype=np.float32)),
                torch.from_numpy(
                    apply_metadata_mask(meta.astype(np.float32)[np.newaxis], self.input_mode)
                ),
                torch.tensor(rec.label, dtype=torch.long),
            )
        if self.temporal_augment and len(visit_paths) > 2 and rng.rand() < self.visit_drop_prob:
            drop = int(rng.randint(1, len(visit_paths)))
            keep = [i for i in range(len(visit_paths)) if i != drop]
            visit_paths = [visit_paths[i] for i in keep]
            visit_meta = [visit_meta[i] for i in keep]
        vols = np.stack([self._volume(paths, rng) for paths in visit_paths])  # (T, 1, D, H, W)
        meta = apply_metadata_mask(np.stack(visit_meta).astype(np.float32), self.input_mode)
        return (
            torch.from_numpy(np.ascontiguousarray(vols, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(meta)),
            torch.tensor(rec.label, dtype=torch.long),
        )


# ── Collate functions ─────────────────────────────────────────────────────────


def collate_2d(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable T to the batch max. Mask: 1.0 = real visit, 0.0 = padding.

    The mask travels with the batch so temporal models can ignore padded
    timesteps — packed sequences and attention both consume it.
    """
    max_t = max(s.shape[0] for s, _, _ in batch)
    scans, metas, labels, masks = [], [], [], []
    for scans_t, meta_t, label in batch:
        t = scans_t.shape[0]
        pad_t = max_t - t
        if pad_t:
            scans_t = torch.nn.functional.pad(scans_t, (0, 0, 0, 0, 0, 0, 0, pad_t))
            meta_t = torch.nn.functional.pad(meta_t, (0, 0, 0, pad_t))
        masks.append(torch.cat([torch.ones(t), torch.zeros(pad_t)]))
        scans.append(scans_t)
        metas.append(meta_t)
        labels.append(label)
    return torch.stack(scans), torch.stack(metas), torch.stack(labels), torch.stack(masks)


def collate_3d_single(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vols = torch.stack([b[0] for b in batch])
    metas = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    ones = torch.ones(vols.shape[0], 1)
    return vols, metas, labels, ones


def collate_3d_seq(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_t = max(v.shape[0] for v, _, _ in batch)
    scans, metas, labels, masks = [], [], [], []
    for vols_t, meta_t, label in batch:
        t = vols_t.shape[0]
        pad_t = max_t - t
        if pad_t:
            vols_t = torch.cat([vols_t, torch.zeros(pad_t, *vols_t.shape[1:])], dim=0)
            meta_t = torch.nn.functional.pad(meta_t, (0, 0, 0, pad_t))
        masks.append(torch.cat([torch.ones(t), torch.zeros(pad_t)]))
        scans.append(vols_t)
        metas.append(meta_t)
        labels.append(label)
    return torch.stack(scans), torch.stack(metas), torch.stack(labels), torch.stack(masks)


def make_weighted_sampler(records: Sequence) -> WeightedRandomSampler:
    """Inverse-frequency sampling over sequence labels (imbalance handling)."""
    labels = np.array([r.label for r in records])
    if len(labels) == 0:
        raise ValueError("cannot build a weighted sampler from an empty record list")
    # map by label VALUE: np.unique's counts are positional, so weights[lbl]
    # broke on single-class folds (IndexError) and any label set not 0..k-1
    values, counts = np.unique(labels, return_counts=True)
    weight_of = {int(v): 1.0 / c for v, c in zip(values, counts, strict=True)}
    sample_weights = np.array([weight_of[int(lbl)] for lbl in labels])
    return WeightedRandomSampler(sample_weights.tolist(), len(labels), replacement=True)


def make_dataloaders(
    train_records: Sequence,
    val_records: Sequence,
    test_records: Sequence | None,
    family: str,
    data_config,
    training_config,
    input_mode: str = "mri_only",
    cache: SessionCache | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """DataLoaders for a model family ('2d' | '3d' | '3d_lstm') from configs.

    Batch sizes honour the RTX 4050 6 GB tuning: 3D families use
    ``batch_size_3d`` (the trainer applies gradient accumulation on top).
    """
    num_workers = training_config.num_workers
    train_ds: Dataset
    val_ds: Dataset
    test_ds: Dataset | None
    if family == "2d":
        common = {
            "input_mode": input_mode,
            "num_slices": data_config.num_slices,
            "img_size": data_config.slice_size,
            "slice_strategy": data_config.slice_strategy,
            "series_selection": data_config.series_selection,
            "seed": training_config.seed,
            "cache": cache,
        }
        train_ds = Oasis2DDataset(train_records, augment=True, temporal_augment=True, **common)
        val_ds = Oasis2DDataset(val_records, **common)
        test_ds = Oasis2DDataset(test_records, **common) if test_records is not None else None
        collate, batch_size = collate_2d, training_config.batch_size
    elif family == "3d":
        common = {
            "input_mode": input_mode,
            "mode": "single",
            "vol_depth": data_config.vol_depth,
            "img_size": data_config.slice_size,
            "slice_strategy": data_config.slice_strategy,
            "series_selection": data_config.series_selection,
            "seed": training_config.seed,
            "cache": cache,
        }
        train_ds = Oasis3DDataset(train_records, augment=True, **common)
        val_ds = Oasis3DDataset(val_records, **common)
        test_ds = Oasis3DDataset(test_records, **common) if test_records is not None else None
        collate, batch_size = collate_3d_single, training_config.batch_size_3d
    elif family == "3d_lstm":
        common = {
            "input_mode": input_mode,
            "mode": "sequential",
            "vol_depth": data_config.vol_depth,
            "img_size": data_config.slice_size,
            "slice_strategy": data_config.slice_strategy,
            "series_selection": data_config.series_selection,
            "seed": training_config.seed,
            "cache": cache,
        }
        train_ds = Oasis3DDataset(train_records, augment=True, temporal_augment=True, **common)
        val_ds = Oasis3DDataset(val_records, **common)
        test_ds = Oasis3DDataset(test_records, **common) if test_records is not None else None
        collate, batch_size = collate_3d_seq, training_config.batch_size_3d
    else:
        raise ValueError(f"unknown model family: {family}")

    sampler = make_weighted_sampler(train_records) if training_config.use_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader


__all__ = [
    "Oasis2DDataset",
    "Oasis3DDataset",
    "collate_2d",
    "collate_3d_seq",
    "collate_3d_single",
    "make_dataloaders",
    "make_weighted_sampler",
]
