"""cache.py — per-session preprocessed-tensor cache (plan T2.2).

Replaces the predecessor's 1.08 GB monolithic ``dataset_cache.pt`` with one
compressed ``.npz`` per MRI session plus a manifest. Cache entries are keyed by
the preprocessing parameters that produced them: a policy/strategy/size change
is a miss, never stale data served as fresh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"


class SessionCache:
    """Read/write preprocessed sessions under ``processed_dir/<subject>/<mri_id>.<mode>.npz``."""

    def __init__(self, processed_dir: str | Path, mode: str = "2d"):
        if mode not in ("2d", "3d"):
            raise ValueError(f"cache mode must be '2d' or '3d', got '{mode}'")
        self.dir = Path(processed_dir)
        self.mode = mode
        # One manifest per mode: a shared manifest made the 3D cache see the 2D
        # entries as its own and skip building entirely (and let the two cache
        # objects clobber each other's writes).
        self.manifest_path = self.dir / f"manifest.{mode}.json"
        legacy = self.dir / MANIFEST_NAME
        if not self.manifest_path.exists() and legacy.exists():
            # one-time migration: the pre-split manifest holds 2D entries
            legacy_manifest = json.loads(legacy.read_text(encoding="utf-8"))
            entries = {
                k: v
                for k, v in legacy_manifest.get("sessions", {}).items()
                if v.get("mode") == self.mode
            }
            self.manifest = {"sessions": entries}
            self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        elif self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"sessions": {}}
        self.hits = 0
        self.misses = 0

    def _entry(self, mri_id: str) -> dict | None:
        entry = self.manifest["sessions"].get(mri_id)
        if entry is not None and entry.get("mode") != self.mode:
            return None
        return entry

    def contains(self, mri_id: str) -> bool:
        return self._entry(mri_id) is not None

    def get(
        self,
        mri_id: str,
        num_slices: int,
        img_size: int,
        slice_strategy: str,
        series_selection: str,
    ) -> np.ndarray | None:
        """Return the cached volume/stack, or None on miss or parameter mismatch."""
        entry = self._entry(mri_id)
        if entry is None or entry.get("mode") != self.mode:
            return None
        params_match = (
            entry.get("num_slices") == num_slices
            and entry.get("img_size") == img_size
            and entry.get("slice_strategy") == slice_strategy
            and entry.get("series_selection") == series_selection
        )
        if not params_match:
            return None
        path = self.dir / entry["file"]
        if not path.exists():
            return None
        with np.load(path) as archive:
            self.hits += 1
            return archive["vol"]

    def put(
        self,
        mri_id: str,
        subject_id: str,
        volume: np.ndarray,
        num_slices: int,
        img_size: int,
        slice_strategy: str,
        series_selection: str,
    ) -> Path:
        """Persist one session and update the manifest atomically.

        The filename encodes the producing parameters, so runs with different
        slice counts/sizes coexist instead of silently overwriting each other.
        """
        subject_dir = self.dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        rel = Path(subject_id) / (
            f"{mri_id}.{self.mode}.s{num_slices}.{img_size}.{slice_strategy}.npz"
        )
        np.savez_compressed(self.dir / rel, vol=volume.astype(np.float32))
        self.manifest["sessions"][mri_id] = {
            "file": str(rel),
            "subject": subject_id,
            "mode": self.mode,
            "shape": list(volume.shape),
            "num_slices": num_slices,
            "img_size": img_size,
            "slice_strategy": slice_strategy,
            "series_selection": series_selection,
        }
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return self.dir / rel

    def stats(self) -> dict:
        entries = self.manifest["sessions"]
        return {
            "mode": self.mode,
            "sessions": len(entries),
            "by_subject": len({e["subject"] for e in entries.values()}),
        }


def build_session_cache(
    loader,
    processed_dir: str | Path,
    num_slices: int = 32,
    img_size: int = 128,
    slice_strategy: str = "middle",
    series_selection: str | None = None,
    vol_depth: int = 64,
    include_3d: bool = False,
) -> dict:
    """Preprocess every session of a fitted :class:`OASIS2Loader` into the cache.

    Fails loud on any unreadable session (VolumeLoadError propagates) —
    preprocessing is where corrupt data must stop the pipeline.
    """
    import nibabel as nib

    from neurosight.data.preprocessing import preprocess_2d, preprocess_3d

    policy = series_selection or loader.config.series_selection
    cache_2d = SessionCache(processed_dir, mode="2d")
    cache_3d = SessionCache(processed_dir, mode="3d") if include_3d else None

    done = skipped = 0
    sessions = list(loader.iter_sessions())
    for i, (mri_id, subject_id, paths) in enumerate(sessions, start=1):
        cached_2d = cache_2d.contains(mri_id)
        cached_3d = cache_3d is not None and cache_3d.contains(mri_id)
        if cached_2d and (cache_3d is None or cached_3d):
            skipped += 1
            continue
        # nibabel stubs don't declare get_fdata; the loaded object always has it
        volumes = [
            nib.load(p).get_fdata().astype(np.float32) for p in paths  # type: ignore[attr-defined]
        ]
        stack_2d = np.mean(
            [preprocess_2d(v, num_slices, img_size, slice_strategy) for v in volumes], axis=0
        )
        cache_2d.put(mri_id, subject_id, stack_2d, num_slices, img_size, slice_strategy, policy)
        if cache_3d is not None:
            vol_3d = np.mean(
                [preprocess_3d(v, target=(vol_depth, img_size, img_size)) for v in volumes], axis=0
            )
            # 3D entries key on vol_depth (the depth dimension) so the dataset's
            # cache lookup (which passes vol_depth) matches the write
            cache_3d.put(mri_id, subject_id, vol_3d[0], vol_depth, img_size, slice_strategy, policy)
        done += 1
        if i % 25 == 0 or i == len(sessions):
            logger.info("cache build %d/%d sessions", i, len(sessions))

    stats = {"processed": done, "skipped_cached": skipped, "2d": cache_2d.stats()}
    if cache_3d is not None:
        stats["3d"] = cache_3d.stats()
    return stats


__all__ = ["SessionCache", "build_session_cache"]
