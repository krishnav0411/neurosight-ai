"""Smoke test for the bounded hyperparameter search.

Regression guard: run_search crashed twice on its first real overnight run
(a root NeuroSightConfig was passed where the training section was expected).
This test exercises the full path on CPU with a tiny fixture in seconds.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from conftest import make_mini_df, write_raw_tree
from neurosight.config import DataConfig, NeuroSightConfig, load_config
from neurosight.data.loader import OASIS2Loader
from neurosight.training.search import run_search


def _zlib_crc(text: str) -> int:
    return zlib.crc32(text.encode())


@pytest.fixture()
def search_setup(configs_dir: Path, tmp_path: Path, monkeypatch):
    cfg: NeuroSightConfig = load_config(configs_dir)
    training = cfg.training.model_copy(
        update={"epochs": 1, "use_mlflow": False, "use_sampler": False}
    )
    cfg = cfg.model_copy(update={"training": training})

    df = make_mini_df()
    roots = write_raw_tree(df, tmp_path / "raw")
    # real loadable volumes beside the name-level placeholders — the "first"
    # series policy sorts mpr-1 < mpr-1.nifti and picks the loadable .nii
    for root in roots:
        for session in root.iterdir():
            raw_dir = session / "RAW"
            if raw_dir.is_dir():
                rng_v = np.random.RandomState(_zlib_crc(session.name))
                vol = rng_v.rand(32, 32, 48).astype(np.float32)
                nib.save(nib.Nifti1Image(vol, np.eye(4)), raw_dir / "mpr-1.nii")

    data_cfg = DataConfig(
        csv_path=tmp_path / "mini.csv",
        raw_paths=[str(r) for r in roots],
        processed_dir=tmp_path / "processed",
        uploads_dir=tmp_path / "uploads",
    )
    loader = OASIS2Loader(data_cfg, df=df, raw_paths=roots)
    monkeypatch.chdir(tmp_path)  # run_search writes reports/ relative to cwd
    return cfg, loader, tmp_path


def test_run_search_smoke_end_to_end(search_setup) -> None:
    cfg, loader, tmp_path = search_setup
    results = run_search(
        cfg,
        loader,
        "3_cnn_lstm",
        grid_override={"learning_rate": [3e-4], "dropout": [0.5], "lstm_hidden": [32]},
        epochs=1,
    )
    assert len(results) == 1
    assert 0.0 <= results[0]["mean_auc"] <= 1.0
    assert (tmp_path / "reports" / "csv" / "search_3_cnn_lstm_mri_only.json").exists()
