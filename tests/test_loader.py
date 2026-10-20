"""Tests for the leakage-safe OASIS-2 loader (plan T1.4).

The core guarantee under test: **subject-level splits happen before Converted
sequence splitting**, so no subject — and never a Converted subject's pre/post
pair — appears in two splits.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import make_canonical_df, make_mini_df, write_raw_tree
from neurosight.config import DataConfig
from neurosight.data.loader import OASIS2Loader
from neurosight.data.modes import apply_metadata_mask


def make_config(tmp_path: Path, raw_paths: list[Path], **overrides) -> DataConfig:
    return DataConfig(
        csv_path=tmp_path / "mini.csv",
        raw_paths=[str(p) for p in raw_paths],
        processed_dir=tmp_path / "processed",
        uploads_dir=tmp_path / "uploads",
        **overrides,
    )


@pytest.fixture()
def mini_loader(tmp_path: Path) -> OASIS2Loader:
    df = make_mini_df()
    roots = write_raw_tree(df, tmp_path)
    # C01 visit 1 loses its raw folder -> visit dropped, pre-conversion still usable
    for root in roots:
        session = root / "C01_MR1"
        if session.is_dir():
            shutil.rmtree(session)
    return OASIS2Loader(make_config(tmp_path, roots), df=df, raw_paths=roots)


def test_mini_loader_sequence_structure(mini_loader: OASIS2Loader) -> None:
    seqs = {(s.subject_id, s.kind): s for s in mini_loader.sequences}
    # Survivors: N01 full, N02 full, D01 full, C01 pre, C02 post, C03 pre + post
    assert set(seqs) == {
        ("N01", "full"),
        ("N02", "full"),
        ("D01", "full"),
        ("C01", "pre"),
        ("C02", "post"),
        ("C03", "pre"),
        ("C03", "post"),
    }
    assert seqs[("N01", "full")].label == 0 and seqs[("N01", "full")].n_visits == 2
    assert seqs[("D01", "full")].label == 1
    assert seqs[("C01", "pre")].label == 0 and seqs[("C01", "pre")].n_visits == 2
    assert seqs[("C02", "post")].label == 1 and seqs[("C02", "post")].n_visits == 2
    assert seqs[("C03", "pre")].label == 0 and seqs[("C03", "post")].label == 1


def test_converted_straddle_never_separated(mini_loader: OASIS2Loader) -> None:
    splits = mini_loader.split_sequences()
    where: dict[str, set[str]] = {}
    for split_name, seqs in splits.items():
        for s in seqs:
            where.setdefault(s.subject_id, set()).add(split_name)
    for subject, split_names in where.items():
        assert len(split_names) == 1, f"{subject} straddles splits: {split_names}"


def test_exclusions_are_counted_not_silent(mini_loader: OASIS2Loader) -> None:
    reasons = {e["reason"] for e in mini_loader.exclusions}
    # C01 post = 1 visit; C02 pre = 1 visit
    assert "converted_post_too_short" in reasons
    assert "converted_pre_too_short" in reasons
    assert mini_loader.summary()["exclusions"] >= 2


def test_series_policy_first_middle_mean(tmp_path: Path) -> None:
    df = make_mini_df()
    roots = write_raw_tree(df, tmp_path, extra_series_for={"N02_MR2"})

    first = OASIS2Loader(
        make_config(tmp_path, roots, series_selection="first"), df=df, raw_paths=roots
    )
    n02_first = next(s for s in first.sequences if s.subject_id == "N02").visit_paths[1]
    assert len(n02_first) == 1 and n02_first[0].endswith("mpr-1.nifti.img")
    assert Path(n02_first[0]).exists()  # resolved path must be the real file

    mean = OASIS2Loader(
        make_config(tmp_path, roots, series_selection="mean"), df=df, raw_paths=roots
    )
    n02_mean = next(s for s in mean.sequences if s.subject_id == "N02").visit_paths[1]
    assert len(n02_mean) == 2

    middle = OASIS2Loader(
        make_config(tmp_path, roots, series_selection="middle"), df=df, raw_paths=roots
    )
    n02_middle = next(s for s in middle.sequences if s.subject_id == "N02").visit_paths[1]
    assert len(n02_middle) == 1 and n02_middle[0].endswith(
        "mpr-2.nifti.img"
    )  # middle of [mpr-1, mpr-2]


def test_metadata_imputation_and_mode_masks(mini_loader: OASIS2Loader) -> None:
    n01 = next(s for s in mini_loader.sequences if s.subject_id == "N01")
    meta = n01.visit_meta[0]
    assert meta.shape == (8,)
    assert meta.dtype == np.float32
    # SES=2.0 in the fixture -> dim 3 == 2/5
    assert meta[3] == pytest.approx(2.0 / 5.0)
    # mri_only masks everything; mri_demo keeps demo+MRI-derived but zeroes MMSE (dim 4)
    assert np.all(apply_metadata_mask(meta, "mri_only") == 0)
    demo = apply_metadata_mask(meta, "mri_demo")
    assert demo[:4].sum() > 0 and demo[4] == 0 and demo[5:].sum() > 0
    clinical = apply_metadata_mask(meta, "mri_clinical")
    assert clinical[4] > 0  # MMSE visible only in the explicit clinical mode


def test_canonical_split_disjoint_and_stratified(tmp_path: Path) -> None:
    df = make_canonical_df()
    roots = write_raw_tree(df, tmp_path)
    loader = OASIS2Loader(make_config(tmp_path, roots), df=df, raw_paths=roots)
    splits = loader.split_sequences()

    subj = {name: {s.subject_id for s in seqs} for name, seqs in splits.items()}
    assert not (subj["train"] & subj["val"]), "train/val subject overlap = leakage"
    assert not (subj["train"] & subj["test"]), "train/test subject overlap = leakage"
    assert not (subj["val"] & subj["test"]), "val/test subject overlap = leakage"
    # Every subject — including each Converted pre/post pair — lives in exactly one split
    by_subject: dict[str, set[str]] = {}
    for name, seqs in splits.items():
        for s in seqs:
            by_subject.setdefault(s.subject_id, set()).add(name)
    straddlers = {k: v for k, v in by_subject.items() if len(v) > 1}
    assert not straddlers, f"subjects straddling splits: {straddlers}"
    # Split sizes respect the 70/15/15 ratios at subject level
    total = sum(len(v) for v in subj.values())
    assert len(subj["test"]) / total == pytest.approx(0.15, abs=0.03)


def test_cv_folds_disjoint_and_complete(tmp_path: Path) -> None:
    df = make_canonical_df()
    roots = write_raw_tree(df, tmp_path)
    loader = OASIS2Loader(make_config(tmp_path, roots), df=df, raw_paths=roots)
    folds = loader.cv_folds(5)
    assert len(folds) == 5
    all_val_subjects: set[str] = set()
    for train, val in folds:
        train_subjects = {s.subject_id for s in train}
        val_subjects = {s.subject_id for s in val}
        assert not (train_subjects & val_subjects), "fold leakage: subject in train and val"
        assert not (all_val_subjects & val_subjects), "subject repeated across validation folds"
        all_val_subjects |= val_subjects
    expected = {s.subject_id for s in loader.sequences}
    assert all_val_subjects == expected  # every subject is validated exactly once per CV round


def test_nan_safety(tmp_path: Path) -> None:
    df = make_mini_df()
    df.loc[df["MRI ID"] == "N01_MR1", "CDR"] = np.nan
    df.loc[df["MRI ID"] == "N01_MR1", "MR Delay"] = np.nan
    roots = write_raw_tree(df, tmp_path)
    loader = OASIS2Loader(make_config(tmp_path, roots), df=df, raw_paths=roots)
    n01 = next(s for s in loader.sequences if s.subject_id == "N01")
    assert n01.visit_labels == [0, 0]  # NaN CDR -> Group fallback, not a spurious 1
    assert n01.visit_delays[0] == 0  # NaN delay -> 0, not a crash


def test_label_mode_not_implemented_yet(tmp_path: Path) -> None:
    df = make_mini_df()
    roots = write_raw_tree(df, tmp_path)
    with pytest.raises(NotImplementedError, match="per_visit"):
        OASIS2Loader(make_config(tmp_path, roots, label_mode="per_visit"), df=df, raw_paths=roots)


def test_unpaired_img_orphan_is_skipped(tmp_path: Path) -> None:
    """An .img without its .hdr twin (real case: OAS2_0056_MR2/mpr-1) is not a
    selection candidate — the policy falls through to the next paired series."""
    df = make_mini_df()
    roots = write_raw_tree(df, tmp_path, extra_series_for={"D01_MR1"})
    for root in roots:
        orphan = root / "D01_MR1" / "RAW" / "mpr-1.nifti.hdr"
        if orphan.exists():
            orphan.unlink()
    loader = OASIS2Loader(make_config(tmp_path, roots), df=df, raw_paths=roots)
    d01 = next(s for s in loader.sequences if s.subject_id == "D01")
    assert d01.visit_paths[0][0].endswith("mpr-2.nifti.img")
    assert d01.n_visits == 2


def test_min_visits_exclusion(tmp_path: Path) -> None:
    df = make_mini_df()
    df = df[
        df["Subject ID"] != "C03"
    ]  # N01 keeps 2 visits: removing one raw folder excludes the subject
    roots = write_raw_tree(df, tmp_path)
    for root in roots:
        session = root / "N01_MR2"
        if session.is_dir():
            shutil.rmtree(session)
    loader = OASIS2Loader(make_config(tmp_path, roots), df=df, raw_paths=roots)
    assert not any(s.subject_id == "N01" for s in loader.sequences)
    assert any(
        e["subject_id"] == "N01" and e["reason"] == "fewer_than_min_visits"
        for e in loader.exclusions
    )


def test_series_selection_middle_with_no_series_returns_empty(tmp_path: Path) -> None:
    """A RAW folder with no paired mpr series must resolve to [] under the
    'middle' policy too — the old code indexed the empty list (IndexError) and
    OASIS2Loader construction crashed instead of excluding the visit."""
    df = make_mini_df()
    roots = write_raw_tree(df, tmp_path)
    for root in roots:
        raw = root / "N01_MR1" / "RAW"
        if raw.is_dir():
            for f in raw.iterdir():
                f.unlink()
    # construction itself must succeed — the old code IndexError'd in _build
    loader = OASIS2Loader(
        make_config(tmp_path, roots, series_selection="middle"), df=df, raw_paths=roots
    )
    assert not [s for s in loader.sequences if s.subject_id == "N01"]
    assert any(
        e["subject_id"] == "N01" and e["reason"] == "fewer_than_min_visits"
        for e in loader.exclusions
    )
    assert {s.subject_id for s in loader.sequences} >= {"N02", "D01"}
