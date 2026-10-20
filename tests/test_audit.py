"""Tests for the CSV schema validator and raw-dataset audit (plan T1.2, T1.3)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import make_canonical_df, write_raw_tree
from neurosight.data.audit import CsvAuditError, audit_raw, validate_csv


@pytest.fixture()
def canonical_csv(tmp_path: Path) -> Path:
    path = tmp_path / "oasis_longitudinal.csv"
    make_canonical_df().to_csv(path, index=False, encoding="utf-8")
    return path


def test_validate_canonical_fixture(canonical_csv: Path) -> None:
    report = validate_csv(canonical_csv)
    assert report["rows"] == 373
    assert report["unique_subjects"] == 150
    assert report["row_group_counts"] == {"Nondemented": 190, "Demented": 146, "Converted": 37}
    assert report["subject_group_counts"] == {"Nondemented": 72, "Demented": 64, "Converted": 14}
    assert report["nan_counts"] == {"MMSE": 2, "SES": 19}
    assert report["sha256"] and len(report["sha256"]) == 64


def test_missing_column_lists_it(canonical_csv: Path) -> None:
    df = make_canonical_df().drop(columns=["CDR"])
    with pytest.raises(CsvAuditError, match=r"missing columns.*CDR"):
        validate_csv(canonical_csv, df=df)


def test_renamed_column_rejected(canonical_csv: Path) -> None:
    df = make_canonical_df().rename(columns={"Subject ID": "SubjectID"})
    with pytest.raises(CsvAuditError, match="missing columns"):
        validate_csv(canonical_csv, df=df)


def test_wrong_row_count_rejected(canonical_csv: Path) -> None:
    df = make_canonical_df().iloc[:-1]
    with pytest.raises(CsvAuditError, match="row count is 372"):
        validate_csv(canonical_csv, df=df)


def test_corrupted_cdr_rejected(canonical_csv: Path) -> None:
    df = make_canonical_df()
    df.loc[0, "CDR"] = 3.0
    with pytest.raises(CsvAuditError, match="CDR values outside"):
        validate_csv(canonical_csv, df=df)


def test_unknown_group_rejected(canonical_csv: Path) -> None:
    df = make_canonical_df()
    df.loc[0, "Group"] = "MildImpaired"
    with pytest.raises(CsvAuditError, match="unknown Group values"):
        validate_csv(canonical_csv, df=df)


def test_duplicate_mri_id_rejected(canonical_csv: Path) -> None:
    df = make_canonical_df()
    df.loc[1, "MRI ID"] = df.loc[0, "MRI ID"]
    with pytest.raises(CsvAuditError, match="duplicate MRI IDs"):
        validate_csv(canonical_csv, df=df)


def test_nan_count_deviation_rejected(canonical_csv: Path) -> None:
    df = make_canonical_df()
    df.loc[0, "MMSE"] = 29.0  # first NaN row filled -> NaN count 2 -> 1
    with pytest.raises(CsvAuditError, match="MMSE NaN count"):
        validate_csv(canonical_csv, df=df)


def test_audit_raw_synthetic(tmp_path: Path) -> None:
    df = make_canonical_df()
    roots = write_raw_tree(df, tmp_path)
    # Drop 3 sessions entirely -> missing_scan; strip one .hdr -> empty_on_disk;
    # add one folder absent from the CSV -> extra_on_disk.
    dropped = sorted(df["MRI ID"].astype(str))[:3]
    for mri_id in dropped:
        for root in roots:
            session = root / mri_id
            if session.is_dir():
                shutil.rmtree(session)
    victim = sorted(df["MRI ID"].astype(str))[10]
    for root in roots:
        hdr = root / victim / "RAW" / "mpr-1.nifti.hdr"
        if hdr.exists():
            hdr.unlink()
    extra = roots[0] / "OAS2_9999_MR1" / "RAW"
    extra.mkdir(parents=True)
    (extra / "mpr-1.nifti.img").touch()
    (extra / "mpr-1.nifti.hdr").touch()

    out = tmp_path / "reports" / "data_audit.json"
    report = audit_raw(df, roots, out_path=out, min_matched=360)
    assert report["csv_sessions"] == 373
    assert report["missing_scan"] == dropped
    assert report["empty_on_disk"] == [victim]
    assert report["extra_on_disk"] == ["OAS2_9999_MR1"]
    # The stripped .hdr leaves one img-only series on the victim session
    assert report["unpaired_files"] == {victim: ["mpr-1"]}
    assert report["usable_with_series"] == 369
    assert report["series_per_session_hist"] == {"0": 1, "1": 369}
    assert json.loads(out.read_text(encoding="utf-8"))["matched"] == 370


def test_audit_raw_hard_floor(tmp_path: Path) -> None:
    df = make_canonical_df().head(10)
    roots = write_raw_tree(df, tmp_path)
    with pytest.raises(CsvAuditError, match="floor"):
        audit_raw(df, roots, min_matched=360)
