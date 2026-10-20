"""Shared pytest fixtures. CI uses only synthetic fixtures — no real OASIS-2 data."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    "Subject ID",
    "MRI ID",
    "Group",
    "Visit",
    "MR Delay",
    "M/F",
    "Hand",
    "Age",
    "EDUC",
    "SES",
    "MMSE",
    "CDR",
    "eTIV",
    "nWBV",
    "ASF",
]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    return REPO_ROOT / "configs"


def make_canonical_df() -> pd.DataFrame:
    """Synthetic demographics CSV with the exact canonical OASIS-2 composition.

    150 subjects / 373 rows: row groups 190/146/37, subject groups 72/64/14.
    Converted subjects get CDR 0 before conversion and >0 after.
    """
    plan: list[tuple[str, int, int]] = []  # (group, n_visits, conversion_visit or 0)
    for _i in range(46):
        plan.append(("Nondemented", 3, 0))
    for _i in range(26):
        plan.append(("Nondemented", 2, 0))
    for _i in range(18):
        plan.append(("Demented", 3, 0))
    for _i in range(46):
        plan.append(("Demented", 2, 0))
    # Converted: alternate pre-visit patterns; conversions land on visit 2 or 3
    conv_specs = [(3, 3), (3, 2)] * 4 + [(3, 3)] + [(2, 2)] * 5  # 9x3-visit + 5x2-visit
    for visits, conv in conv_specs:
        plan.append(("Converted", visits, conv))

    rng = np.random.RandomState(0)
    rows: list[dict] = []
    for si, (group, n_visits, conv_visit) in enumerate(plan, start=1):
        subject = f"OAS2_{si:04d}"
        for visit in range(1, n_visits + 1):
            if group == "Converted":
                cdr = 0.0 if visit < conv_visit else (0.5 if visit == conv_visit else 1.0)
            elif group == "Demented":
                cdr = 0.5 if visit % 2 else 1.0
            else:
                cdr = 0.0
            rows.append(
                {
                    "Subject ID": subject,
                    "MRI ID": f"{subject}_MR{visit}",
                    "Group": group,
                    "Visit": visit,
                    "MR Delay": 0 if visit == 1 else 365 * (visit - 1) + int(rng.randint(0, 60)),
                    "M/F": "M" if si % 2 else "F",
                    "Hand": "R",
                    "Age": int(60 + (si % 37)),
                    "EDUC": int(12 + (si % 12)),
                    "SES": 1.0 + (si % 5),
                    "MMSE": 30.0 - cdr * 6,
                    "CDR": cdr,
                    "eTIV": float(1400 + (si * 7) % 600),
                    "nWBV": float(0.70 + (si % 10) / 100),
                    "ASF": float(0.9 + (si % 8) / 40),
                }
            )
    df = pd.DataFrame(rows)
    # NaN counts are per ROW in the canonical file: 19 SES rows, 2 MMSE rows
    df.loc[df.index[:19], "SES"] = np.nan
    df.loc[df.index[:2], "MMSE"] = np.nan
    assert len(df) == 373 and df["Subject ID"].nunique() == 150
    assert df["Group"].value_counts().to_dict() == {
        "Nondemented": 190,
        "Demented": 146,
        "Converted": 37,
    }
    return df[COLUMNS]


def write_raw_tree(
    df: pd.DataFrame, root: Path, extra_series_for: set[str] | None = None
) -> list[Path]:
    """Create fake OAS2_RAW_PARTn trees: RAW/mpr-1.nifti.img+.hdr per session.

    Returns the list of part roots. ``extra_series_for`` adds an mpr-2 pair to
    those MRI IDs (for series-policy tests). Files are empty — the loader only
    resolves paths; it never opens scans.
    """
    extra_series_for = extra_series_for or set()
    roots = [root / "OAS2_RAW_PART1", root / "OAS2_RAW_PART2"]
    for i, mri_id in enumerate(sorted(df["MRI ID"].astype(str))):
        part = roots[i % 2]
        raw_dir = part / mri_id / "RAW"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "mpr-1.nifti.img").touch()
        (raw_dir / "mpr-1.nifti.hdr").touch()
        if mri_id in extra_series_for:
            (raw_dir / "mpr-2.nifti.img").touch()
            (raw_dir / "mpr-2.nifti.hdr").touch()
    # Noise that the audit must ignore (only when the canonical session exists)
    legacy = roots[0] / "OAS2_0001_MR1"
    if legacy.is_dir():
        (legacy / "OLD").mkdir(exist_ok=True)
        (legacy / "OLD" / "mpr-1_to_711-2C_t4").touch()
        (legacy / "3906-3_to_mpr-1.log").touch()
    return roots


def make_mini_df() -> pd.DataFrame:
    """Six subjects exercising every loader branch (Converted splits, exclusions).

    N01: Nondemented, 2 visits            -> full, label 0
    N02: Nondemented, 3 visits            -> full, label 0
    D01: Demented,   2 visits             -> full, label 1
    C01: Converted,  4 visits (0,0,0,0.5) -> pre (3 visits; 1 visit's raw folder is
         deleted by the test -> still 2 usable, label 0); post 1 visit excluded
    C02: Converted,  3 visits (0,0.5,1.0) -> pre 1 visit excluded; post (2 visits, 1)
    C03: Converted,  4 visits (0,0,0.5,1) -> pre (2, 0) AND post (2, 1) — straddle case
    """

    def row(subject: str, visit: int, group: str, cdr: float) -> dict:
        return {
            "Subject ID": subject,
            "MRI ID": f"{subject}_MR{visit}",
            "Group": group,
            "Visit": visit,
            "MR Delay": 365 * (visit - 1),
            "M/F": "M",
            "Hand": "R",
            "Age": 70,
            "EDUC": 16,
            "SES": 2.0,
            "MMSE": 30.0 - cdr * 6,
            "CDR": cdr,
            "eTIV": 1500.0,
            "nWBV": 0.73,
            "ASF": 1.2,
        }

    rows = [
        row("N01", 1, "Nondemented", 0.0),
        row("N01", 2, "Nondemented", 0.0),
        row("N02", 1, "Nondemented", 0.0),
        row("N02", 2, "Nondemented", 0.0),
        row("N02", 3, "Nondemented", 0.0),
        row("D01", 1, "Demented", 0.5),
        row("D01", 2, "Demented", 1.0),
        row("C01", 1, "Converted", 0.0),
        row("C01", 2, "Converted", 0.0),
        row("C01", 3, "Converted", 0.0),
        row("C01", 4, "Converted", 0.5),
        row("C02", 1, "Converted", 0.0),
        row("C02", 2, "Converted", 0.5),
        row("C02", 3, "Converted", 1.0),
        row("C03", 1, "Converted", 0.0),
        row("C03", 2, "Converted", 0.0),
        row("C03", 3, "Converted", 0.5),
        row("C03", 4, "Converted", 1.0),
    ]
    return pd.DataFrame(rows)[COLUMNS]


def make_study_records(tmp_path: Path) -> list:
    """Two subjects (N01: 3 visits label 0, D01: 2 visits label 1) as real tiny
    NIfTI studies — used by dataset and training tests."""
    import nibabel as nib

    from neurosight.data.loader import SequenceRecord

    records = []
    for subject, n_visits, label in (("N01", 3, 0), ("D01", 2, 1)):
        paths_per_visit, metas, delays, visit_labels = [], [], [], []
        for v in range(1, n_visits + 1):
            mri_id = f"{subject}_MR{v}"
            raw = tmp_path / mri_id / "RAW"
            raw.mkdir(parents=True, exist_ok=True)
            seed = zlib.crc32(mri_id.encode())  # stable digest: hash() is salted per process
            vol = np.random.RandomState(seed).rand(32, 32, 48).astype(np.float32) * 100
            path = raw / "mpr-1.nii"
            nib.save(nib.Nifti1Image(vol, np.eye(4)), path)
            paths_per_visit.append([str(path)])
            metas.append(np.linspace(0.1, 0.8, 8, dtype=np.float32))
            delays.append(365 * (v - 1))
            visit_labels.append(label)
        records.append(
            SequenceRecord(
                subject_id=subject,
                kind="full",
                visit_paths=paths_per_visit,
                visit_meta=metas,
                visit_labels=visit_labels,
                visit_delays=delays,
                label=label,
            )
        )
    return records
