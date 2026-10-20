"""audit.py — CSV schema validation and raw-dataset integrity audit (plan T1.2, T1.3).

Two hard gates run before any training:

1. :func:`validate_csv` — the demographics CSV must match the canonical OASIS-2
   schema and the reference distributions measured on 2026-09-14 (see DATA.md).
2. :func:`audit_raw` — every CSV session must resolve to a raw scan folder and
   vice versa; the result is written to ``reports/data_audit.json``.

Nothing downstream may run on un-audited data: silent drop of sessions is the
failure mode that corrupted the predecessor project's numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
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
VALID_GROUPS = {"Nondemented", "Demented", "Converted"}
VALID_CDR = {0.0, 0.5, 1.0, 2.0}
SESSION_ID_RE = re.compile(r"^OAS2_\d{4}_MR\d+$")

# Reference distributions of the canonical OASIS-2 demographics file
# (measured and recorded in DATA.md; a deviation means the wrong/edited file).
EXPECTED_ROWS = 373
ROW_GROUP_COUNTS = {"Nondemented": 190, "Demented": 146, "Converted": 37}
SUBJECT_GROUP_COUNTS = {"Nondemented": 72, "Demented": 64, "Converted": 14}
EXPECTED_SUBJECTS = 150
EXPECTED_NAN = {"MMSE": 2, "SES": 19}

# Floor for matched sessions: 373 on disk vs 373 in CSV as of 2026-09-14; a big
# loss would signal a broken dataset mount, not a normal missing-file case.
MIN_MATCHED_SESSIONS = 360

# Files that live in session folders but are not scan series (measured on disk)
_IGNORED_FILE_RE = re.compile(r"\.(log|ima\.gz)$", re.IGNORECASE)


class CsvAuditError(ValueError):
    """Raised when the demographics CSV fails schema or distribution checks."""


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_csv(csv_path: str | Path, df: pd.DataFrame | None = None) -> dict:
    """Validate the demographics CSV against the canonical OASIS-2 reference.

    Args:
        csv_path: path used for error messages and checksum (required even when
            ``df`` is supplied, so reports always name their source).
        df: optional pre-loaded dataframe (tests inject synthetic CSVs); when
            None the CSV is read from ``csv_path`` with ``encoding="latin1"``.

    Returns a report dict. Raises :class:`CsvAuditError` on any violation.
    """
    path = Path(csv_path)
    if df is None:
        if not path.exists():
            raise FileNotFoundError(f"demographics CSV not found: {path}")
        df = pd.read_csv(path, encoding="latin1")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise CsvAuditError(f"{path}: missing columns {missing}; found {list(df.columns)}")

    errors: list[str] = []

    # Row count and group distributions (exact — the dataset is fixed)
    if len(df) != EXPECTED_ROWS:
        errors.append(f"row count is {len(df)}, expected {EXPECTED_ROWS}")
    row_counts = df["Group"].value_counts().to_dict()
    for group, expected in ROW_GROUP_COUNTS.items():
        actual = row_counts.get(group, 0)
        if actual != expected:
            errors.append(f"row-level Group '{group}' = {actual}, expected {expected}")
    if set(row_counts) - VALID_GROUPS:
        errors.append(f"unknown Group values: {sorted(set(row_counts) - VALID_GROUPS)}")

    subj_groups = df.groupby("Subject ID")["Group"].first()
    if df["Subject ID"].nunique() != EXPECTED_SUBJECTS:
        errors.append(
            f"unique subjects = {df['Subject ID'].nunique()}, expected {EXPECTED_SUBJECTS}"
        )
    subj_counts = subj_groups.value_counts().to_dict()
    for group, expected in SUBJECT_GROUP_COUNTS.items():
        if subj_counts.get(group, 0) != expected:
            errors.append(
                f"subject-level Group '{group}' = {subj_counts.get(group, 0)}, expected {expected}"
            )

    # MRI ID must uniquely identify sessions (the join key to raw folders)
    dup = df["MRI ID"].astype(str).str.strip()
    if dup.duplicated().any():
        errors.append(f"duplicate MRI IDs: {sorted(dup[dup.duplicated()].unique())}")

    # Domain checks
    bad_cdr = df.loc[~df["CDR"].isin(VALID_CDR) & df["CDR"].notna(), "CDR"].unique()
    if len(bad_cdr):
        errors.append(f"CDR values outside {sorted(VALID_CDR)}: {sorted(bad_cdr)}")
    if (df["Visit"] < 1).any():
        errors.append("Visit contains values < 1")
    bad_group = df.loc[~df["Group"].isin(VALID_GROUPS), "Group"].unique()
    if len(bad_group):
        errors.append(f"Group values outside {sorted(VALID_GROUPS)}: {sorted(bad_group)}")

    nan_counts = {c: int(df[c].isna().sum()) for c in ("MMSE", "SES")}
    for col, expected in EXPECTED_NAN.items():
        if nan_counts[col] != expected:
            errors.append(f"{col} NaN count = {nan_counts[col]}, expected {expected}")

    if errors:
        raise CsvAuditError(f"{path}: {len(errors)} violation(s):\n  - " + "\n  - ".join(errors))

    report = {
        "source": str(path),
        "sha256": file_sha256(path) if path.exists() else None,
        "rows": len(df),
        "columns": list(df.columns),
        "row_group_counts": row_counts,
        "subject_group_counts": subj_counts,
        "unique_subjects": int(df["Subject ID"].nunique()),
        "nan_counts": nan_counts,
    }
    logger.info(
        "CSV validated: %s rows, subjects=%s, groups(row)=%s",
        report["rows"],
        report["unique_subjects"],
        report["row_group_counts"],
    )
    return report


def _collect_session(session_dir: Path) -> dict:
    """Inspect one ``OAS2_XXXX_MRY`` folder; ignore legacy/non-scan files."""
    raw_dir = session_dir / "RAW"
    if not raw_dir.is_dir():
        return {"mri_id": session_dir.name, "exists": False, "series": [], "unpaired": []}

    stems: dict[str, dict[str, bool]] = {}
    for f in sorted(raw_dir.iterdir()):
        if not f.is_file() or _IGNORED_FILE_RE.search(f.name):
            continue
        m = re.match(r"^(mpr-\d+)\.(nifti\.(img|hdr)|nii)$", f.name, re.IGNORECASE)
        if not m:
            continue
        entry = stems.setdefault(m.group(1).lower(), {"img": False, "hdr": False})
        if f.suffix.lower() == ".nii":
            # a single-file NIfTI carries its own header — complete on its own
            entry["img"] = True
            entry["hdr"] = True
        elif f.suffix.lower() == ".img":
            entry["img"] = True
        else:
            entry["hdr"] = True

    complete = sorted(s for s, e in stems.items() if e["img"] and e["hdr"])
    unpaired = sorted(s for s, e in stems.items() if e["img"] != e["hdr"])
    return {
        "mri_id": session_dir.name,
        "exists": True,
        "series": complete,
        "unpaired": unpaired,
    }


def audit_raw(
    df: pd.DataFrame,
    raw_paths: Sequence[str | Path],
    out_path: str | Path | None = None,
    min_matched: int = MIN_MATCHED_SESSIONS,
) -> dict:
    """Cross-check CSV sessions against raw scan folders on disk.

    Args:
        df: validated demographics dataframe (must contain an ``MRI ID`` column).
        raw_paths: the raw dataset root folders.
        out_path: when given, the report is written here as JSON.
        min_matched: hard floor — fewer matched sessions raises ``CsvAuditError``.

    The report counts only *paired* ``mpr-N.img/.hdr`` series; ``OLD/`` folders,
    ``.log`` files and stray archives are ignored (they exist in the real data).
    """
    csv_ids = {str(x).strip() for x in df["MRI ID"]}
    disk: dict[str, dict] = {}
    for base in raw_paths:
        base = Path(base)
        if not base.is_dir():
            logger.warning("raw path does not exist: %s", base)
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and SESSION_ID_RE.match(child.name):
                disk[child.name] = _collect_session(child)

    matched = {k: v for k, v in disk.items() if v["exists"] and k in csv_ids}
    missing_scan = sorted(csv_ids - set(disk))  # in CSV, no folder on disk
    empty_on_disk = sorted(k for k in matched if not matched[k]["series"])
    extra_on_disk = sorted(set(disk) - csv_ids)  # on disk, not in CSV
    unpaired = {k: matched[k]["unpaired"] for k in matched if matched[k]["unpaired"]}

    series_hist: dict[int, int] = {}
    for v in matched.values():
        n = len(v["series"])
        series_hist[n] = series_hist.get(n, 0) + 1

    usable = len(matched) - len(empty_on_disk)
    report = {
        "csv_sessions": len(csv_ids),
        "sessions_on_disk": len(disk),
        "matched": len(matched),
        "usable_with_series": usable,
        "missing_scan": missing_scan,
        "empty_on_disk": empty_on_disk,
        "extra_on_disk": extra_on_disk,
        "unpaired_files": unpaired,
        "series_per_session_hist": {str(k): v for k, v in sorted(series_hist.items())},
        "raw_paths": [str(p) for p in raw_paths],
    }

    if len(matched) < min_matched:
        raise CsvAuditError(
            f"only {len(matched)} CSV sessions matched on disk (floor {min_matched}); "
            f"refusing to continue. Report: missing_scan={len(missing_scan)}, "
            f"empty_on_disk={len(empty_on_disk)}"
        )
    if empty_on_disk:
        logger.warning(
            "%d matched sessions have no complete mpr series: %s",
            len(empty_on_disk),
            empty_on_disk[:10],
        )

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Raw audit written: %s (matched %s/%s)", out, len(matched), len(csv_ids))
    else:
        logger.info(
            "Raw audit: matched %s/%s sessions (usable %s, extra %s, unpaired %s)",
            len(matched),
            len(csv_ids),
            usable,
            len(extra_on_disk),
            len(unpaired),
        )
    return report


def main() -> None:  # pragma: no cover - thin CLI wrapper
    """CLI: ``python -m neurosight.data.audit [config_dir]`` — validate + audit."""
    import sys

    from neurosight.config import load_config

    config_dir = sys.argv[1] if len(sys.argv) > 1 else "configs"
    cfg = load_config(config_dir)
    csv_report = validate_csv(cfg.data.csv_path)
    df = pd.read_csv(cfg.data.csv_path, encoding="latin1")
    raw_report = audit_raw(df, cfg.data.raw_paths)
    full = {"csv": csv_report, "raw": raw_report}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_audit.json").write_text(json.dumps(full, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
