"""validation.py — upload validation pipeline (plan doc 05 §3.1).

Fail fast with per-file reasons: extension allow-list, size caps, .img/.hdr
pairing, and a nibabel header probe. No user-supplied filename is ever used in
a filesystem path — uploads are re-stemmed under the job id.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib

from neurosight.config import ServingConfig
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

VISIT_RE = re.compile(
    r"(?:OAS2_\d{4}_MR\d+|SYNTH_\d{4}_MR\d+|sub-[a-zA-Z0-9]+_ses-\d+)", re.IGNORECASE
)
ALLOWED_SUFFIXES = {".img", ".hdr", ".nii"}


class UploadValidationError(ValueError):
    """Raised with a human-readable list of per-file problems."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class ValidatedUpload:
    """Persisted, grouped upload ready for inference."""

    visit_labels: list[str] = field(default_factory=list)
    visits: list[list[str]] = field(default_factory=list)  # scan paths per visit
    total_bytes: int = 0


def _group_key(filename: str, explicit_label: str | None) -> str:
    """Group a file into a visit: the client-supplied label (parallel to the
    file order) wins, else the OAS2_XXXX_MRY pattern in the name, else a shared
    'single' bucket. Browsers and servers strip directory paths, so the client
    MUST send explicit labels when filenames alone are ambiguous."""
    if explicit_label:
        return explicit_label.strip().upper()
    m = VISIT_RE.search(filename)
    if m:
        return m.group(0).upper()
    return "SINGLE_VISIT"


def validate_and_persist(
    files: Sequence[tuple[str, bytes | Path]],
    cfg: ServingConfig,
    job_id: str,
    visit_labels: list[str] | None = None,
) -> ValidatedUpload:
    """Validate uploaded (filename, bytes_or_path) pairs and persist them under
    ``uploads_dir/<job_id>/``.

    Accepts either in-memory bytes or staged temporary file paths for streaming
    uploads to avoid holding entire multi-file payloads in RAM.

    Pairing happens on the ORIGINAL filenames; each complete series is then
    persisted under one shared random base name (so ``.img``/``.hdr`` twins keep
    matching on disk). Raises :class:`UploadValidationError` with every problem
    found, naming the original files.
    """
    import shutil

    explicit: list[str] = list(visit_labels or [])
    if explicit and len(explicit) != len(files):
        raise UploadValidationError(
            [f"visit_labels has {len(explicit)} entries for {len(files)} files"]
        )
    problems: list[str] = []
    total_bytes = 0
    for _, item in files:
        if isinstance(item, Path):
            total_bytes += item.stat().st_size
        else:
            total_bytes += len(item)
    if total_bytes > cfg.max_upload_mb * 1024 * 1024:
        problems.append(f"study exceeds {cfg.max_upload_mb} MB")
    if not files:
        problems.append("no files uploaded")

    # visit group -> series key -> {img|hdr|nii: (original_name, item)}
    grouped: dict[str, dict[str, dict[str, tuple[str, bytes | Path]]]] = {}
    for i, (fname, item) in enumerate(files):
        safe = Path(fname).name
        suffix = Path(safe).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            problems.append(f"{safe}: unsupported extension (allowed .img/.hdr/.nii)")
            continue
        group = _group_key(safe, explicit[i] if i < len(explicit) else None)
        if suffix == ".img":
            key = safe.lower()[: -len(".img")]
            part = "img"
        elif suffix == ".hdr":
            key = safe.lower()[: -len(".hdr")]
            part = "hdr"
        else:
            key = Path(safe).stem.lower()
            part = "nii"
        grouped.setdefault(group, {}).setdefault(key, {})[part] = (safe, item)

    if problems:
        raise UploadValidationError(problems)

    def _save_data(item: bytes | Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(item, Path):
            shutil.move(str(item), str(dest))
        else:
            dest.write_bytes(item)

    # Pair per visit; orphan errors name the ORIGINAL files.
    upload_dir = cfg.uploads_dir / job_id
    visits: list[list[str]] = []
    labels: list[str] = []
    for group in sorted(grouped):
        complete: list[str] = []
        for _key, parts in sorted(grouped[group].items()):
            if "nii" in parts:
                orig, item = parts["nii"]
                dest = upload_dir / f"{uuid.uuid4().hex}.nii"
                _save_data(item, dest)
                complete.append(str(dest))
            elif "img" in parts and "hdr" in parts:
                base = uuid.uuid4().hex
                img_dest: Path | None = None
                for part, suffix in (("img", ".img"), ("hdr", ".hdr")):
                    _, item = parts[part]
                    dest = upload_dir / f"{base}{suffix}"
                    _save_data(item, dest)
                    if part == "img":
                        img_dest = dest  # pipeline loads the .img; the .hdr sits beside it
                assert img_dest is not None
                complete.append(str(img_dest))
            else:
                missing = "img" if "hdr" in parts else "hdr"
                orig = next(iter(parts.values()))[0]
                twin = ".hdr" if missing == "img" else ".img"
                problems.append(f"{orig}: incomplete pair (missing its {twin} twin)")
        if complete:
            visits.append(complete)
            labels.append(group)

    def _fail(failures: list[str]) -> None:
        """Reject a study AFTER files were persisted: remove the upload dir.

        create_analysis returns 422 without ever creating an AnalysisJob row,
        so the retention sweeper (which iterates job rows) could never purge
        these scans — medical images must not linger past rejection.
        """
        import shutil

        shutil.rmtree(upload_dir, ignore_errors=True)
        raise UploadValidationError(failures)

    if problems:
        _fail(problems)
    if not visits:
        _fail(["no complete visits found"])
    if len(visits) > cfg.max_visits:
        _fail([f"{len(visits)} visits exceed the {cfg.max_visits}-visit cap"])

    # Header probe: reject files nibabel cannot open (zip-bombs / garbage)
    for group, scan_paths in zip(labels, visits, strict=True):
        for scan_file in scan_paths:
            try:
                img = nib.load(scan_file)  # type: ignore[no-untyped-call]
                ndim = img.ndim  # type: ignore[attr-defined]
                if ndim not in (3, 4):
                    _fail([f"{Path(scan_file).name}: {ndim}D volume is not MRI-like"])
            except UploadValidationError:
                raise
            except Exception as exc:
                _fail([f"unreadable scan in visit {group}: {exc}"])

    return ValidatedUpload(
        visit_labels=labels,
        visits=visits,
        total_bytes=total_bytes,
    )


__all__ = ["UploadValidationError", "ValidatedUpload", "validate_and_persist"]
