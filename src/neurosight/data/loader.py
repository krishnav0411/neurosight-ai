"""loader.py — OASIS-2 patient-level sequence loader with leakage-safe splits.

Ported from the predecessor project's ``data_loader.py`` with the fixes required
by ``docs/plan/00-PLAN-REVIEW.md``:

- Config-driven paths and policy (no hardcoded ``D:/`` paths).
- **Subject-level split BEFORE Converted-sequence splitting** — a Converted
  subject's pre- and post-conversion sequences can never straddle splits (B3 fix).
- Configurable mpr-series policy (``first``/``middle``/``mean``) — the old code
  silently took the first file found per session (A3 fix).
- No silent skips: every excluded visit/subject is logged and counted in
  ``self.exclusions`` (B6 fix).

The upstream mandatory gate is ``neurosight.data.audit`` (validate_csv +
audit_raw); this loader assumes it has passed and does not re-validate, so the
same loader can run on synthetic test fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from neurosight.config import DataConfig
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

# Metadata vector layout — see neurosight/data/modes.py (kept in sync there)
_SES_MAX, _MMSE_MAX, _ETIV_MAX, _ASF_MAX = 5.0, 30.0, 2000.0, 2.0
_EDUC_MAX, _AGE_MAX = 20.0, 100.0

_SUBJECT_GROUP_CODE = {"Nondemented": 0, "Demented": 1, "Converted": 2}


@dataclass
class SequenceRecord:
    """One training unit: an ordered list of MRI visits for a single patient.

    For ``Converted`` subjects a subject yields two records: ``kind="pre"``
    (visits before the first CDR>0 visit, label 0) and ``kind="post"`` (from the
    conversion visit onward, label 1). Everyone else yields one ``kind="full"``.
    """

    subject_id: str
    kind: str  # "full" | "pre" | "post"
    visit_paths: list[list[str]] = field(default_factory=list)  # per visit: .img paths
    visit_meta: list[np.ndarray] = field(default_factory=list)  # per visit: (8,) float32
    visit_labels: list[int] = field(default_factory=list)  # per visit: 0/1
    visit_delays: list[int] = field(default_factory=list)  # MR Delay per visit
    label: int = 0  # sequence-level label

    @property
    def n_visits(self) -> int:
        return len(self.visit_paths)


class OASIS2Loader:
    """Build patient-level sequences from the demographics CSV + raw folders."""

    def __init__(
        self,
        config: DataConfig,
        df: pd.DataFrame | None = None,
        raw_paths: list[Path] | None = None,
    ):
        self.config = config
        self.df = (df if df is not None else pd.read_csv(config.csv_path, encoding="latin1")).copy()
        self.df.columns = [str(c).strip() for c in self.df.columns]
        self.raw_paths = [Path(p) for p in (raw_paths or config.raw_paths)]
        self.exclusions: list[dict[str, str]] = []
        self.sequences = self._build()

    # ── session discovery ────────────────────────────────────────────────────

    def _session_dir(self, mri_id: str) -> Path | None:
        wanted = mri_id.strip()
        for base in self.raw_paths:
            if not base.is_dir():
                logger.warning("raw path does not exist: %s", base)
                continue
            candidate = base / wanted / "RAW"
            if candidate.is_dir():
                return candidate
        return None

    def _series_paths(self, mri_id: str) -> list[str]:
        """Resolve one session to scan paths selected by the series policy.

        Keys are the filename minus its final extension (``mpr-1.nifti`` for the
        Analyze pairs on disk, ``mpr-1`` for .nii), and the *actual* file path is
        returned — never a reconstructed name (the reconstruction corrupted the
        real dataset's ``mpr-N.nifti.img`` names on first use).
        """
        raw_dir = self._session_dir(mri_id)
        if raw_dir is None:
            return []
        series: dict[str, Path] = {}
        for p in raw_dir.iterdir():
            lname = p.name.lower()
            if not p.is_file() or not lname.startswith("mpr"):
                continue
            if lname.endswith(".img"):
                # An Analyze .img is unreadable without its .hdr twin — the real
                # dataset contains exactly one orphan (OAS2_0056_MR2/mpr-1), so
                # only paired series are candidates for selection.
                if (raw_dir / (lname[: -len(".img")] + ".hdr")).exists():
                    series[lname[: -len(".img")]] = p
            elif lname.endswith(".nii"):
                series[p.stem] = p
        ordered = sorted(series)
        if not ordered:
            return []  # empty under the "middle" policy too — never index [0] of nothing
        policy = self.config.series_selection
        if policy == "first":
            selected = ordered[:1]
        elif policy == "middle":
            selected = [ordered[len(ordered) // 2]]
        else:  # "mean" — dataset layer averages the volumes of all series
            selected = ordered
        return [str(series[key]) for key in selected]

    def iter_sessions(self) -> list[tuple[str, str, list[str]]]:
        """All CSV sessions resolved under the series policy: (mri_id, subject_id, paths).

        The cache builder walks this — including sessions of subjects later
        excluded for visit-count reasons — so preprocessing is done exactly
        once per session regardless of which sequences consume it.
        """
        out: list[tuple[str, str, list[str]]] = []
        for _, row in self.df.sort_values(["Subject ID", "Visit"]).iterrows():
            mri_id = str(row["MRI ID"]).strip()
            paths = self._series_paths(mri_id)
            if paths:
                out.append((mri_id, str(row["Subject ID"]).strip(), paths))
        return out

    # ── per-visit fields ─────────────────────────────────────────────────────

    def _visit_meta(self, row: pd.Series) -> np.ndarray:
        ses = row["SES"] if not pd.isna(row["SES"]) else self._ses_mean
        mmse = row["MMSE"] if not pd.isna(row["MMSE"]) else self._mmse_mean
        return np.array(
            [
                1.0 if str(row["M/F"]).strip() == "M" else 0.0,
                float(row["Age"]) / _AGE_MAX,
                float(row["EDUC"]) / _EDUC_MAX,
                float(ses) / _SES_MAX,
                float(mmse) / _MMSE_MAX,
                float(row["eTIV"]) / _ETIV_MAX,
                float(row["nWBV"]),
                float(row["ASF"]) / _ASF_MAX,
            ],
            dtype=np.float32,
        )

    def _visit_label(self, row: pd.Series, group: str) -> int:
        cdr = row["CDR"]
        if pd.isna(cdr):
            # Missing CDR: fall back to the subject Group (Group-verified design)
            return 0 if group == "Nondemented" else 1
        try:
            return 0 if float(cdr) == 0.0 else 1
        except (TypeError, ValueError):
            return 0 if group == "Nondemented" else 1

    def _visit_delay(self, row: pd.Series) -> int:
        delay = row.get("MR Delay", 0)
        return 0 if pd.isna(delay) else int(delay)

    def _exclude(self, subject_id: str, reason: str, detail: str = "") -> None:
        self.exclusions.append({"subject_id": subject_id, "reason": reason, "detail": detail})
        logger.warning("excluded %s: %s %s", subject_id, reason, detail)

    # ── sequence construction ────────────────────────────────────────────────

    def _build(self) -> list[SequenceRecord]:
        if self.config.label_mode != "binary_cdr":
            raise NotImplementedError(
                f"label_mode '{self.config.label_mode}' is not implemented; use 'binary_cdr'"
            )
        self._ses_mean = float(self.df["SES"].mean())
        self._mmse_mean = float(self.df["MMSE"].mean())

        sequences: list[SequenceRecord] = []
        for subject_id, grp in self.df.groupby("Subject ID"):
            grp = grp.sort_values("Visit").reset_index(drop=True)
            group = str(grp["Group"].iloc[0])

            visits: list[dict] = []
            n_expected = 0
            for _, row in grp.iterrows():
                n_expected += 1
                paths = self._series_paths(str(row["MRI ID"]))
                if not paths:
                    continue  # unresolved visits are counted and logged, never faked
                visits.append(
                    {
                        "paths": paths,
                        "meta": self._visit_meta(row),
                        "label": self._visit_label(row, group),
                        "delay": self._visit_delay(row),
                    }
                )
            missing = n_expected - len(visits)
            if missing:
                logger.info("%s: %d/%d visits lack raw scans", subject_id, missing, n_expected)
            if len(visits) < self.config.min_visits:
                self._exclude(
                    subject_id,
                    "fewer_than_min_visits",
                    f"{len(visits)} resolvable visits, need {self.config.min_visits}",
                )
                continue

            if group == "Converted":
                sequences.extend(self._split_converted(subject_id, visits))
                continue

            label = 0 if group == "Nondemented" else 1
            cdr_labels = [v["label"] for v in visits]
            if cdr_labels[-1] != label:
                logger.warning(
                    "%s: Group=%s but last-visit CDR implies label %s — Group wins",
                    subject_id,
                    group,
                    cdr_labels[-1],
                )
            sequences.append(
                SequenceRecord(
                    subject_id=subject_id,
                    kind="full",
                    visit_paths=[v["paths"] for v in visits],
                    visit_meta=[v["meta"] for v in visits],
                    visit_labels=cdr_labels,
                    visit_delays=[v["delay"] for v in visits],
                    label=label,
                )
            )

        logger.info(
            "OASIS2Loader: %d sequences from %d subjects (%d exclusions)",
            len(sequences),
            self.df["Subject ID"].nunique(),
            len(self.exclusions),
        )
        return sequences

    def _split_converted(self, subject_id: str, visits: list[dict]) -> list[SequenceRecord]:
        """Split a Converted subject at the first CDR>0 visit into pre/post records."""
        split = next((i for i, v in enumerate(visits) if v["label"] == 1), None)
        out: list[SequenceRecord] = []

        def _emit(kind: str, chunk: list[dict], label: int) -> None:
            if len(chunk) >= self.config.min_visits:
                out.append(
                    SequenceRecord(
                        subject_id=subject_id,
                        kind=kind,
                        visit_paths=[v["paths"] for v in chunk],
                        visit_meta=[v["meta"] for v in chunk],
                        visit_labels=[v["label"] for v in chunk],
                        visit_delays=[v["delay"] for v in chunk],
                        label=label,
                    )
                )
            elif chunk:
                self._exclude(
                    subject_id,
                    f"converted_{kind}_too_short",
                    f"{len(chunk)} visit(s), need {self.config.min_visits}",
                )

        if split is None:
            # Converted subject whose recorded visits never crossed CDR>0 —
            # usable only as a (pre-conversion) nondemented sequence.
            logger.warning(
                "%s: Converted with no post-conversion visit; using as pre-sequence", subject_id
            )
            _emit("pre", visits, 0)
        else:
            _emit("pre", visits[:split], 0)
            _emit("post", visits[split:], 1)
        return out

    # ── subject-level splits (leakage safety) ────────────────────────────────

    def _subject_frame(self) -> tuple[list[str], np.ndarray]:
        subj = sorted(self.df["Subject ID"].astype(str).unique())
        group_of = (
            self.df.assign(Subject=lambda d: d["Subject ID"].astype(str))
            .drop_duplicates("Subject")
            .set_index("Subject")["Group"]
        )
        y = np.array([_SUBJECT_GROUP_CODE[group_of[s]] for s in subj])
        return subj, y

    def subject_splits(self) -> dict[str, list[str]]:
        """70/15/15 subject-level split stratified by subject Group.

        Splitting happens on subjects *before* sequence construction is used, so
        a Converted subject's pre/post sequences always land in the same split.
        Degrades to an unstratified split (with a warning) when any class has
        fewer than 2 subjects — the case in --debug runs, never in production.
        """
        subj, y = self._subject_frame()
        seed = self.config.random_seed
        stratify: np.ndarray | None = y
        if (np.bincount(y) < 2).any():
            logger.warning(
                "stratified split unavailable (class with <2 subjects); "
                "falling back to unstratified split"
            )
            stratify = None
        val_adj = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
        tv_idx, test_idx = train_test_split(
            np.arange(len(subj)),
            test_size=self.config.test_ratio,
            random_state=seed,
            stratify=stratify,
        )
        train_idx, val_idx = train_test_split(
            tv_idx,
            test_size=val_adj,
            random_state=seed,
            stratify=stratify[tv_idx] if stratify is not None else None,
        )
        return {
            "train": [subj[i] for i in train_idx],
            "val": [subj[i] for i in val_idx],
            "test": [subj[i] for i in test_idx],
        }

    def split_sequences(self) -> dict[str, list[SequenceRecord]]:
        """Group this loader's sequences by their subject's split."""
        splits = self.subject_splits()
        by_subject: dict[str, list[SequenceRecord]] = {}
        for seq in self.sequences:
            by_subject.setdefault(seq.subject_id, []).append(seq)
        return {
            name: [s for subj in ids for s in by_subject.get(subj, [])]
            for name, ids in splits.items()
        }

    def cv_folds(self, n_folds: int) -> list[tuple[list[SequenceRecord], list[SequenceRecord]]]:
        """Subject-level stratified K-fold CV (train/val sequence lists per fold).

        ``n_folds`` comes from the training config and is explicit here so the
        data layer stays independent of the training layer.
        """
        subj, y = self._subject_frame()
        by_subject: dict[str, list[SequenceRecord]] = {}
        for seq in self.sequences:
            by_subject.setdefault(seq.subject_id, []).append(seq)
        folds: list[tuple[list[SequenceRecord], list[SequenceRecord]]] = []
        if (np.bincount(y) < n_folds).any():
            logger.warning(
                "stratified CV unavailable (a class has fewer members than folds); "
                "falling back to unstratified K-fold"
            )
            from sklearn.model_selection import KFold

            splitter = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.config.random_seed,
            )
            fold_iter = splitter.split(np.arange(len(subj)))
        else:
            skf = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.config.random_seed,
            )
            fold_iter = skf.split(np.arange(len(subj)), y)
        for tr_idx, va_idx in fold_iter:
            train = [s for i in tr_idx for s in by_subject.get(subj[i], [])]
            val = [s for i in va_idx for s in by_subject.get(subj[i], [])]
            folds.append((train, val))
        return folds

    # ── introspection ────────────────────────────────────────────────────────

    def labels(self) -> np.ndarray:
        return np.array([s.label for s in self.sequences])

    def summary(self) -> dict:
        subjects = {s.subject_id for s in self.sequences}
        return {
            "sequences": len(self.sequences),
            "subjects": len(subjects),
            "by_kind": {
                k: sum(1 for s in self.sequences if s.kind == k) for k in ("full", "pre", "post")
            },
            "by_label": {
                int(v): int(c)
                for v, c in zip(*np.unique(self.labels(), return_counts=True), strict=False)
            },
            "avg_visits": (
                float(np.mean([s.n_visits for s in self.sequences])) if self.sequences else 0.0
            ),
            "exclusions": len(self.exclusions),
            "exclusion_reasons": sorted({e["reason"] for e in self.exclusions}),
        }


__all__ = ["OASIS2Loader", "SequenceRecord"]
