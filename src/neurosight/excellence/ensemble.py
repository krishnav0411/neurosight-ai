"""ensemble.py — same-fold cross-variant ensembling (plan 07, T1).

Leakage contract: within k-fold CV, fold i's val subjects are INSIDE the
training set of every other fold's model (fold j trains on everything except
fold j's val, which contains fold i's val for every j != i). Cross-fold
averaging therefore scores training data and is forbidden. The only clean
ensemble members for fold i are models blind to fold i's val: the per-variant
fold-i checkpoints (different architectures, same training subjects).

Fold-index convention: prediction dict keys are the **0-based** positions of
``OASIS2Loader.cv_folds(n_folds)``; checkpoint files are **1-based**
(``best_{key}_cv{N}_{key}_{mode}.pth`` as written by ``training/cv.py``), so
fold ``i`` is scored by checkpoint ``cv{i + 1}`` of each variant.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neurosight.config import NeuroSightConfig
from neurosight.data.cache import SessionCache
from neurosight.data.dataset import Oasis2DDataset, collate_2d
from neurosight.data.loader import OASIS2Loader
from neurosight.evaluation.evaluator import collect_predictions
from neurosight.models.registry import build_model, family
from neurosight.utils.io import load_checkpoint
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed

logger = get_logger(__name__)

CollectFn = Callable[
    [nn.Module, DataLoader, torch.device], tuple[np.ndarray, np.ndarray, np.ndarray]
]
FoldPredictions = dict[int, dict[str, Any]]


def collect_fold_checkpoints(
    models_dir: str | Path,
    model_key: str,
    n_folds: int = 5,
    *,
    input_mode: str | None = None,
    run_suffix: str | None = None,
) -> dict[int, Path]:
    """Map fold number (1..n_folds) to its CV checkpoint path.

    Checkpoint files are ``best_{key}_cv{N}_{key}_{mode}[_{run_suffix}].pth``
    as written by ``training/cv.py`` — a seeded rerun appends a suffix like
    ``_s43`` to the run name. Matching is EXACT, never a wildcard, so parallel
    runs in one models directory can never be mixed into a single ensemble:

    * ``run_suffix=None`` matches only the unsuffixed
      ``best_{key}_cv{N}_{key}_{mode}.pth``;
    * ``run_suffix="s43"`` matches only ``best_{key}_cv{N}_{key}_{mode}_s43.pth``.

    ``input_mode`` pins the mode component the same way; when it is None the
    mode stays a wildcard (legacy discovery), and a fold matching MORE THAN
    ONE file then raises — pick ``input_mode``/``run_suffix`` to disambiguate.

    Raises:
        FileNotFoundError: listing every fold whose checkpoint is missing.
        ValueError: when an unanchored (``input_mode=None``) fold matches
            several files — e.g. unsuffixed + seeded runs sharing a directory.
    """
    directory = Path(models_dir)
    mode = f"_{re.escape(input_mode)}" if input_mode else "_.+"
    seed = f"_{re.escape(run_suffix)}" if run_suffix else ""
    pattern = re.compile(
        rf"^best_{re.escape(model_key)}_cv(\d+)_{re.escape(model_key)}{mode}{seed}\.pth$"
    )

    found: dict[int, Path] = {}
    ambiguous: dict[int, list[str]] = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            match = pattern.match(path.name)
            if match:
                fold = int(match.group(1))
                if fold in found:
                    ambiguous.setdefault(fold, [found[fold].name]).append(path.name)
                else:
                    found[fold] = path

    if ambiguous:
        details = "; ".join(f"cv{fold}: {names}" for fold, names in sorted(ambiguous.items()))
        raise ValueError(
            f"ambiguous CV checkpoints for '{model_key}' in '{directory}' — several files "
            f"match the same fold ({details}). Pass input_mode (and run_suffix for seeded "
            f"reruns) so each fold matches exactly one checkpoint."
        )

    missing = [n for n in range(1, n_folds + 1) if n not in found]
    if missing:
        raise FileNotFoundError(
            f"missing CV checkpoints for '{model_key}' in '{directory}': folds {missing} "
            f"(pattern best_{model_key}_cv<N>_{model_key}{mode}{seed}.pth); "
            f"found folds {sorted(found)}"
        )
    return {n: found[n] for n in range(1, n_folds + 1)}


def _eval_loader(
    records: Sequence, cfg: NeuroSightConfig, cache: SessionCache | None
) -> DataLoader:
    """Shuffle-free eval DataLoader over the given records (2D family)."""
    dataset = Oasis2DDataset(
        records,
        input_mode=cfg.model.input_mode,
        num_slices=cfg.data.num_slices,
        img_size=cfg.data.slice_size,
        slice_strategy=cfg.data.slice_strategy,
        series_selection=cfg.data.series_selection,
        seed=cfg.training.seed,
        cache=cache,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_2d,
        num_workers=cfg.training.num_workers,
    )


def samefold_fold_predictions(
    variant_key: str,
    cfg: NeuroSightConfig,
    cache: SessionCache | None = None,
    n_folds: int = 5,
    models_dir: str | Path = "models",
    *,
    collect_fn: CollectFn | None = None,
    device: torch.device | None = None,
    run_suffix: str | None = None,
) -> FoldPredictions:
    """Same-fold single-model CV predictions per fold.

    Fold ``i`` (0-based) is scored by the variant's OWN fold-i checkpoint
    ``cv{i + 1}`` — the exact model that was trained blind to fold i's val
    records — so this is the honest S0-style CV baseline for one variant and
    the clean member builder for :func:`samefold_cross_variant`. Batch
    inference runs under ``torch.no_grad`` + ``model.eval()`` (via
    :func:`neurosight.evaluation.evaluator.collect_predictions` or a
    TTA-enabled replacement). ``run_suffix`` selects a seeded rerun's
    checkpoints exactly (see :func:`collect_fold_checkpoints`).
    """
    if family(variant_key) != "2d":
        raise NotImplementedError(
            f"same-fold CV predictions are 2D-only (plan ruling); '{variant_key}' is not"
        )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)

    loader = OASIS2Loader(cfg.data)
    fold_ckpts = collect_fold_checkpoints(
        models_dir,
        variant_key,
        n_folds=n_folds,
        input_mode=cfg.model.input_mode,
        run_suffix=run_suffix,
    )
    folds = loader.cv_folds(n_folds)
    if len(folds) != n_folds:
        raise ValueError(f"loader produced {len(folds)} folds, expected {n_folds}")

    collect = collect_fn or collect_predictions
    results: FoldPredictions = {}
    for fold_idx, (_train_records, val_records) in enumerate(folds):
        ckpt = fold_ckpts[fold_idx + 1]  # checkpoint numbering is 1-based; OWN fold only
        model = build_model(variant_key, cfg.model)
        load_checkpoint(ckpt, model=model)
        model.to(device)
        y_true, _y_pred, y_prob = collect(model, _eval_loader(val_records, cfg, cache), device)
        results[fold_idx] = {
            "y_true": y_true,
            "y_prob": np.asarray(y_prob, dtype=np.float64),
            "checkpoint": str(ckpt),
        }
        logger.info(
            "%s fold %d: checkpoint %s -> n=%d", variant_key, fold_idx, ckpt.name, len(y_true)
        )
    return results


def average_variant_probs(
    per_variant: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Mean of aligned per-fold ``y_prob`` across variants.

    ``y_true`` must be identical across variants for every fold — different
    label orders would silently corrupt the average.
    """
    if not per_variant:
        raise ValueError("no variants to average")
    variants = sorted(per_variant)
    fold_ids = set(next(iter(per_variant.values())))
    for key, folds in per_variant.items():
        if set(folds) != fold_ids:
            raise ValueError(
                f"variant '{key}' covers folds {sorted(folds)}, expected {sorted(fold_ids)}"
            )

    averaged: dict[int, dict[str, Any]] = {}
    for fold in sorted(fold_ids):
        y_true: np.ndarray | None = None
        probs: list[np.ndarray] = []
        for key in variants:
            entry = per_variant[key][fold]
            if y_true is None:
                y_true = np.asarray(entry["y_true"])
            elif not np.array_equal(y_true, np.asarray(entry["y_true"])):
                raise ValueError(
                    f"y_true mismatch across variants at fold {fold} (variant '{key}')"
                )
            probs.append(np.asarray(entry["y_prob"], dtype=np.float64))
        assert y_true is not None
        averaged[fold] = {"y_true": y_true, "y_prob": np.mean(probs, axis=0)}
    return averaged


def samefold_cross_variant(
    variant_keys: Sequence[str],
    cfg: NeuroSightConfig,
    cache: SessionCache | None = None,
    n_folds: int = 5,
    models_dir: str | Path = "models",
    *,
    collect_fn: CollectFn | None = None,
    device: torch.device | None = None,
    run_suffix: str | None = None,
) -> FoldPredictions:
    """Mean softmax ACROSS VARIANTS, per fold, on that fold's own models.

    For fold ``i`` every 2D variant contributes its fold-i checkpoint (trained
    blind to fold i's val) — same training subjects, different architectures —
    and the reported probability is the mean softmax across those members.
    ``run_suffix`` selects a seeded rerun's checkpoints exactly, so two seeds
    in one models directory are never averaged together. Non-2D variants are
    skipped with a logged warning (2D-only plan ruling); a call with no 2D
    variant at all raises.
    """
    two_d: list[str] = []
    for key in variant_keys:
        if family(key) == "2d":
            two_d.append(key)
        else:
            logger.warning(
                "samefold_cross_variant: skipping non-2D variant '%s' (2D-only plan ruling)",
                key,
            )
    if not two_d:
        raise ValueError(f"no 2D variants to ensemble among {list(variant_keys)}")

    per_variant = {
        key: samefold_fold_predictions(
            key,
            cfg,
            cache=cache,
            n_folds=n_folds,
            models_dir=models_dir,
            collect_fn=collect_fn,
            device=device,
            run_suffix=run_suffix,
        )
        for key in two_d
    }
    averaged = average_variant_probs(per_variant)
    for fold, entry in averaged.items():
        # provenance: the exact checkpoints that formed this fold's ensemble
        entry["members"] = [per_variant[key][fold]["checkpoint"] for key in sorted(per_variant)]
    return averaged


__all__ = [
    "average_variant_probs",
    "collect_fold_checkpoints",
    "samefold_cross_variant",
    "samefold_fold_predictions",
]
