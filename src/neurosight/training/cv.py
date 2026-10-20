"""cv.py — subject-level 5-fold cross-validation for one model variant (plan T4.2)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from neurosight.config import NeuroSightConfig
from neurosight.data.cache import SessionCache
from neurosight.data.dataset import make_dataloaders
from neurosight.data.loader import OASIS2Loader
from neurosight.models.registry import build_model, family
from neurosight.training.losses import get_class_weights
from neurosight.training.trainer import Trainer
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed

logger = get_logger(__name__)


def train_single_fold(
    cfg: NeuroSightConfig,
    model_key: str,
    train_records,
    val_records,
    run_name: str,
    cache: SessionCache | None = None,
    seed_override: int | None = None,
) -> dict:
    """Train one model instance on the given record lists; return best metrics.

    ``seed_override`` re-seeds the run independently of ``cfg.training.seed``
    (the --seed enabler for seed-ensemble training, plan 07 T5).
    """
    seed = cfg.training.seed if seed_override is None else seed_override
    set_seed(seed, deterministic=cfg.training.deterministic)
    model = build_model(model_key, cfg.model)
    train_loader, val_loader, _ = make_dataloaders(
        train_records,
        val_records,
        None,
        family(model_key),
        cfg.data,
        cfg.training,
        input_mode=cfg.model.input_mode,
        cache=cache,
    )
    labels = [r.label for r in train_records]
    weights = get_class_weights(labels) if not cfg.training.use_sampler else None
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg.training,
        run_name=run_name,
        class_weights=weights,
        input_mode=cfg.model.input_mode,
        model_key=model_key,
    )
    trainer.train()
    return {
        "best_val_auc": trainer.best_val_auc,
        "best_epoch": trainer.best_epoch,
        "best_val_acc": max(trainer.history["val_acc"], default=0.0),
    }


def run_cv(
    cfg: NeuroSightConfig,
    loader: OASIS2Loader,
    model_key: str,
    cache: SessionCache | None = None,
    n_folds: int | None = None,
    max_epochs_override: int | None = None,
    out_dir: str | Path = "reports/csv",
    seed_override: int | None = None,
    run_suffix: str = "",
) -> dict:
    """5-fold subject-level CV for one variant. Saves reports/csv/cv_<key>.json.

    ``run_suffix`` (e.g. "_s43" for an explicit --seed) is appended to every
    run name and the report filename so seed-varied runs never overwrite each
    other's checkpoints or reports.
    """
    n_folds = n_folds or cfg.training.n_folds
    if max_epochs_override is not None:
        training = cfg.training.model_copy(update={"epochs": max_epochs_override})
        cfg = cfg.model_copy(update={"training": training})

    folds = loader.cv_folds(n_folds)
    fold_results = []
    for fi, (train_records, val_records) in enumerate(folds, start=1):
        t0 = time.time()
        logger.info(
            "CV fold %d/%d for %s: train=%d val=%d",
            fi,
            n_folds,
            model_key,
            len(train_records),
            len(val_records),
        )
        result = train_single_fold(
            cfg,
            model_key,
            train_records,
            val_records,
            run_name=f"cv{fi}_{model_key}_{cfg.model.input_mode}{run_suffix}",
            cache=cache,
            seed_override=seed_override,
        )
        result["fold"] = fi
        result["minutes"] = round((time.time() - t0) / 60, 1)
        fold_results.append(result)
        logger.info(
            "fold %d done: AUC %.4f (%.1f min)", fi, result["best_val_auc"], result["minutes"]
        )

    aucs = [r["best_val_auc"] for r in fold_results]
    summary = {
        "model_key": model_key,
        "input_mode": cfg.model.input_mode,
        "n_folds": n_folds,
        "fold_results": fold_results,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }
    logger.info(
        "CV %s: AUC %.4f +/- %.4f | folds %s",
        model_key,
        summary["mean_auc"],
        summary["std_auc"],
        [round(a, 3) for a in aucs],
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"cv_{model_key}_{cfg.model.input_mode}{run_suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def train_final_model(
    cfg: NeuroSightConfig,
    loader: OASIS2Loader,
    model_key: str,
    cache: SessionCache | None = None,
    seed_override: int | None = None,
    run_suffix: str = "",
) -> dict:
    """Train the final model on the train+val pool with an OUT-OF-SAMPLE
    selection split; save best.

    The pool is split 85/15 (stratified by sequence label, seed 42) into fit and
    selection records: the model trains on the fit split while early stopping
    and checkpointing run on the selection split, so the reported metrics are
    selection quality on unseen records — validating on the model's own
    training records would inflate them. The held-out test set stays untouched.
    Returned metrics are labelled ``selection_*`` to make that explicit.

    With an explicit ``--seed`` the run name gets the ``run_suffix`` (plan 07
    T5): "final_{key}_{mode}_s{seed}" so the 5 seed checkpoints coexist.
    """
    splits = loader.split_sequences()
    pool = list(splits["train"]) + list(splits["val"])
    labels = [r.label for r in pool]
    n_test = math.ceil(0.15 * len(pool))
    n_classes = int(np.unique(np.asarray(labels)).size)
    stratify: list[int] | None = labels
    if (np.bincount(np.asarray(labels)) < 2).any() or n_test < n_classes:
        # tiny pools (debug runs) cannot support a stratified split honestly
        logger.warning(
            "selection split unstratified (pool of %d records -> %d selection rows)",
            len(pool),
            n_test,
        )
        stratify = None
    fit_records, selection_records = train_test_split(
        pool, test_size=0.15, random_state=42, stratify=stratify
    )
    raw = train_single_fold(
        cfg,
        model_key,
        fit_records,
        selection_records,
        run_name=f"final_{model_key}_{cfg.model.input_mode}{run_suffix}",
        cache=cache,
        seed_override=seed_override,
    )
    return {
        "selection_val_auc": raw["best_val_auc"],
        "selection_epoch": raw["best_epoch"],
        "selection_val_acc": raw["best_val_acc"],
    }


__all__ = ["run_cv", "train_final_model", "train_single_fold"]
