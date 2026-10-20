"""search.py — bounded hyperparameter search for the champion family (plan T4.3).

Grid: lr {1e-4, 3e-4} x dropout {0.3, 0.5} x lstm_hidden {32, 64} = 8 configs
x 3 folds. Slice count is fixed at the cached 32 — varying it would multiply
preprocessing runs for a knob the predecessor's wider search showed to be
minor. Deviation from the plan's 16-config grid is recorded in the reports file.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import cast

import numpy as np

from neurosight.config import NeuroSightConfig
from neurosight.data.cache import SessionCache
from neurosight.data.loader import OASIS2Loader
from neurosight.models.registry import build_model, family
from neurosight.training.losses import get_class_weights
from neurosight.training.trainer import Trainer
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed

logger = get_logger(__name__)

SEARCH_GRID = {
    "learning_rate": [1e-4, 3e-4],
    "dropout": [0.3, 0.5],
    "lstm_hidden": [32, 64],
}
N_FOLDS_SEARCH = 3


def _train_config(cfg: NeuroSightConfig, **updates):
    training = cfg.training.model_copy(update=updates)
    return cfg.model_copy(update={"training": training})


def run_search(
    cfg: NeuroSightConfig,
    loader: OASIS2Loader,
    model_key: str = "3_cnn_lstm",
    grid_override: dict[str, list] | None = None,
    epochs: int = 20,
) -> list[dict]:
    """Bounded grid search with 3-fold subject-level CV; writes a results JSON.

    ``grid_override``/``epochs`` exist for tests and --quick runs; production
    uses the full bounded grid at 20 epochs per fold.
    """
    from neurosight.data.dataset import make_dataloaders

    grid: dict[str, list] = cast(
        dict[str, list], dict(grid_override) if grid_override else dict(SEARCH_GRID)
    )
    n_folds = N_FOLDS_SEARCH
    cache = SessionCache(cfg.data.processed_dir, mode="2d")
    folds = loader.cv_folds(n_folds)
    value_lists = [list(v) for v in grid.values()]
    combos = [tuple(v) for v in itertools.product(*value_lists)]
    keys = list(grid)
    results = []

    logger.info(
        "hyperparameter search: %d configs x %d folds (slices fixed at %d)",
        len(combos),
        n_folds,
        cfg.data.num_slices,
    )
    for ci, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo, strict=True))
        fold_aucs = []
        t0 = time.time()
        for fi, (train_records, val_records) in enumerate(folds, start=1):
            set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
            search_cfg = _train_config(
                cfg,
                epochs=epochs,  # bounded: search is a ranking, not a final fit
                learning_rate=params["learning_rate"],
            )
            model = build_model(
                model_key,
                cfg.model.model_copy(
                    update={"dropout": params["dropout"], "lstm_hidden": params["lstm_hidden"]}
                ),
            )
            train_loader, val_loader, _ = make_dataloaders(
                train_records,
                val_records,
                None,
                family(model_key),
                cfg.data,
                search_cfg.training,
                input_mode=cfg.model.input_mode,
                cache=cache,
            )
            # the sampler already balances classes by resampling: layering
            # inverse-frequency weights on top would double-weight minority
            # examples (train_single_fold gates on the same flag)
            weights = (
                None
                if search_cfg.training.use_sampler
                else get_class_weights([r.label for r in train_records])
            )
            trainer = Trainer(
                model,
                train_loader,
                val_loader,
                search_cfg.training,
                run_name=f"search{ci}_f{fi}_{model_key}",
                class_weights=weights,
                input_mode=cfg.model.input_mode,
                model_key=model_key,
            )
            trainer.train()
            fold_aucs.append(trainer.best_val_auc)

        entry = {
            **params,
            "mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "minutes": round((time.time() - t0) / 60, 1),
        }
        results.append(entry)
        logger.info(
            "[%d/%d] %s -> AUC %.4f +/-%.4f (%.1f min)",
            ci,
            len(combos),
            params,
            entry["mean_auc"],
            entry["std_auc"],
            entry["minutes"],
        )

    results.sort(key=lambda r: r["mean_auc"], reverse=True)
    out = Path("reports/csv")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"search_{model_key}_{cfg.model.input_mode}.json").write_text(
        json.dumps(
            {"grid_note": "slices fixed at 32 (cache coherence)", "results": results}, indent=2
        ),
        encoding="utf-8",
    )
    best = results[0]
    logger.info("best config: %s (AUC %.4f)", best, best["mean_auc"])
    return results


__all__ = ["SEARCH_GRID", "run_search"]
