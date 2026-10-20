"""train.py — CLI entry point: CV, ablation, and final-model training.

python scripts/train.py --debug                       # 5 subjects, 2 epochs, 1 fold
python scripts/train.py --model 3_cnn_lstm            # CV + final model
python scripts/train.py --ablation                    # all 11 variants (overnight)
python scripts/train.py --model 5_temporal_attn --input-mode mri_demo
python scripts/train.py --model 7_dual_attn --mode final --seed 43   # E5 seed run
"""

from __future__ import annotations

import json

import click

from neurosight.config import load_config
from neurosight.data.cache import SessionCache
from neurosight.data.loader import OASIS2Loader
from neurosight.models.registry import ABLATION_VARIANTS, family
from neurosight.training.cv import run_cv, train_final_model
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed

logger = get_logger(__name__)


@click.command()
@click.option("--config-dir", default="configs", show_default=True)
@click.option(
    "--model",
    "model_keys",
    multiple=True,
    default=("3_cnn_lstm",),
    help="Registry key(s) to train.",
)
@click.option("--ablation", is_flag=True, help="Train all 11 registered variants.")
@click.option(
    "--input-mode",
    default=None,
    help="Override the config input mode (mri_only|demo_only|mri_demo|mri_clinical).",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Override the training seed; run names get an _s<seed> suffix so the "
    "seed-ensemble checkpoints never collide (plan 07 T5).",
)
@click.option(
    "--mode",
    type=click.Choice(["cv", "final", "both"]),
    default="both",
    show_default=True,
    help="cv: 5-fold CV. final: train on train+val.",
)
@click.option("--debug", is_flag=True, help="5 subjects, 2 epochs, 1 fold — pipeline smoke test.")
def main(
    config_dir: str,
    model_keys: tuple[str],
    ablation: bool,
    input_mode: str | None,
    seed: int | None,
    mode: str,
    debug: bool,
) -> None:
    cfg = load_config(config_dir)
    if input_mode:
        cfg = cfg.model_copy(
            update={"model": cfg.model.model_copy(update={"input_mode": input_mode})}
        )
    run_suffix = ""
    if seed is not None:
        training = cfg.training.model_copy(update={"seed": seed})
        cfg = cfg.model_copy(update={"training": training})
        run_suffix = f"_s{seed}"
    if debug:
        training = cfg.training.model_copy(update={"epochs": 2, "n_folds": 2})
        cfg = cfg.model_copy(update={"training": training})

    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
    loader = OASIS2Loader(cfg.data)
    if debug:
        keep = {s.subject_id for s in loader.sequences[:5]}
        loader.sequences = [s for s in loader.sequences if s.subject_id in keep]
        logger.info("DEBUG mode: %d subjects / %d sequences", len(keep), len(loader.sequences))

    keys = (
        sorted(ABLATION_VARIANTS, key=lambda k: int(k.split("_")[0]))
        if ablation
        else list(model_keys)
    )
    for key in keys:
        fam = family(key)
        cache = SessionCache(cfg.data.processed_dir, mode="3d" if fam != "2d" else "2d")
        logger.info("=== %s (%s, %s) ===", key, fam, cfg.model.input_mode)

        if mode in ("cv", "both"):
            # debug lowers cfg.training.n_folds to 2 (the CV floor)
            summary = run_cv(
                cfg, loader, key, cache=cache, seed_override=seed, run_suffix=run_suffix
            )
            if debug:
                logger.info("DEBUG CV result: %s", json.dumps(summary["fold_results"][0]))

        if mode in ("final", "both"):
            final = train_final_model(
                cfg, loader, key, cache=cache, seed_override=seed, run_suffix=run_suffix
            )
            logger.info("final %s: selection_val_auc=%.4f", key, final["selection_val_auc"])

    click.echo("training complete")


if __name__ == "__main__":
    main()
