"""excellence.py — Model-Excellence harness runner (docs/plan/07, task T4).

Thin CLI over the excellence modules: runs the same-fold CV systems and prints
the comparison against the S0 champion baseline. All selection happens on CV —
the held-out test set is NOT touched here:

    python scripts/excellence.py                                 # S1 + S2 + S3
    python scripts/excellence.py --systems S1 S2 --model 7_dual_attn
    python scripts/excellence.py --systems S3 --no-tta

Leakage contract: fold i is scored ONLY by models trained blind to fold i's
val records — its per-variant fold-i checkpoints. In k-fold CV, fold i's val
subjects sit in every other fold's training set, so any cross-fold member
would score training data; cross-fold ensembling is therefore banned here.

Systems (plan 07 section 2, same-fold):
    S0   champion v1 baseline constants (--baseline-auc / --baseline-std)
    S1a  single-model CV baseline, champion variant (S0-style, per fold)
    S1b  single-model CV baseline, 8_transformer variant
    S2   same-fold cross-variant ensemble (mean softmax of the S1 members)
    S3   S2 + TTA (identity, h-flip, slice-shift ±2)

Seeded reruns: checkpoints written by a seeded rerun carry a run-name suffix
(e.g. ``_s43``); pass ``--run-suffix s43`` to score that rerun's checkpoints
exactly — unsuffixed and other-seed files are never mixed into one ensemble.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click
import torch

from neurosight.config import NeuroSightConfig, load_config
from neurosight.data.cache import SessionCache
from neurosight.excellence.ensemble import (
    FoldPredictions,
    samefold_cross_variant,
    samefold_fold_predictions,
)
from neurosight.excellence.report import evaluate_fold_predictions, format_comparison_table
from neurosight.excellence.tta import tta_collect_predictions
from neurosight.models.registry import family
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed

logger = get_logger(__name__)

ALL_SYSTEMS = ("S1", "S2", "S3")
# plan 07 section 2: S2 also averages the transformer fold models (2D-only ruling)
SECOND_VARIANT = "8_transformer"


def systems_to_compute(requested: Iterable[str]) -> set[str]:
    """Map requested systems to the system runs the harness must compute.

    Systems are independent — no request implies another. In particular S3
    computes its OWN cross-variant predictions with TTA (it never reuses S2's
    non-TTA ensemble run), so ``--systems S3`` must compute S3 only; requesting
    S2 as well is the user's explicit choice.
    """
    wanted = set(requested) & set(ALL_SYSTEMS)
    return {system for system in ALL_SYSTEMS if system in wanted}


@click.command()
@click.option(
    "--systems",
    multiple=True,
    type=click.Choice(ALL_SYSTEMS),
    default=ALL_SYSTEMS,
    show_default=True,
    help="Systems to evaluate (S1 reports per-variant rows S1a/S1b).",
)
@click.option(
    "--model",
    "model_key",
    default="7_dual_attn",
    show_default=True,
    help="Champion variant: S1a and S2's first member.",
)
@click.option("--config-dir", default="configs", show_default=True)
@click.option("--n-folds", default=5, show_default=True)
@click.option(
    "--tta/--no-tta",
    "use_tta",
    default=True,
    show_default=True,
    help="TTA for S3 (identity, h-flip, slice-shift ±2); applies to S3+.",
)
@click.option(
    "--run-suffix",
    "run_suffix",
    default=None,
    help="Seed run suffix (e.g. 's43') — collect checkpoints from that seeded rerun only.",
)
@click.option("--models-dir", default="models", show_default=True)
@click.option("--out", "out_dir", default="reports/csv/excellence", show_default=True)
@click.option("--baseline-auc", default=0.7444, show_default=True, help="S0 mean CV AUC.")
@click.option("--baseline-std", default=0.0965, show_default=True, help="S0 CV AUC std.")
@click.option(
    "--bake-off",
    is_flag=True,
    help="Held-out test bake-off — REFUSED without the acknowledgement flag.",
)
@click.option(
    "--i-know-this-touches-the-test-set",
    "test_set_ack",
    is_flag=True,
    help="Acknowledgement required by --bake-off (the test set is touched exactly once).",
)
def main(
    systems: tuple[str, ...],
    model_key: str,
    config_dir: str,
    n_folds: int,
    use_tta: bool,
    run_suffix: str | None,
    models_dir: str,
    out_dir: str,
    baseline_auc: float,
    baseline_std: float,
    bake_off: bool,
    test_set_ack: bool,
) -> None:
    if bake_off and not test_set_ack:
        raise click.UsageError(
            "--bake-off evaluates the held-out test set, which may be touched exactly once. "
            "Re-run with --i-know-this-touches-the-test-set to acknowledge."
        )
    if bake_off:
        # the guard is implemented; the bake-off itself lands in plan task T7
        raise click.ClickException(
            "the bake-off is a later task (plan 07 T7); run the CV systems for now"
        )

    for key in dict.fromkeys((model_key, SECOND_VARIANT)):
        if family(key) != "2d":
            raise click.ClickException(
                f"variant '{key}' is not a 2D model — the same-fold ensemble is 2D-only"
            )

    cfg: NeuroSightConfig = load_config(config_dir)
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = SessionCache(cfg.data.processed_dir, mode="2d")
    collect_tta = tta_collect_predictions if use_tta else None
    variants = [model_key] + ([SECOND_VARIANT] if SECOND_VARIANT != model_key else [])
    requested = [s for s in ALL_SYSTEMS if s in systems]
    to_compute = systems_to_compute(requested)
    logger.info(
        "excellence: systems=%s computed=%s variants=%s folds=%d tta=%s run_suffix=%s device=%s",
        requested,
        sorted(to_compute),
        variants,
        n_folds,
        use_tta,
        run_suffix,
        device,
    )

    predictions: dict[str, FoldPredictions] = {}
    summaries: dict[str, dict[str, Any]] = {}

    def evaluate(system: str, fold_preds: FoldPredictions) -> None:
        predictions[system] = fold_preds
        summaries[system] = evaluate_fold_predictions(fold_preds)

    if "S1" in to_compute:
        # per-variant single-model CV baselines: each member visible on its own
        evaluate(
            "S1a",
            samefold_fold_predictions(
                variants[0],
                cfg,
                cache=cache,
                n_folds=n_folds,
                models_dir=models_dir,
                run_suffix=run_suffix,
                device=device,
            ),
        )
        if len(variants) > 1:
            evaluate(
                "S1b",
                samefold_fold_predictions(
                    variants[1],
                    cfg,
                    cache=cache,
                    n_folds=n_folds,
                    models_dir=models_dir,
                    run_suffix=run_suffix,
                    device=device,
                ),
            )
    if "S2" in to_compute:
        # S2 runs only when S2 itself is requested — S3 computes its own
        # cross-variant predictions with TTA below and never reuses this run
        evaluate(
            "S2",
            samefold_cross_variant(
                variants,
                cfg,
                cache=cache,
                n_folds=n_folds,
                models_dir=models_dir,
                run_suffix=run_suffix,
                device=device,
            ),
        )
    if "S3" in to_compute:
        if not use_tta:
            logger.warning("--no-tta: S3 will duplicate S2 (TTA is S3's only difference)")
        evaluate(
            "S3",
            samefold_cross_variant(
                variants,
                cfg,
                cache=cache,
                n_folds=n_folds,
                models_dir=models_dir,
                run_suffix=run_suffix,
                collect_fn=collect_tta,
                device=device,
            ),
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for system, summary in sorted(summaries.items()):
        payload = {
            "system": system,
            "model_key": model_key,
            "input_mode": cfg.model.input_mode,
            "n_folds": n_folds,
            # only S3 uses TTA (identity, h-flip, slice-shift ±2)
            "tta": system == "S3" and bool(use_tta),
            **summary,
        }
        (out_path / f"excellence_{system}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    table = format_comparison_table(summaries, baseline_auc=baseline_auc, baseline_std=baseline_std)
    (out_path / "comparison.txt").write_text(table, encoding="utf-8")
    click.echo(table)


if __name__ == "__main__":
    main()
