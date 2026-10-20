"""evaluate.py — CLI: held-out test evaluation of a trained final model.

    python scripts/evaluate.py --model 3_cnn_lstm
    python scripts/evaluate.py --model 3_cnn_lstm --calibrate --gradcam

Produces reports/final_evaluation.txt, reports/csv/final_metrics.json,
figures (ROC/calibration/confusion), and updates models/manifest.json with the
fitted temperature so serving matches research numbers exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from neurosight.config import load_config
from neurosight.data.cache import SessionCache
from neurosight.data.dataset import Oasis2DDataset, collate_2d
from neurosight.data.loader import OASIS2Loader
from neurosight.evaluation.calibration import (
    expected_calibration_error,
    fit_temperature,
)
from neurosight.evaluation.evaluator import (
    compute_metrics,
    save_evaluation_report,
    save_metrics_json,
)
from neurosight.models.registry import build_model, family
from neurosight.utils.io import load_checkpoint
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed

logger = get_logger(__name__)


def resolve_final_checkpoint(checkpoint_dir: str | Path, model_key: str, input_mode: str) -> Path:
    """Exact final checkpoint for (model_key, input_mode) — no fallbacks.

    A missing final checkpoint fails loudly: the previous "any best_*" glob
    fallback could select a CV fold's (or another input mode's) checkpoint and
    publish it in models/manifest.json as the requested final model.
    """
    ckpt = Path(checkpoint_dir) / f"best_{model_key}_final_{model_key}_{input_mode}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"no final checkpoint for {model_key}/{input_mode}: {ckpt}")
    return ckpt


@click.command()
@click.option("--config-dir", default="configs", show_default=True)
@click.option("--model", "model_key", default="3_cnn_lstm", show_default=True)
@click.option("--input-mode", default="mri_only", show_default=True)
@click.option(
    "--run-name", default=None, help="Checkpoint run suffix (default: final_<model>_<mode>)"
)
@click.option("--calibrate/--no-calibrate", default=True)
@click.option("--gradcam/--no-gradcam", default=True)
@click.option(
    "--manifest-path",
    default=None,
    help="Target path for evaluation manifest (defaults to reports/evaluation_manifest.json)",
)
@click.option(
    "--force-manifest/--no-force-manifest",
    default=False,
    help="Allow overwriting active serving manifest at models/manifest.json if explicitly targeted",
)
def main(
    config_dir: str,
    model_key: str,
    input_mode: str,
    run_name: str | None,
    calibrate: bool,
    gradcam: bool,
    manifest_path: str | None = None,
    force_manifest: bool = False,
) -> None:
    cfg = load_config(config_dir)
    model_cfg = cfg.model.model_copy(update={"input_mode": input_mode})
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = OASIS2Loader(cfg.data)
    splits = loader.split_sequences()
    test_records = splits["test"]
    logger.info("held-out test set: %d sequences", len(test_records))

    fam = family(model_key)
    if fam != "2d":
        raise SystemExit("held-out evaluation in v1 covers the 2d family (see model manifest)")
    cache = SessionCache(cfg.data.processed_dir, mode="2d")

    def eval_loader(records):
        ds = Oasis2DDataset(
            records,
            input_mode=input_mode,
            num_slices=cfg.data.num_slices,
            img_size=cfg.data.slice_size,
            slice_strategy=cfg.data.slice_strategy,
            series_selection=cfg.data.series_selection,
            seed=cfg.training.seed,
            cache=cache,
        )
        return DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_2d,
            num_workers=cfg.training.num_workers,
        )

    test_loader = eval_loader(test_records)

    model = build_model(model_key, model_cfg)
    ckpt = resolve_final_checkpoint(cfg.training.checkpoint_dir, model_key, input_mode)
    load_checkpoint(ckpt, model=model)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    # ── collect raw (uncalibrated) predictions ───────────────────────────────
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for scans, meta, labels, mask in test_loader:
            logits = model(scans.to(device), meta.to(device), mask.to(device))
            all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    y_true = np.array(all_labels)
    y_prob_raw = np.array(all_probs)

    # ── calibration on validation logits ─────────────────────────────────────
    temperature = 1.0
    if calibrate and len(np.unique(y_true)) > 1:
        val_loader = eval_loader(splits["val"])
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for scans, meta, labels, mask in val_loader:
                logits = model(scans.to(device), meta.to(device), mask.to(device))
                val_logits_list.append(logits.cpu())
                val_labels_list.append(labels)
        val_logits = torch.cat(val_logits_list)
        val_labels = torch.cat(val_labels_list)
        scaler = fit_temperature(val_logits, val_labels)
        temperature = float(scaler.temperature.item())
        # apply temperature to the softmax ratio of the test probabilities
        p1 = np.clip(y_prob_raw, 1e-9, 1 - 1e-9)
        p0 = 1 - p1
        y_prob = p1 ** (1 / temperature) / (p0 ** (1 / temperature) + p1 ** (1 / temperature))
    else:
        y_prob = y_prob_raw

    y_pred = (y_prob >= 0.5).astype(int)
    ece = expected_calibration_error(y_true, y_prob)

    metrics = compute_metrics(
        y_true,
        y_pred,
        y_prob,
        model_name=f"{model_key} (final)",
        n_params=n_params,
        input_mode=input_mode,
        ece=ece,
    )
    metrics["temperature"] = temperature
    metrics["checkpoint"] = str(ckpt)

    out_json = Path("reports/csv") / f"final_metrics_{model_key}_{input_mode}.json"
    save_metrics_json(metrics, out_json)
    save_evaluation_report({model_key: metrics}, Path("reports") / "final_evaluation.txt")

    # ── serve-time manifest update (temperature travels with the model) ──────
    manifest = {
        "active_model": model_key,
        "input_mode": input_mode,
        "checkpoint": str(ckpt),
        "temperature": temperature,
        "metrics_digest": {
            "test_acc": metrics["accuracy"],
            "test_acc_ci": metrics["accuracy_ci"],
            "test_auc": metrics["auc_roc"],
            "f1": metrics["f1"],
            "ece": metrics["ece"],
            "n_test": metrics["n_test"],
        },
        "git_state": "see git log",
    }
    target_manifest = (
        Path(manifest_path) if manifest_path else Path("reports/evaluation_manifest.json")
    )
    canonical_serving = Path("models/manifest.json")
    if target_manifest.resolve() == canonical_serving.resolve() and canonical_serving.exists():
        try:
            existing = json.loads(canonical_serving.read_text(encoding="utf-8"))
            if existing.get("active_system") == "neurosight-e5" and not force_manifest:
                logger.warning(
                    "Refusing to overwrite active E5 serving manifest at %s without "
                    "--force-manifest. Writing evaluation manifest to "
                    "reports/evaluation_manifest.json instead.",
                    canonical_serving,
                )
                target_manifest = Path("reports/evaluation_manifest.json")
        except Exception:
            pass
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Evaluation manifest saved to %s", target_manifest)

    # ── Grad-CAM evidence for the first test patient ─────────────────────────
    if gradcam and fam == "2d":
        from neurosight.evaluation.gradcam import GradCAM, save_gradcam_grid

        ds = test_loader.dataset
        scans, meta, _label = ds[0]
        cam = GradCAM(model)
        heatmaps = cam.generate(
            scans.unsqueeze(0), meta.unsqueeze(0), torch.ones(1, scans.shape[0])
        )
        save_gradcam_grid(
            scans.unsqueeze(0).cpu(),
            heatmaps,
            f"{model_key}_test_p0",
            int(y_pred[0]),
            int(y_true[0]),
            "reports/figures/gradcam",
        )
        cam.remove_hooks()

    click.echo(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k
                in (
                    "accuracy",
                    "accuracy_ci",
                    "auc_roc",
                    "f1",
                    "ece",
                    "n_test",
                    "input_mode",
                    "temperature",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
