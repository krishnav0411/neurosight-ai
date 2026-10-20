"""bakeoff_e5.py — the single declared held-out test evaluation of NeuroSight-E5.

Plan 07 T7 guardrail: the test set is touched exactly once here, for the
declared final system. NeuroSight-E5 = mean softmax over 5 seed-trained
`7_dual_attn` final models (seeds 42-46) + temperature fitted on the selection
split + split-conformal prediction sets calibrated on the same selection split.

Outputs: reports/excellence/bakeoff_e5.json (+ .txt) and models/manifest.json
update. Also recomputes champion v1 (seed-42 single model) test predictions for
McNemar's test — no new information leaves this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
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
    bootstrap_ci,
    compute_metrics,
    mcnemar_test,
    save_metrics_json,
)
from neurosight.excellence.conformal import fit_conformal, prediction_set
from neurosight.models.registry import build_model
from neurosight.utils.io import load_checkpoint
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed

logger = get_logger(__name__)

CLASS_LABELS = ["Non-Demented", "Demented"]
SEED_CHECKPOINTS = {
    "s42": "best_7_dual_attn_final_7_dual_attn_mri_only.pth",
    "s43": "best_7_dual_attn_final_7_dual_attn_mri_only_s43.pth",
    "s44": "best_7_dual_attn_final_7_dual_attn_mri_only_s44.pth",
    "s45": "best_7_dual_attn_final_7_dual_attn_mri_only_s45.pth",
    "s46": "best_7_dual_attn_final_7_dual_attn_mri_only_s46.pth",
}


def collect_logits(model, loader, device):
    """Return (y_true, logits) for a loader."""
    y_true, logits_all = [], []
    model.eval()
    with torch.no_grad():
        for scans, meta, labels, mask in loader:
            out = model(scans.to(device), meta.to(device), mask.to(device))
            logits_all.append(out.cpu())
            y_true.extend(labels.numpy())
    return np.array(y_true), torch.cat(logits_all)


def main() -> None:
    cfg = load_config("configs")
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = SessionCache(cfg.data.processed_dir, mode="2d")

    loader = OASIS2Loader(cfg.data)
    splits = loader.split_sequences()
    test_records = splits["test"]
    logger.info("held-out test set: %d sequences (single declared touch)", len(test_records))

    # Replicate train_final_model's fit/selection split exactly — the selection
    # half is the calibration set for temperature + conformal (never the test).
    train_records = splits["train"] + splits["val"]
    sel_labels = [r.label for r in train_records]
    _, selection_records = train_test_split(
        train_records,
        test_size=0.15,
        random_state=42,
        stratify=sel_labels if len(set(sel_labels)) > 1 else None,
    )
    logger.info("calibration (selection) split: %d sequences", len(selection_records))

    def eval_loader(records):
        ds = Oasis2DDataset(
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
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_2d,
            num_workers=cfg.training.num_workers,
        )

    models: dict[str, torch.nn.Module] = {}
    models_dir = Path(cfg.training.checkpoint_dir)
    for seed, name in SEED_CHECKPOINTS.items():
        path = models_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing ensemble member: {path}")
        m = build_model("7_dual_attn", cfg.model)
        load_checkpoint(path, model=m)
        models[seed] = m.to(device).eval()

    sel_true, sel_logits = {}, {}
    test_true, test_logits = {}, {}
    for seed, m in models.items():
        yt, lg = collect_logits(m, eval_loader(selection_records), device)
        sel_true[seed], sel_logits[seed] = yt, lg
        yt, lg = collect_logits(m, eval_loader(test_records), device)
        test_true[seed], test_logits[seed] = yt, lg

    def ensemble_softmax(logits: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([torch.softmax(lg, dim=1) for lg in logits.values()]).mean(0)

    sel_probs_t = ensemble_softmax(sel_logits)
    test_probs_t = ensemble_softmax(test_logits)
    assert (sel_true["s42"] == sel_true["s43"]).all() and (
        test_true["s42"] == test_true["s43"]
    ).all(), "ensemble members saw different eval orders"
    y_sel, y_test = sel_true["s42"], test_true["s42"]

    # temperature fitted on the selection ensemble's logits
    scaler = fit_temperature(torch.stack(list(sel_logits.values())).sum(0), torch.tensor(y_sel))
    temperature = float(scaler.temperature.item())

    def calibrated(probs_t: torch.Tensor) -> np.ndarray:
        raw = probs_t[:, 1].numpy()
        return scaler.apply_probs(raw) if temperature != 1.0 else raw

    sel_cal = calibrated(sel_probs_t)
    test_cal = calibrated(test_probs_t)
    test_raw = test_probs_t[:, 1].numpy()
    test_pred = (test_cal >= 0.5).astype(int)

    ece_raw = expected_calibration_error(y_test, test_raw)
    ece_scaled = expected_calibration_error(y_test, test_cal)

    # split-conformal: fit on selection (even rows), coverage measured on test
    conf_threshold = fit_conformal(sel_cal, y_sel, coverage=0.9)
    sets = [prediction_set(float(p), conf_threshold) for p in test_cal]
    covered = sum(1 for p, s in zip(y_test, sets, strict=True) if CLASS_LABELS[p] in s)
    conformal_coverage = covered / len(y_test)
    set_sizes = np.mean([len(s) for s in sets])

    # champion v1 = seed-42 single model (its first/only test touch was M5)
    v1_probs_t = torch.softmax(test_logits["s42"], dim=1)
    v1_cal = (
        scaler.apply_probs(v1_probs_t[:, 1].numpy())
        if temperature != 1.0
        else v1_probs_t[:, 1].numpy()
    )
    v1_pred = (v1_cal >= 0.5).astype(int)
    m_chi2, m_p = mcnemar_test(y_test, test_pred, v1_pred)

    acc_mean, acc_lo, acc_hi = bootstrap_ci(
        y_test, test_pred, lambda t, p_: float((t == p_).mean())
    )
    metrics = compute_metrics(
        y_test,
        test_pred,
        test_cal,
        model_name="NeuroSight-E5 (5-seed ensemble)",
        input_mode=cfg.model.input_mode,
        ece=ece_scaled,
    )
    metrics.update(
        {
            "ece_raw": ece_raw,
            "temperature": temperature,
            "conformal_threshold": conf_threshold,
            "conformal_coverage": conformal_coverage,
            "conformal_mean_set_size": float(set_sizes),
            "mcnemar_vs_champion_v1": {"chi2": m_chi2, "p": m_p},
            "v1_test_accuracy": float((v1_pred == y_test).mean()),
            "members": list(SEED_CHECKPOINTS),
            "test_n": len(y_test),
            "accuracy_ci": [acc_lo, acc_hi],
            "accuracy_mean_boot": acc_mean,
        }
    )

    out_dir = Path("reports/excellence")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, out_dir / "bakeoff_e5.json")

    # manifest update — the deployed system
    manifest = {
        "active_system": "neurosight-e5",
        "model_key": "7_dual_attn",
        "input_mode": cfg.model.input_mode,
        "members": {s: str(models_dir / n) for s, n in SEED_CHECKPOINTS.items()},
        "temperature": temperature,
        "conformal_threshold": conf_threshold,
        "metrics_digest": {
            "test_acc": metrics["accuracy"],
            "test_acc_ci": metrics["accuracy_ci"],
            "test_auc": metrics["auc_roc"],
            "f1": metrics["f1"],
            "ece": ece_scaled,
            "ece_raw": ece_raw,
            "conformal_coverage": conformal_coverage,
            "n_test": len(y_test),
        },
    }
    Path("models/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in ("confusion_matrix",)},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
