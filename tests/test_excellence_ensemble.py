"""Tests for the same-fold ensemble core and fold-report summaries (plan 07, T1).

Leakage contract: fold i's val subjects are in the TRAINING set of every other
fold's model, so cross-fold averaging would score training data. The only clean
ensemble members for fold i are models that are blind to fold i's val — the
per-variant fold-i checkpoints (different architectures/seeds, same training
subjects). These tests pin that contract.

CPU-only and synthetic: stub checkpoints are real tiny state dicts, the CV
loader is monkeypatched — no real OASIS-2 data, no real checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

import neurosight.excellence.ensemble as ensemble
import neurosight.excellence.report as excellence_report
from neurosight.evaluation import calibration
from neurosight.evaluation.evaluator import collect_predictions
from neurosight.excellence.ensemble import (
    average_variant_probs,
    collect_fold_checkpoints,
    samefold_cross_variant,
    samefold_fold_predictions,
)
from neurosight.excellence.report import evaluate_fold_predictions, format_comparison_table
from neurosight.models.registry import build_model
from neurosight.utils.io import load_checkpoint, save_checkpoint


def _write_ckpt(path: Path) -> Path:
    save_checkpoint(path, torch.nn.Linear(2, 2))
    return path


def _write_seed_set(tmp_path: Path, key: str, mode: str, suffix: str | None, folds: range) -> None:
    """One checkpoint per fold for a run of ``{key}_{mode}[_{suffix}]``."""
    for n in folds:
        name = f"best_{key}_cv{n}_{key}_{mode}" + (f"_{suffix}" if suffix else "") + ".pth"
        _write_ckpt(tmp_path / name)


def _fold_preds(y_true: list[int], y_prob: list[float]) -> dict:
    return {"y_true": np.array(y_true), "y_prob": np.array(y_prob)}


def _stub_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # PlainCNN's FC expects 128x128 slices — the smallest valid stub batch.
    return (
        torch.rand(1, 1, 4, 128, 128),
        torch.zeros(1, 1, 8),
        torch.tensor([1]),
        torch.ones(1, 1),
    )


def _stub_cv_loader(monkeypatch: pytest.MonkeyPatch, batch) -> None:
    """Replace the real data path: N folds, fold i's val = records[i]."""

    class _StubOASIS2Loader:
        def __init__(self, config, df=None, raw_paths=None):
            self.config = config

        def cv_folds(self, n_folds: int):
            records = [object() for _ in range(n_folds)]
            return [
                ([records[j] for j in range(n_folds) if j != i], [records[i]])
                for i in range(n_folds)
            ]

    monkeypatch.setattr(ensemble, "OASIS2Loader", _StubOASIS2Loader)
    monkeypatch.setattr(ensemble, "_eval_loader", lambda records, cfg_, cache: [batch])


def _member_prob(ckpt: Path, key: str, cfg, batch) -> tuple[np.ndarray, np.ndarray]:
    """Independent recomputation of one member's predictions on the stub batch."""
    model = build_model(key, cfg.model)
    load_checkpoint(ckpt, model=model)
    y_true, _pred, y_prob = collect_predictions(model, [batch], torch.device("cpu"))
    return y_true, np.asarray(y_prob, dtype=np.float64)


# ── collect_fold_checkpoints ─────────────────────────────────────────────────


def test_collect_fold_checkpoints_discovers_all_folds(tmp_path: Path) -> None:
    for n in range(1, 6):
        _write_ckpt(tmp_path / f"best_7_dual_attn_cv{n}_7_dual_attn_mri_only.pth")
    found = collect_fold_checkpoints(tmp_path, "7_dual_attn", n_folds=5)
    assert sorted(found) == [1, 2, 3, 4, 5]
    assert all(p.exists() for p in found.values())
    assert found[3].name == "best_7_dual_attn_cv3_7_dual_attn_mri_only.pth"


def test_collect_fold_checkpoints_missing_folds_lists_them(tmp_path: Path) -> None:
    _write_ckpt(tmp_path / "best_7_dual_attn_cv1_7_dual_attn_mri_only.pth")
    _write_ckpt(tmp_path / "best_7_dual_attn_cv3_7_dual_attn_mri_only.pth")
    with pytest.raises(FileNotFoundError) as err:
        collect_fold_checkpoints(tmp_path, "7_dual_attn", n_folds=5)
    message = str(err.value)
    for fold in ("2", "4", "5"):
        assert fold in message
    assert "7_dual_attn" in message


def test_collect_fold_checkpoints_ignores_other_variants_finals_and_extra_folds(
    tmp_path: Path,
) -> None:
    for n in (1, 2):
        _write_ckpt(tmp_path / f"best_7_dual_attn_cv{n}_7_dual_attn_mri_only.pth")
    _write_ckpt(tmp_path / "best_8_transformer_cv1_8_transformer_mri_only.pth")
    _write_ckpt(tmp_path / "best_7_dual_attn_final_7_dual_attn_mri_only.pth")
    _write_ckpt(tmp_path / "best_7_dual_attn_cv9_7_dual_attn_mri_only.pth")
    found = collect_fold_checkpoints(tmp_path, "7_dual_attn", n_folds=2)
    assert sorted(found) == [1, 2]


def test_collect_fold_checkpoints_input_mode_filter(tmp_path: Path) -> None:
    _write_ckpt(tmp_path / "best_7_dual_attn_cv1_7_dual_attn_mri_only.pth")
    demo = _write_ckpt(tmp_path / "best_7_dual_attn_cv1_7_dual_attn_mri_demo.pth")
    found = collect_fold_checkpoints(tmp_path, "7_dual_attn", n_folds=1, input_mode="mri_demo")
    assert found[1] == demo


def test_collect_fold_checkpoints_run_suffix_none_matches_only_unsuffixed(
    tmp_path: Path,
) -> None:
    """Unsuffixed + two seeded reruns share a models directory: run_suffix=None
    must find EXACTLY the unsuffixed set — the old wildcard would match both
    ..._mri_only.pth and ..._mri_only_s43.pth and arbitrarily pick one."""
    key, mode = "7_dual_attn", "mri_only"
    _write_seed_set(tmp_path, key, mode, None, range(1, 6))
    _write_seed_set(tmp_path, key, mode, "s42", range(1, 6))
    _write_seed_set(tmp_path, key, mode, "s43", range(1, 6))

    found = collect_fold_checkpoints(tmp_path, key, n_folds=5, input_mode=mode)

    assert sorted(found) == [1, 2, 3, 4, 5]
    assert all(p.name.endswith(f"{key}_{mode}.pth") for p in found.values())


def test_collect_fold_checkpoints_run_suffix_picks_exactly_that_seed(tmp_path: Path) -> None:
    key, mode = "7_dual_attn", "mri_only"
    _write_seed_set(tmp_path, key, mode, None, range(1, 6))
    _write_seed_set(tmp_path, key, mode, "s42", range(1, 6))
    _write_seed_set(tmp_path, key, mode, "s43", range(1, 6))

    found = collect_fold_checkpoints(tmp_path, key, n_folds=5, input_mode=mode, run_suffix="s43")

    assert sorted(found) == [1, 2, 3, 4, 5]
    assert all(p.name.endswith(f"{key}_{mode}_s43.pth") for p in found.values())


def test_collect_fold_checkpoints_incomplete_seed_set_raises_missing_folds(
    tmp_path: Path,
) -> None:
    """A seed present for only SOME folds must fail loudly — never fall back to
    mixing that seed's folds with another run's checkpoints."""
    key, mode = "7_dual_attn", "mri_only"
    _write_seed_set(tmp_path, key, mode, "s42", range(1, 6))
    _write_seed_set(tmp_path, key, mode, "s43", range(1, 3))  # folds 1-2 only

    with pytest.raises(FileNotFoundError) as err:
        collect_fold_checkpoints(tmp_path, key, n_folds=5, input_mode=mode, run_suffix="s43")
    message = str(err.value)
    for fold in ("3", "4", "5"):
        assert fold in message
    assert "s43" in message


def test_collect_fold_checkpoints_wildcard_mode_ambiguity_raises(tmp_path: Path) -> None:
    """input_mode=None (wildcard mode) with unsuffixed AND seeded files for the
    same fold must fail loudly instead of silently picking one of them."""
    key = "7_dual_attn"
    _write_seed_set(tmp_path, key, "mri_only", None, range(1, 3))
    _write_seed_set(tmp_path, key, "mri_only", "s43", range(1, 3))

    with pytest.raises(ValueError, match="ambiguous"):
        collect_fold_checkpoints(tmp_path, key, n_folds=2)


def test_samefold_fold_predictions_run_suffix_selects_seeded_checkpoints(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    key = "3_cnn_lstm"
    ckpts = {
        n: save_checkpoint(
            tmp_path / f"best_{key}_cv{n}_{key}_mri_only_s43.pth", build_model(key, cfg.model)
        )
        for n in (1, 2)
    }
    _stub_cv_loader(monkeypatch, _stub_batch())

    out = samefold_fold_predictions(
        key,
        cfg,
        cache=None,
        n_folds=2,
        models_dir=str(tmp_path),
        device=torch.device("cpu"),
        run_suffix="s43",
    )
    assert out[0]["checkpoint"] == str(ckpts[1])
    assert out[1]["checkpoint"] == str(ckpts[2])


# ── same-fold predictions (leakage-safe by construction) ─────────────────────


def test_samefold_fold_predictions_uses_own_fold_checkpoint(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    key = "3_cnn_lstm"
    ckpts = {
        n: save_checkpoint(
            tmp_path / f"best_{key}_cv{n}_{key}_mri_only.pth", build_model(key, cfg.model)
        )
        for n in (1, 2)
    }
    batch = _stub_batch()
    _stub_cv_loader(monkeypatch, batch)

    out = samefold_fold_predictions(
        key, cfg, cache=None, n_folds=2, models_dir=str(tmp_path), device=torch.device("cpu")
    )
    assert set(out) == {0, 1}
    # fold i is scored by fold i's OWN checkpoint only (cv{i+1}, 1-based files)
    assert out[0]["checkpoint"] == str(ckpts[1])
    assert out[1]["checkpoint"] == str(ckpts[2])
    y_true, y_prob = _member_prob(ckpts[1], key, cfg, batch)
    assert np.array_equal(out[0]["y_true"], y_true)
    assert np.allclose(out[0]["y_prob"], y_prob)


def test_samefold_cross_variant_members_are_own_fold_checkpoints(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KEY leakage test: fold i's ensemble is exactly the per-variant fold-i
    checkpoints — no cross-fold member is even representable."""
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    keys = ("1_cnn_avg", "3_cnn_lstm")
    ckpts = {
        (key, n): save_checkpoint(
            tmp_path / f"best_{key}_cv{n}_{key}_mri_only.pth", build_model(key, cfg.model)
        )
        for key in keys
        for n in (1, 2)
    }
    batch = _stub_batch()
    _stub_cv_loader(monkeypatch, batch)

    out = samefold_cross_variant(
        list(keys), cfg, cache=None, n_folds=2, models_dir=str(tmp_path), device=torch.device("cpu")
    )
    assert set(out) == {0, 1}
    assert out[0]["members"] == [str(ckpts[("1_cnn_avg", 1)]), str(ckpts[("3_cnn_lstm", 1)])]
    assert out[1]["members"] == [str(ckpts[("1_cnn_avg", 2)]), str(ckpts[("3_cnn_lstm", 2)])]

    # the reported prob is the mean softmax across exactly those members
    y_true, prob_a = _member_prob(ckpts[("1_cnn_avg", 1)], "1_cnn_avg", cfg, batch)
    _, prob_b = _member_prob(ckpts[("3_cnn_lstm", 1)], "3_cnn_lstm", cfg, batch)
    assert np.array_equal(out[0]["y_true"], y_true)
    assert np.allclose(out[0]["y_prob"], (prob_a + prob_b) / 2)


def test_samefold_cross_variant_skips_3d_variants_with_warning(
    tmp_path: Path,
    configs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    key = "3_cnn_lstm"
    for n in (1, 2):
        save_checkpoint(
            tmp_path / f"best_{key}_cv{n}_{key}_mri_only.pth", build_model(key, cfg.model)
        )
    _stub_cv_loader(monkeypatch, _stub_batch())

    with caplog.at_level(logging.WARNING, logger="neurosight.excellence.ensemble"):
        out = samefold_cross_variant(
            [key, "9_3d_cnn"],
            cfg,
            cache=None,
            n_folds=2,
            models_dir=str(tmp_path),
            device=torch.device("cpu"),
        )
    assert all(len(entry["members"]) == 1 for entry in out.values())  # 3D never entered
    assert any("9_3d_cnn" in record.message for record in caplog.records)


def test_samefold_cross_variant_requires_a_2d_variant(configs_dir: Path) -> None:
    from neurosight.config import load_config

    with pytest.raises(ValueError, match="2D"):
        samefold_cross_variant(["9_3d_cnn"], load_config(configs_dir))


def test_samefold_fold_predictions_rejects_non_2d(configs_dir: Path) -> None:
    from neurosight.config import load_config

    with pytest.raises(NotImplementedError, match="2D"):
        samefold_fold_predictions("9_3d_cnn", load_config(configs_dir))


# ── cross-variant averaging ──────────────────────────────────────────────────


def test_average_variant_probs_means_aligned_folds() -> None:
    per_variant = {
        "7_dual_attn": {
            0: _fold_preds([0, 1], [0.2, 0.8]),
            1: _fold_preds([1, 0], [0.6, 0.1]),
        },
        "8_transformer": {
            0: _fold_preds([0, 1], [0.4, 0.6]),
            1: _fold_preds([1, 0], [0.4, 0.3]),
        },
    }
    out = average_variant_probs(per_variant)
    assert set(out) == {0, 1}
    assert out[0]["y_prob"] == pytest.approx([0.3, 0.7])
    assert out[1]["y_prob"] == pytest.approx([0.5, 0.2])
    assert (out[0]["y_true"] == [0, 1]).all()


def test_average_variant_probs_rejects_y_true_mismatch() -> None:
    per_variant = {
        "a": {0: _fold_preds([0, 1], [0.2, 0.8])},
        "b": {0: _fold_preds([1, 0], [0.2, 0.8])},
    }
    with pytest.raises(ValueError, match="y_true"):
        average_variant_probs(per_variant)


def test_average_variant_probs_rejects_fold_coverage_mismatch() -> None:
    per_variant = {
        "a": {0: _fold_preds([0, 1], [0.2, 0.8]), 1: _fold_preds([1, 0], [0.3, 0.4])},
        "b": {0: _fold_preds([0, 1], [0.4, 0.6])},
    }
    with pytest.raises(ValueError, match="fold"):
        average_variant_probs(per_variant)


# ── report summaries ─────────────────────────────────────────────────────────


def test_evaluate_fold_predictions_summarizes() -> None:
    per_fold = {
        0: _fold_preds([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]),
        1: _fold_preds([0, 1, 1, 1], [0.3, 0.6, 0.7, 0.9]),
    }
    summary = evaluate_fold_predictions(per_fold)
    assert summary["n_folds"] == 2
    assert summary["folds"]["0"]["auc"] == pytest.approx(1.0)
    assert 0.0 <= summary["folds"]["1"]["auc"] <= 1.0
    aucs = [summary["folds"]["0"]["auc"], summary["folds"]["1"]["auc"]]
    assert summary["mean_auc"] == pytest.approx(float(np.mean(aucs)))
    assert summary["std_auc"] == pytest.approx(float(np.std(aucs)))
    assert 0.0 <= summary["folds"]["0"]["ece_raw"] <= 1.0
    assert "temperature" in summary["folds"]["0"]


def test_temperature_scaling_reduces_ece_for_overconfident_probs() -> None:
    y_true = np.array([0] * 20 + [1] * 20)
    y_prob = np.full(40, 0.99)  # always claims 99% demented; half wrong
    summary = evaluate_fold_predictions({0: {"y_true": y_true, "y_prob": y_prob}})
    fold = summary["folds"]["0"]
    assert fold["ece_raw"] > 0.4
    assert fold["ece_scaled"] < fold["ece_raw"]  # the fitted T softens
    assert fold["temperature"] >= 0.05  # reported T is the EFFECTIVE floored T


def test_evaluate_fold_predictions_fits_temperature_on_even_index_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-sample scaled ECE: the temperature fit sees ONLY the even-index
    (calibration) rows; the odd rows score the scaled ECE and are never shown
    to the fit (in-sample scaled ECE is optimistically biased)."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    seen_fit: dict[str, np.ndarray] = {}
    seen_ece: list[tuple[np.ndarray, np.ndarray]] = []
    real_fit = calibration.fit_temperature
    real_ece = calibration.expected_calibration_error

    def fit_spy(logits, labels, **kwargs):
        seen_fit["logits"] = logits.numpy()
        seen_fit["labels"] = labels.numpy()
        return real_fit(logits, labels, **kwargs)

    def ece_spy(y_true_, y_prob_, **kwargs):
        seen_ece.append((np.asarray(y_true_), np.asarray(y_prob_)))
        return real_ece(y_true_, y_prob_, **kwargs)

    monkeypatch.setattr(excellence_report, "fit_temperature", fit_spy)
    monkeypatch.setattr(excellence_report, "expected_calibration_error", ece_spy)
    summary = evaluate_fold_predictions({0: {"y_true": y_true, "y_prob": y_prob}})

    # the temperature fit received exactly the even-index rows (labels 0,0,0)
    assert seen_fit["labels"].tolist() == [0, 0, 0]
    assert np.allclose(seen_fit["logits"], excellence_report.probs_to_logits(y_prob[0::2]))
    # raw ECE ran on the full fold; scaled ECE ran on the odd-index half only
    assert seen_ece[0][0].tolist() == [0, 1, 0, 1, 0, 1]
    assert seen_ece[1][0].tolist() == [1, 1, 1]  # y_true[1::2]

    fold = summary["folds"]["0"]
    assert (fold["ece_calib_n"], fold["ece_eval_n"]) == (3, 3)
    assert 0.0 <= fold["ece_scaled"] <= 1.0
    assert fold["temperature"] >= calibration.MIN_TEMPERATURE


def test_evaluate_fold_predictions_tiny_fold_reports_raw_ece_only() -> None:
    """Folds with <4 observations cannot support an honest calibration/eval
    split — ece_scaled and temperature are null; raw ECE still reported."""
    summary = evaluate_fold_predictions(
        {0: {"y_true": np.array([0, 1, 1]), "y_prob": np.array([0.2, 0.8, 0.7])}}
    )
    fold = summary["folds"]["0"]
    assert fold["ece_scaled"] is None
    assert fold["temperature"] is None
    assert 0.0 <= fold["ece_raw"] <= 1.0
    assert (fold["ece_calib_n"], fold["ece_eval_n"]) == (2, 1)
    assert summary["mean_ece_scaled"] is None


def test_evaluate_fold_predictions_mean_scaled_skips_null_folds() -> None:
    big = {"y_true": np.array([0, 1, 0, 1]), "y_prob": np.array([0.2, 0.8, 0.3, 0.7])}
    tiny = {"y_true": np.array([0, 1]), "y_prob": np.array([0.4, 0.6])}
    summary = evaluate_fold_predictions({0: big, 1: tiny})
    assert summary["folds"]["0"]["ece_scaled"] is not None
    assert summary["folds"]["1"]["ece_scaled"] is None
    assert summary["mean_ece_scaled"] == pytest.approx(summary["folds"]["0"]["ece_scaled"])


def test_comparison_table_lists_systems_and_baseline() -> None:
    summaries = {
        "S1": {
            "n_folds": 5,
            "mean_auc": 0.79,
            "std_auc": 0.08,
            "mean_ece_raw": 0.09,
            "mean_ece_scaled": 0.05,
        },
    }
    table = format_comparison_table(summaries, baseline_auc=0.7444, baseline_std=0.0965)
    assert "S0" in table and "S1" in table
    assert "0.7444" in table and "0.0965" in table
    assert "0.7900" in table  # the system row carries its mean AUC


def test_comparison_table_renders_dash_for_null_scaled_ece() -> None:
    """A system whose every fold was too small to split reports a null
    mean_ece_scaled — the table renders '-' instead of crashing."""
    summaries = {
        "S1": {
            "n_folds": 1,
            "mean_auc": 0.5,
            "std_auc": 0.0,
            "mean_ece_raw": 0.1,
            "mean_ece_scaled": None,
        },
    }
    table = format_comparison_table(summaries, baseline_auc=0.5, baseline_std=0.0)
    s1_row = next(line for line in table.splitlines() if line.strip().startswith("S1"))
    assert "0.5000" in s1_row
    assert "-" in s1_row
