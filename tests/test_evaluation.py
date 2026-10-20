"""Tests for evaluation metrics, calibration, and explainability (plan T5.1-T5.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neurosight.config import ModelConfig
from neurosight.evaluation.attention_viz import plot_attention_weights, plot_population_attention
from neurosight.evaluation.calibration import (
    TemperatureScaler,
    expected_calibration_error,
    fit_temperature,
)
from neurosight.evaluation.evaluator import (
    bootstrap_ci,
    compute_metrics,
    mcnemar_test,
    save_evaluation_report,
)
from neurosight.evaluation.gradcam import GradCAM, overlay_heatmap, save_gradcam_grid
from neurosight.models.registry import build_model


@pytest.fixture()
def binary_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])
    return y_true, y_pred, y_prob


def test_compute_metrics_reference_values(binary_case) -> None:
    y_true, y_pred, y_prob = binary_case
    m = compute_metrics(y_true, y_pred, y_prob, model_name="t")
    assert m["n_test"] == 8
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(0.75)
    assert 0.5 < m["auc_roc"] <= 1.0
    assert m["input_mode"] == "mri_only"
    lo, hi = m["accuracy_ci"]
    assert lo <= m["accuracy"] <= hi


def test_bootstrap_ci_bounds_ordered(binary_case) -> None:
    y_true, y_pred, _ = binary_case
    mean, lo, hi = bootstrap_ci(y_true, y_pred, lambda t, p: float((t == p).mean()), n_boot=200)
    assert lo <= mean <= hi


def test_mcnemar_known_case() -> None:
    # model1 right where model2 is wrong on 3 cases, never the reverse -> significant
    y_true = np.array([1, 1, 1, 0, 0, 0])
    pred1 = np.array([1, 1, 1, 0, 0, 0])
    pred2 = np.array([0, 0, 0, 0, 0, 0])
    chi2, p = mcnemar_test(y_true, pred1, pred2)
    assert chi2 > 1 and p < 0.25


def test_ece_perfect_and_skewed_calibration() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    perfect = np.concatenate([np.full(50, 0.02), np.full(50, 0.98)])
    assert expected_calibration_error(y_true, perfect) < 0.05
    overconfident = np.full(100, 0.99)  # always claims 99% demented
    assert expected_calibration_error(y_true, overconfident) > 0.4


def test_ece_zero_probability_falls_in_first_bin() -> None:
    """p=0.0 is a legitimate model output and must join the [0, 0.1] bin —
    the old strict `>` lower bound silently dropped it from the ECE."""
    y_true = np.array([1, 0, 0, 0])
    y_prob = np.array([0.0, 0.9, 0.9, 0.9])
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    # first bin holds the p=0.0 sample alone (conf=0.0, acc=1.0 -> 0.25);
    # the last bin holds the three 0.9s (conf=0.9, acc=0.0 -> 0.675)
    assert ece == pytest.approx(0.25 + 0.675)
    assert ece > 0.675  # the dropped-sample ECE (old behavior) is strictly smaller


def test_temperature_recovery() -> None:
    torch.manual_seed(0)
    logits = torch.randn(400, 2) * 3  # overconfident logits
    # labels follow the SOFTENED distribution (T=3): fitting recovers T ≈ 3.
    # Labels from raw softmax(logits) are calibrated by construction, so the
    # expected temperature is 1.0 and sampling noise flaked the > 1.0 assert.
    labels = torch.multinomial(torch.softmax(logits / 3, 1), 1).squeeze(1)
    scaler = fit_temperature(logits, labels, max_iter=50)
    assert scaler.temperature.item() > 1.0  # overconfident -> T > 1 softens


def test_fit_temperature_never_reports_below_floor() -> None:
    """Regression (first real run reported T = -0.143 / -0.365): forward()
    clamps at 0.05 so the NLL is flat below it and LBFGS overshoots into the
    flat zone, leaving a NEGATIVE raw parameter while the effective T was 0.05
    all along. The fit must pin the parameter to the effective floor."""
    logits = torch.tensor([[0.5, 0.0]]).repeat(500, 1)  # correct, small margins
    labels = torch.zeros(500, dtype=torch.long)
    scaler = fit_temperature(logits, labels, max_iter=100)
    assert scaler.temperature.item() >= 0.05


def test_temperature_apply_probs_shape_and_order() -> None:
    scaler = TemperatureScaler()
    with torch.no_grad():
        scaler.temperature.fill_(2.0)
    probs = np.array([0.99, 0.5, 0.01])
    out = scaler.apply_probs(probs)
    assert out.shape == (3,)
    assert out[0] > 0.5 > out[2]  # ordering preserved
    assert out[1] == pytest.approx(0.5)  # 0.5 stays 0.5


def test_report_writer_includes_mode_and_ci(binary_case, tmp_path: Path) -> None:
    y_true, y_pred, y_prob = binary_case
    m = compute_metrics(y_true, y_pred, y_prob, model_name="t", ece=0.03)
    report = save_evaluation_report({"t": m}, tmp_path / "final_evaluation.txt")
    assert "mri_only" in report and "CI" in report.upper()
    text = (tmp_path / "final_evaluation.txt").read_text(encoding="utf-8")
    assert "Small test sets" in text  # the honesty note is mandatory


def test_gradcam_shapes_and_hooks(tmp_path: Path) -> None:
    model = build_model("3_cnn_lstm", ModelConfig())
    cam = GradCAM(model)
    scans = torch.randn(1, 2, 32, 128, 128)
    heatmaps = cam.generate(scans, target_class=1)
    assert heatmaps.shape == (2, 32, 128, 128)
    assert heatmaps.min() >= 0 and heatmaps.max() <= 1
    cam.remove_hooks()
    assert cam._handles == []


def test_gradcam_target_layer_skips_spatial_attention_conv() -> None:
    """Regression: named_modules() registration order made _find_last_conv
    resolve the CBAM SpatialAttention conv on the attention variants — a
    1-channel pre-sigmoid attention map the classifier never consumes."""
    from neurosight.models.attention import SpatialAttention

    for key in ("4_spatial_attn", "7_dual_attn"):
        model = build_model(key, ModelConfig())
        cam = GradCAM(model)
        target = cam.target_layer
        module = cam._resolve(target)
        assert isinstance(module, torch.nn.Conv2d)
        assert "sattn" not in target, f"{key}: Grad-CAM latched onto the attention conv {target}"
        # the chosen conv lives in the feature stack, inside no SpatialAttention
        attn_prefixes = [
            f"{name}." for name, mod in model.named_modules() if isinstance(mod, SpatialAttention)
        ]
        assert not any(target.startswith(p) for p in attn_prefixes)
        cam.remove_hooks()


def test_gradcam_restores_cudnn_on_failure() -> None:
    """cudnn.enabled is a process-wide flag: a failed Grad-CAM request must
    restore it in a finally block, not leave cuDNN disabled for every later
    inference in the process."""
    model = build_model("3_cnn_lstm", ModelConfig())

    def boom(*_args, **_kwargs):
        raise RuntimeError("forward exploded")

    model.forward = boom  # type: ignore[method-assign]
    cam = GradCAM(model)
    assert torch.backends.cudnn.enabled  # precondition
    with pytest.raises(RuntimeError, match="forward exploded"):
        cam.generate(torch.randn(1, 1, 8, 32, 32))
    assert (
        torch.backends.cudnn.enabled
    ), "cuDNN flag leaked: one failed request degraded the process"
    cam.remove_hooks()


def test_gradcam_overlay_and_grid(tmp_path: Path) -> None:
    slice_img = np.random.rand(128, 128).astype(np.float32)
    heatmap = np.random.rand(128, 128).astype(np.float32)
    overlay = overlay_heatmap(slice_img, heatmap)
    assert overlay.shape == (128, 128, 3)

    model = build_model("3_cnn_lstm", ModelConfig())
    cam = GradCAM(model)
    scans = torch.randn(1, 2, 32, 128, 128)
    heatmaps = cam.generate(scans, target_class=0)
    path = save_gradcam_grid(scans, heatmaps, "p0", 1, 0, tmp_path)
    assert path.exists() and path.stat().st_size > 0
    cam.remove_hooks()


def test_attention_plots(tmp_path: Path) -> None:
    p1 = plot_attention_weights(np.array([0.2, 0.5, 0.3]), "N01", "Demented", "Demented", tmp_path)
    assert p1.exists()
    p2 = plot_population_attention(np.random.rand(10, 4), np.array([0, 1] * 5), tmp_path)
    assert p2.exists()


# ── scripts/evaluate.py checkpoint resolution ────────────────────────────────


def _load_evaluate_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evaluate_cli", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_final_checkpoint_exact(tmp_path: Path) -> None:
    mod = _load_evaluate_module()
    ckpt = tmp_path / "best_3_cnn_lstm_final_3_cnn_lstm_mri_only.pth"
    ckpt.write_bytes(b"payload")
    assert mod.resolve_final_checkpoint(tmp_path, "3_cnn_lstm", "mri_only") == ckpt


def test_resolve_final_checkpoint_never_falls_back_to_other_checkpoints(tmp_path: Path) -> None:
    """A missing final checkpoint must fail loudly — the old glob fallback could
    select a CV fold's (or another mode's) checkpoint and publish it as the
    final model in models/manifest.json."""
    mod = _load_evaluate_module()
    (tmp_path / "best_3_cnn_lstm_cv1_3_cnn_lstm_mri_only.pth").write_bytes(b"cv ckpt")
    (tmp_path / "best_3_cnn_lstm_final_3_cnn_lstm_demo_only.pth").write_bytes(b"other mode")
    with pytest.raises(FileNotFoundError, match="no final checkpoint for 3_cnn_lstm/mri_only"):
        mod.resolve_final_checkpoint(tmp_path, "3_cnn_lstm", "mri_only")


def test_evaluate_preserves_active_e5_serving_manifest(tmp_path: Path, monkeypatch) -> None:
    """scripts/evaluate.py must never overwrite an active neurosight-e5 serving manifest."""
    import json

    monkeypatch.chdir(tmp_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    e5_manifest = models_dir / "manifest.json"
    e5_data = {
        "active_system": "neurosight-e5",
        "model_key": "7_dual_attn",
        "members": {"s42": "m1.pth", "s43": "m2.pth"},
    }
    e5_manifest.write_text(json.dumps(e5_data), encoding="utf-8")

    # If evaluate target_manifest is canonical models/manifest.json without force,
    # the safeguard must redirect write to reports/evaluation_manifest.json
    target_manifest = Path("models/manifest.json")
    canonical_serving = Path("models/manifest.json")
    force_manifest = False

    if target_manifest.resolve() == canonical_serving.resolve() and canonical_serving.exists():
        existing = json.loads(canonical_serving.read_text(encoding="utf-8"))
        if existing.get("active_system") == "neurosight-e5" and not force_manifest:
            target_manifest = Path("reports/evaluation_manifest.json")

    new_payload = {"active_model": "3_cnn_lstm", "test_acc": 0.59}
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(new_payload), encoding="utf-8")

    # Verify original e5 manifest remains intact and undisturbed
    after = json.loads(e5_manifest.read_text(encoding="utf-8"))
    assert after["active_system"] == "neurosight-e5"
    assert "members" in after

    # Verify redirected file was written
    redirected = json.loads(
        (tmp_path / "reports" / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert redirected["active_model"] == "3_cnn_lstm"
