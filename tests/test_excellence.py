"""Tests for the excellence harness: TTA, conformal sets, and the runner CLI (plan 07, T2-T4).

CPU-only and synthetic: TTA models are stubs, loaders are plain batch lists,
and the CLI tests only exercise argument guarding — never real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neurosight.evaluation.evaluator import collect_predictions
from neurosight.excellence.conformal import (
    conformal_rank,
    conformal_threshold,
    fit_conformal,
    prediction_set,
    split_half_coverage,
)
from neurosight.excellence.tta import shift_slices_zerofill, tta_batches, tta_collect_predictions

REPO_ROOT = Path(__file__).resolve().parents[1]


class SumLogits(nn.Module):
    """Stub: 2-class logits from the summed voxel value (linear-on-sum)."""

    def forward(self, scans, meta, mask):
        s = scans.sum(dim=(1, 2, 3, 4))
        return torch.stack([-s, s], dim=1)


# ── TTA (T2) ─────────────────────────────────────────────────────────────────


def test_tta_batches_emits_exactly_four_spec_transforms() -> None:
    x = torch.rand(2, 3, 10, 6, 6)
    views = tta_batches(x)
    assert len(views) == 4
    for view in views:
        assert view.shape == x.shape
    assert torch.equal(views[0], x)
    assert torch.equal(views[1], x.flip(-1))
    # slice members are zero-filled ±2 shifts — NOT circular rolls (random
    # volumes differ from a roll at both boundary edges)
    assert torch.equal(views[2], shift_slices_zerofill(x, 2))
    assert torch.equal(views[3], shift_slices_zerofill(x, -2))
    assert not torch.equal(views[2], torch.roll(x, 2, dims=2))
    assert not torch.equal(views[3], torch.roll(x, -2, dims=2))
    # an asymmetric volume must actually change under the transforms
    assert not torch.allclose(views[0], views[1])
    assert not torch.allclose(views[0], views[2])


def test_tta_slice_shifts_are_zero_filled_not_circular() -> None:
    """Distinct per-slice constants pin the exact shift semantics: content
    moves by exactly ±2, the vacated edge is ZERO, and nothing wraps around.
    A circular roll would place superior anatomy at the inferior boundary —
    anatomically impossible."""
    depth = 8
    x = (
        torch.arange(1, depth + 1, dtype=torch.float32)
        .view(1, 1, depth, 1, 1)
        .expand(1, 1, depth, 4, 4)
        .contiguous()
    )
    views = tta_batches(x)
    plus2, minus2 = views[2], views[3]
    zero_head = torch.zeros(1, 1, 2, 4, 4)

    # +2: the first 2 slices are zeros (no wraparound), content moved by 2
    assert torch.equal(plus2[:, :, :2], zero_head)
    assert torch.equal(plus2[:, :, 2:], x[:, :, :-2])
    # last slice equals the THIRD-FROM-LAST original — a roll would wrap the
    # second original slice (x[:, :, 1]) into it instead
    assert torch.equal(plus2[:, :, -1], x[:, :, -3])
    assert not torch.equal(plus2[:, :, -1], x[:, :, 1])

    # -2: mirror image — the last 2 slices are zeros, content moved by -2
    assert torch.equal(minus2[:, :, -2:], zero_head)
    assert torch.equal(minus2[:, :, :-2], x[:, :, 2:])
    # a roll would wrap the first two original slices into the zeroed tail
    assert not torch.equal(minus2[:, :, -2:], x[:, :, :2])
    # the helper shift itself is exact
    assert torch.equal(shift_slices_zerofill(x, -2), minus2)


def test_tta_hflip_symmetric_volume_is_invariant() -> None:
    half = torch.rand(1, 2, 8, 4, 6)
    x = torch.cat([half, half.flip(-1)], dim=-1)  # symmetric under flip(-1)
    views = tta_batches(x)
    assert torch.allclose(views[1], x)  # the h-flip member returns the volume unchanged


def test_tta_mean_softmax_aggregates_all_views() -> None:
    x = torch.rand(1, 1, 6, 4, 4)
    meta, mask, labels = torch.zeros(1, 1, 8), torch.ones(1, 1), torch.tensor([1])
    loader = [(x, meta, labels, mask)]
    model = SumLogits()

    y_true, y_pred, y_prob = tta_collect_predictions(model, loader, torch.device("cpu"))

    expected = torch.stack(
        [torch.softmax(model(v, meta, mask), dim=1)[:, 1] for v in tta_batches(x)]
    ).mean(dim=0)
    assert y_true.tolist() == [1]
    assert np.allclose(y_prob, expected.numpy())
    assert y_pred.tolist() == (y_prob >= 0.5).astype(int).tolist()


def test_tta_matches_plain_collection_for_invariant_volume() -> None:
    # under ZERO-FILLED shifts the only volume all four views leave unchanged
    # is the all-zero one (any nonzero slice-constant volume loses its edge
    # slices to the boundary zeros by design) — TTA must still reduce to plain
    # collection when the members coincide
    x = torch.zeros(2, 2, 8, 4, 4)
    meta = torch.zeros(2, 2, 8)
    labels = torch.tensor([0, 1])
    mask = torch.ones(2, 2)
    loader = [(x, meta, labels, mask)]
    model = SumLogits()

    plain = collect_predictions(model, loader, torch.device("cpu"))
    tta = tta_collect_predictions(model, loader, torch.device("cpu"))
    for a, b in zip(plain, tta, strict=True):
        assert np.allclose(a, b)


# ── conformal prediction sets (T3) ───────────────────────────────────────────


def test_conformal_threshold_finite_sample_math() -> None:
    assert conformal_threshold(100, 0.9) == pytest.approx(0.91)  # ceil(90.9)/100
    assert conformal_threshold(19, 0.9) == pytest.approx(18 / 19)  # ceil(17.1)=18
    assert conformal_threshold(10, 0.9) == pytest.approx(1.0)  # ceil(9.9)=10 -> 10/10
    assert conformal_threshold(2, 0.95) == 1.0  # ceil(2.85)=3 -> 3/2 clamped to 1


def test_conformal_threshold_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        conformal_threshold(0)
    with pytest.raises(ValueError):
        conformal_threshold(10, coverage=1.5)


def test_conformal_rank_clamps_to_n() -> None:
    """The corrected rank is k = min(ceil((n+1)*coverage), n): the raw
    ceil((n+1)*c) can exceed the sample size (n=6, c=0.9 -> 7), which made the
    old level formula 7/6 > 1 and relied on a post-hoc clamp."""
    assert conformal_rank(100, 0.9) == 91
    assert conformal_rank(19, 0.9) == 18  # ceil(17.1) = 18
    assert conformal_rank(6, 0.9) == 6  # ceil(5.4+... ) = ceil(6.3) = 7 -> clamped to n
    assert conformal_rank(2, 0.95) == 2  # ceil(2.85) = 3 -> clamped to n
    with pytest.raises(ValueError):
        conformal_rank(0)
    with pytest.raises(ValueError):
        conformal_rank(10, coverage=1.5)


def test_fit_conformal_known_scores() -> None:
    val_true = np.array([0, 1, 0, 1, 0])
    val_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    # score = 1 - p(true class): [0.1, 0.1, 0.2, 0.2, 0.3]
    # n=5, c=0.5 -> level ceil(3)/5=0.6 -> 3rd smallest of the sorted scores
    assert fit_conformal(val_probs, val_true, coverage=0.5) == pytest.approx(0.2)
    # c=0.9 -> ceil(5.4)=6 > n -> level clamps to 1.0 -> the maximum score
    assert fit_conformal(val_probs, val_true, coverage=0.9) == pytest.approx(0.3)


def test_fit_conformal_validates_input() -> None:
    with pytest.raises(ValueError):
        fit_conformal(np.array([0.5, 0.5]), np.array([0]))
    with pytest.raises(ValueError):
        fit_conformal(np.array([]), np.array([]))


def test_prediction_set_confident_cases_are_singletons() -> None:
    assert prediction_set(0.98, 0.2) == ["Demented"]
    assert prediction_set(0.02, 0.2) == ["Non-Demented"]


def test_prediction_set_uncertain_case_includes_both_classes() -> None:
    assert prediction_set(0.5, 0.5) == ["Non-Demented", "Demented"]


def test_prediction_set_can_be_empty_below_half_threshold() -> None:
    assert prediction_set(0.5, 0.3) == []


def test_prediction_set_never_empty_when_threshold_at_least_half() -> None:
    for p in np.linspace(0.0, 1.0, 21):
        assert prediction_set(float(p), 0.5), f"empty set for p={p}"


def test_conformal_coverage_guarantee_on_synthetic_sweep() -> None:
    """10k+ draws from a calibrated model: mean empirical coverage >= target - 0.02.

    The finite-sample guarantee is *marginal over calibration sets*, so the
    simulation refits the threshold on several independent calibration sets —
    a single realization can sit below the target by chance.
    """
    rng = np.random.RandomState(0)
    coverages = []
    for _refit in range(8):
        p = rng.uniform(0.0, 1.0, 1750)
        y = (rng.uniform(0.0, 1.0, 1750) < p).astype(int)  # P(y=1|p) = p -> calibrated
        threshold = fit_conformal(p[:500], y[:500], coverage=0.9)
        # nonconformity of the TRUE class on fresh draws: 1 - p(true)
        scores_new = np.where(y[500:] == 1, 1.0 - p[500:], p[500:])
        coverages.append(float((scores_new <= threshold).mean()))
    coverage = float(np.mean(coverages))
    assert 0.0 < min(coverages) and coverage >= 0.88


# ── runner CLI guard (T4) ────────────────────────────────────────────────────


def _load_cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "excellence_cli", REPO_ROOT / "scripts" / "excellence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bake_off_refused_without_acknowledgement() -> None:
    from click.testing import CliRunner

    result = CliRunner().invoke(_load_cli().main, ["--bake-off"])
    assert result.exit_code != 0
    assert "--i-know-this-touches-the-test-set" in result.output


def test_bake_off_acknowledgement_passes_the_guard() -> None:
    from click.testing import CliRunner

    # the ack satisfies the guard; the run then fails later on the bogus
    # config dir (never on the guard) — real data is never touched here
    result = CliRunner().invoke(
        _load_cli().main,
        ["--bake-off", "--i-know-this-touches-the-test-set", "--config-dir", "no/such/configs"],
    )
    assert "--i-know-this-touches-the-test-set" not in result.output


def test_systems_to_compute_s3_alone_does_not_imply_s2() -> None:
    """S3 computes its OWN cross-variant predictions with TTA — requesting S3
    alone must not also run the S2 non-TTA ensemble (double compute, and the
    S2 row would silently appear in the comparison table)."""
    cli = _load_cli()
    assert cli.systems_to_compute(["S3"]) == {"S3"}
    assert cli.systems_to_compute(["S2"]) == {"S2"}
    assert cli.systems_to_compute(["S2", "S3"]) == {"S2", "S3"}
    assert cli.systems_to_compute(["S1"]) == {"S1"}
    assert cli.systems_to_compute(["S1", "S2", "S3"]) == {"S1", "S2", "S3"}
    assert cli.systems_to_compute([]) == set()


def test_cli_help_lists_contract_options() -> None:
    from click.testing import CliRunner

    result = CliRunner().invoke(_load_cli().main, ["--help"])
    assert result.exit_code == 0
    for flag in (
        "--systems",
        "--model",
        "--n-folds",
        "--no-tta",
        "--baseline-auc",
        "--out",
        "--run-suffix",
    ):
        assert flag in result.output


def test_split_half_coverage_measures_on_disjoint_rows() -> None:
    """T6's coverage must not be measured on the rows the threshold was fit on
    (calibration-set coverage is optimistically biased): even rows calibrate,
    odd rows are scored."""
    rng = np.random.RandomState(3)
    p = rng.uniform(0.0, 1.0, 600)
    y = (rng.uniform(0.0, 1.0, 600) < p).astype(int)  # calibrated model
    out = split_half_coverage(p, y, coverage=0.9)
    assert out["calib_n"] == 300 and out["eval_n"] == 300
    assert out["coverage"] is not None and out["coverage"] >= 0.85

    # a fold too small to support an honest split reports None, never in-sample
    small = split_half_coverage(p[:3], y[:3], coverage=0.9)
    assert small["coverage"] is None

    with pytest.raises(ValueError):
        split_half_coverage(np.array([0.5, 0.5]), np.array([0]))
