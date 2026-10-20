"""Tests for losses, metrics, trainer, and the CV driver (plan T4.1, T4.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import make_mini_df, make_study_records, write_raw_tree
from neurosight.config import DataConfig, TrainingConfig
from neurosight.data.dataset import make_dataloaders
from neurosight.data.loader import OASIS2Loader
from neurosight.models.registry import build_model
from neurosight.training import cv as cv_module
from neurosight.training.cv import run_cv, train_final_model
from neurosight.training.losses import FocalLoss, get_class_weights
from neurosight.training.metrics import MetricsTracker
from neurosight.training.trainer import Trainer


def test_focal_loss_zero_gamma_equals_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    focal = FocalLoss(gamma=0.0)(logits, targets)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    assert focal.item() == pytest.approx(ce.item(), rel=1e-5)


def test_focal_loss_down_weights_easy_examples() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])  # confident (easy)
    targets = torch.tensor([0, 1])
    focal = FocalLoss(gamma=2.0)(logits, targets)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    assert 0.0 < focal.item() < ce.item()


def test_focal_loss_alpha_scales_classes() -> None:
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1])
    plain = FocalLoss(gamma=0.0)(logits, targets)
    weighted = FocalLoss(gamma=0.0, alpha=torch.tensor([2.0, 1.0]))(logits, targets)
    assert weighted.item() > plain.item()  # class-0 examples weigh more


def test_class_weights_inverse_frequency() -> None:
    labels = [0] * 9 + [1]
    w = get_class_weights(labels)
    assert w.shape == (2,)
    assert w[1] > w[0]  # rare class weighs more
    assert w.mean().item() == pytest.approx(1.0)


def test_class_weights_absent_class_is_finite() -> None:
    w = get_class_weights([0, 0, 0], num_classes=2)
    assert torch.isfinite(w).all()


def test_metrics_tracker() -> None:
    tracker = MetricsTracker()
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]])
    labels = torch.tensor([0, 1, 0])
    tracker.update(logits, labels, loss=0.5)
    m = tracker.compute()
    assert m["accuracy"] == pytest.approx(1.0)  # argmax([2,1]) == 0 == label
    assert 0.5 < m["auc"] <= 1.0
    tracker.reset()
    assert tracker.compute()["accuracy"] == 0.0


def test_metrics_tracker_single_class_auc_is_zero() -> None:
    tracker = MetricsTracker()
    tracker.update(torch.tensor([[3.0, 0.0]]), torch.tensor([0]), loss=0.1)
    assert tracker.compute()["auc"] == 0.0


def _mini_training_config(tmp_path: Path, **overrides) -> TrainingConfig:
    defaults: dict = {
        "epochs": 3,
        "batch_size": 2,
        "patience": 5,
        "use_mlflow": False,
        "checkpoint_dir": tmp_path / "checkpoints",
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_trainer_learns_on_fixture(tmp_path: Path, configs_dir: Path) -> None:
    """End-to-end smoke: the training loop can fit the 2-sequence fixture."""
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    records = make_study_records(tmp_path)
    train_loader, val_loader, _ = make_dataloaders(
        records, records[:1], None, "2d", cfg.data, cfg.training
    )
    trainer = Trainer(
        build_model("3_cnn_lstm", cfg.model),
        train_loader,
        val_loader,
        # patience=40 disables early stopping: 2 samples = 1 step/epoch, and a
        # 0.5M-param model needs dozens of steps to fit them
        _mini_training_config(tmp_path, epochs=60, patience=60),
        run_name="pytest",
        class_weights=get_class_weights([r.label for r in records]),
        model_key="3_cnn_lstm",
    )
    history = trainer.train()
    assert len(history["val_auc"]) == 60
    # the loop genuinely learns: 2 tiny samples get perfectly fit at some point
    assert max(history["train_acc"]) == 1.0
    best = tmp_path / "checkpoints" / "best_3_cnn_lstm_pytest.pth"
    assert best.exists()
    trainer.load_best()  # best checkpoint restores cleanly


def test_trainer_gradient_accumulation(tmp_path: Path, configs_dir: Path) -> None:
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    records = make_study_records(tmp_path)
    train_loader, val_loader, _ = make_dataloaders(
        records, records[:1], None, "2d", cfg.data, cfg.training
    )
    trainer = Trainer(
        build_model("3_cnn_lstm", cfg.model),
        train_loader,
        val_loader,
        _mini_training_config(tmp_path, epochs=1),
        run_name="accum",
        accum_steps=2,
        model_key="3_cnn_lstm",
    )
    history = trainer.train()
    assert len(history["train_loss"]) == 1  # one epoch completes without error


def _mini_loader(tmp_path: Path) -> OASIS2Loader:
    """Six-subject loader over loadable tiny scans (shared by CV smoke tests)."""
    import nibabel as nib

    df = make_mini_df()
    roots = write_raw_tree(df, tmp_path / "raw")
    # the raw tree holds empty name-level placeholders; the training run opens
    # scans, so drop a real (tiny) loadable .nii beside each — policy 'first'
    # sorts mpr-1 < mpr-1.nifti and picks the loadable one
    for root in roots:
        for session in root.iterdir():
            raw_dir = session / "RAW"
            if raw_dir.is_dir():
                vol = (
                    np.random.RandomState(hash(session.name) % 2**31)
                    .rand(32, 32, 48)
                    .astype(np.float32)
                )
                nib.save(nib.Nifti1Image(vol, np.eye(4)), raw_dir / "mpr-1.nii")
    return OASIS2Loader(
        DataConfig(
            csv_path=tmp_path / "mini.csv",
            raw_paths=[str(r) for r in roots],
            processed_dir=tmp_path / "processed",
            uploads_dir=tmp_path / "uploads",
        ),
        df=df,
        raw_paths=roots,
    )


def _mini_cfg(tmp_path: Path, configs_dir: Path, **training_overrides):
    from neurosight.config import NeuroSightConfig, load_config

    cfg: NeuroSightConfig = load_config(configs_dir)
    defaults: dict = {
        "epochs": 1,
        "n_folds": 2,
        "use_mlflow": False,
        "checkpoint_dir": tmp_path / "checkpoints",
    }
    defaults.update(training_overrides)
    return cfg.model_copy(update={"training": cfg.training.model_copy(update=defaults)})


def test_run_cv_smoke(tmp_path: Path, configs_dir: Path) -> None:
    cfg = _mini_cfg(tmp_path, configs_dir, epochs=2)
    loader = _mini_loader(tmp_path)
    summary = run_cv(
        cfg, loader, "3_cnn_lstm", n_folds=2, max_epochs_override=2, out_dir=tmp_path / "reports"
    )
    assert summary["n_folds"] == 2 and len(summary["fold_results"]) == 2
    assert 0.0 <= summary["mean_auc"] <= 1.0
    assert (tmp_path / "reports" / "cv_3_cnn_lstm_mri_only.json").exists()


def test_run_cv_seed_suffix_keeps_checkpoints_and_reports_distinct(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--seed enabler: seed threading + suffixed run names (no checkpoint collisions)."""
    cfg = _mini_cfg(tmp_path, configs_dir)
    seeds: list[int] = []
    monkeypatch.setattr(cv_module, "set_seed", lambda seed, deterministic=True: seeds.append(seed))

    loader = _mini_loader(tmp_path)
    summary = run_cv(
        cfg,
        loader,
        "3_cnn_lstm",
        n_folds=2,
        max_epochs_override=1,
        out_dir=tmp_path / "reports",
        seed_override=43,
        run_suffix="_s43",
    )
    assert summary["n_folds"] == 2
    assert seeds == [43, 43]  # the seed override reached every fold's set_seed
    assert (tmp_path / "reports" / "cv_3_cnn_lstm_mri_only_s43.json").exists()
    ckpts = sorted((tmp_path / "checkpoints").glob("best_3_cnn_lstm_cv*_s43.pth"))
    assert len(ckpts) == 2  # suffixed run names never collide with the plain CV run


def test_train_final_model_seed_suffix_names_checkpoint(tmp_path: Path, configs_dir: Path) -> None:
    cfg = _mini_cfg(tmp_path, configs_dir)
    loader = _mini_loader(tmp_path)
    train_final_model(cfg, loader, "3_cnn_lstm", seed_override=44, run_suffix="_s44")
    expected = tmp_path / "checkpoints" / "best_3_cnn_lstm_final_3_cnn_lstm_mri_only_s44.pth"
    assert expected.exists()


def test_trainer_accumulates_gradients_across_micro_batches(
    tmp_path: Path, configs_dir: Path
) -> None:
    """zero_grad must run BEFORE the accumulation loop and after each optimizer
    step — the old code zeroed before every backward, so each update used only
    the final micro-batch's gradient divided by accum_steps."""
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    training = cfg.training.model_copy(update={"batch_size": 1})  # 2 records -> 2 batches
    cfg = cfg.model_copy(update={"training": training})
    records = make_study_records(tmp_path)
    train_loader, val_loader, _ = make_dataloaders(
        records, records[:1], None, "2d", cfg.data, cfg.training
    )
    trainer = Trainer(
        build_model("3_cnn_lstm", cfg.model),
        train_loader,
        val_loader,
        _mini_training_config(tmp_path, epochs=1),
        run_name="accumspy",
        accum_steps=2,
        model_key="3_cnn_lstm",
    )

    grad_norms_at_loss: list[float | None] = []
    criterion = trainer.criterion

    def spy_criterion(logits, labels):
        g = next(trainer.model.parameters()).grad
        grad_norms_at_loss.append(None if g is None else float(g.norm()))
        return criterion(logits, labels)

    trainer.criterion = spy_criterion  # type: ignore[assignment]

    zero_marks: list[int] = []
    real_zero = trainer.optimizer.zero_grad

    def spy_zero():
        zero_marks.append(len(grad_norms_at_loss))
        real_zero()

    trainer.optimizer.zero_grad = spy_zero  # type: ignore[method-assign]

    trainer._run_epoch(trainer.train_loader, train=True)  # 2 batches, accum_steps=2

    assert grad_norms_at_loss[0] is None, "gradients were not cleared before the loop"
    # the FIRST micro-batch's gradient must still be there when the second
    # backward runs — the old per-backward zero_grad discarded it
    assert grad_norms_at_loss[1] is not None and grad_norms_at_loss[1] > 0.0
    # exactly: once before the loop + once after the single optimizer step
    assert zero_marks == [0, 2]


def test_train_final_model_selects_on_disjoint_split(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final fit must not validate on its own training records: the
    train+val pool is split into fit/selection halves and early stopping runs
    out-of-sample; metrics come back labelled selection_*."""
    cfg = _mini_cfg(tmp_path, configs_dir)
    loader = _mini_loader(tmp_path)

    captured: dict = {}

    def fake_single_fold(cfg_, model_key, train_records, val_records, run_name, **kwargs):
        captured["fit"] = [(r.subject_id, r.kind) for r in train_records]
        captured["selection"] = [(r.subject_id, r.kind) for r in val_records]
        captured["run_name"] = run_name
        return {"best_val_auc": 0.5, "best_epoch": 1, "best_val_acc": 0.5}

    monkeypatch.setattr(cv_module, "train_single_fold", fake_single_fold)
    metrics = train_final_model(cfg, loader, "3_cnn_lstm")

    fit_ids, sel_ids = set(captured["fit"]), set(captured["selection"])
    assert fit_ids and sel_ids
    assert not (fit_ids & sel_ids), "final model validated on records it trained on"
    splits = loader.split_sequences()
    pool_ids = {(r.subject_id, r.kind) for r in splits["train"] + splits["val"]}
    assert fit_ids | sel_ids == pool_ids, "selection split must come from the train+val pool"
    assert captured["run_name"] == "final_3_cnn_lstm_mri_only"
    assert metrics == {"selection_val_auc": 0.5, "selection_epoch": 1, "selection_val_acc": 0.5}


def test_search_disables_class_weights_when_sampler_used(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """use_sampler already balances classes by resampling; inverse-frequency
    class weights on top double-weight minority examples (and diverge from
    train_single_fold, which gates on the same flag)."""
    import neurosight.data.dataset as dataset_module
    import neurosight.training.search as search_module
    from neurosight.config import load_config

    cfg = load_config(configs_dir)
    records = make_study_records(tmp_path)
    received: dict = {}

    class FakeTrainer:
        best_val_auc = 0.5

        def __init__(self, _model, _train, _val, _tcfg, run_name, class_weights, *args, **kwargs):
            received[run_name] = class_weights

        def train(self):
            return {}

    monkeypatch.setattr(search_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        dataset_module, "make_dataloaders", lambda *a, **k: (object(), object(), None)
    )

    class FakeLoader:
        def cv_folds(self, n):
            return [(records, records[:1])]

    monkeypatch.chdir(tmp_path)  # run_search writes reports/csv relative to cwd
    seen: list = []
    for use_sampler in (True, False):
        training = cfg.training.model_copy(update={"use_sampler": use_sampler})
        run_cfg = cfg.model_copy(update={"training": training})
        search_module.run_search(
            run_cfg,
            FakeLoader(),
            model_key="3_cnn_lstm",
            grid_override={"learning_rate": [1e-4], "dropout": [0.3], "lstm_hidden": [32]},
            epochs=1,
        )
        seen.append(received["search1_f1_3_cnn_lstm"])
    assert seen[0] is None, "sampler + class weights double-weight minority examples"
    assert seen[1] is not None  # without the sampler, class weights still apply
