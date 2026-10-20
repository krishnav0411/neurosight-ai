"""trainer.py — model-agnostic training loop with MLflow tracking and checkpoints.

Selection and early stopping run on **validation AUC**, not accuracy — the
classes are imbalanced enough that accuracy hides the interesting failures
(plan T4.1 / 06-ENGINEERING-PLAN).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neurosight.config import TrainingConfig
from neurosight.training.losses import FocalLoss
from neurosight.training.metrics import MetricsTracker
from neurosight.utils.io import save_checkpoint
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Train a variant on train/val loaders; track to MLflow; checkpoint best+last."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        run_name: str = "run",
        class_weights: torch.Tensor | None = None,
        device: torch.device | None = None,
        input_mode: str = "mri_only",
        model_key: str = "3_cnn_lstm",
        accum_steps: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_name = run_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.accum_steps = max(1, accum_steps)  # 3D LSTM: effective batch via accumulation

        self.criterion: nn.Module
        if config.use_focal:
            self.criterion = FocalLoss(gamma=2.0, alpha=class_weights)
        elif class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            min_lr=config.min_lr,
        )
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
        }
        # -1 so the first epoch always checkpoints (a single-class val epoch
        # yields AUC 0.0, which must still be recorded as "best so far")
        self.best_val_auc = -1.0
        self.best_epoch = 0
        self._input_mode = input_mode
        self._model_key = model_key

    # ── MLflow (observability — a tracking outage must not kill a long run) ──

    def _mlflow_start(self) -> Any:
        if not self.config.use_mlflow:
            return None
        try:
            import pathlib as _pathlib

            import mlflow

            uri = self.config.mlflow_tracking_uri
            if uri.startswith("sqlite:///"):
                _pathlib.Path(uri[len("sqlite:///")]).parent.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run(run_name=self.run_name)
            mlflow.log_params(
                {
                    "model_key": self._model_key,
                    "input_mode": self._input_mode,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "batch_size": self.train_loader.batch_size,
                    "epochs": self.config.epochs,
                    "patience": self.config.patience,
                    "seed": self.config.seed,
                    "use_focal": self.config.use_focal,
                }
            )
            return mlflow
        except Exception as exc:
            logger.warning("MLflow unavailable (%s); continuing without tracking", exc)
            return None

    # ── epochs ───────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        tracker = MetricsTracker()
        self.model.train(train)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            if train:
                # clear ONCE before the accumulation loop (and after each step
                # below): zeroing before every backward would discard the
                # previous micro-batches' gradients and make accum_steps a no-op
                self.optimizer.zero_grad()
            for step, (scans, meta, labels, mask) in enumerate(loader, start=1):
                scans, meta, labels, mask = (
                    scans.to(self.device),
                    meta.to(self.device),
                    labels.to(self.device),
                    mask.to(self.device),
                )
                logits = self.model(scans, meta, mask)
                loss = self.criterion(logits, labels)
                if train:
                    scaled = loss / self.accum_steps
                    scaled.backward()
                    if step % self.accum_steps == 0 or step == len(loader):
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                tracker.update(logits, labels, float(loss.item()))
        return tracker.compute()

    def train(self) -> dict[str, list[float]]:
        """Full training run; returns the metrics history."""
        mlflow = self._mlflow_start()
        patience_counter = 0
        start = time.time()

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_auc"].append(val_metrics["auc"])
            self.scheduler.step(val_metrics["auc"])

            if mlflow is not None:
                try:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_metrics["loss"],
                            "train_acc": train_metrics["accuracy"],
                            "val_loss": val_metrics["loss"],
                            "val_acc": val_metrics["accuracy"],
                            "val_auc": val_metrics["auc"],
                        },
                        step=epoch,
                    )
                except Exception as exc:
                    logger.warning("MLflow log_metrics failed: %s", exc)

            improved = val_metrics["auc"] > self.best_val_auc
            if improved:
                self.best_val_auc = val_metrics["auc"]
                self.best_epoch = epoch
                patience_counter = 0
                self._save("best")
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1 or improved:
                logger.info(
                    "E%02d/%d train=%.1f%% val=%.1f%% auc=%.4f best_auc=%.4f (%.1f min)",
                    epoch,
                    self.config.epochs,
                    train_metrics["accuracy"] * 100,
                    val_metrics["accuracy"] * 100,
                    val_metrics["auc"],
                    self.best_val_auc,
                    (time.time() - start) / 60,
                )
            if patience_counter >= self.config.patience:
                logger.info(
                    "early stop at epoch %d (best AUC %.4f @ %d)",
                    epoch,
                    self.best_val_auc,
                    self.best_epoch,
                )
                break

        self._save("last")
        if mlflow is not None:
            try:
                best_ckpt = self._checkpoint_path("best")
                if best_ckpt.exists():
                    mlflow.log_artifact(str(best_ckpt))
                mlflow.log_metric("best_val_auc", self.best_val_auc)
                mlflow.end_run()
            except Exception as exc:
                logger.warning("MLflow finalization failed: %s", exc)
        return self.history

    # ── checkpointing ────────────────────────────────────────────────────────

    def _checkpoint_path(self, which: str) -> Path:
        return Path(self.config.checkpoint_dir) / f"{which}_{self._model_key}_{self.run_name}.pth"

    def _save(self, which: str) -> None:
        save_checkpoint(
            self._checkpoint_path(which),
            self.model,
            optimizer=self.optimizer,
            epoch=self.best_epoch if which == "best" else len(self.history["val_auc"]),
            metrics={"best_val_auc": self.best_val_auc},
            extra={
                "model_key": self._model_key,
                "input_mode": self._input_mode,
                "run_name": self.run_name,
            },
        )

    def load_best(self) -> None:
        """Restore the best checkpoint into the model (eval mode)."""
        from neurosight.utils.io import load_checkpoint

        payload = load_checkpoint(self._checkpoint_path("best"), model=self.model)
        self.model.eval()
        self.best_val_auc = payload["metrics"].get("best_val_auc", self.best_val_auc)


__all__ = ["Trainer"]
