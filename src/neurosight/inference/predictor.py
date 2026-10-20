"""predictor.py — study-level inference for the serving plane (plan T6.1).

Loads a manifest-pinned checkpoint (registry rebuild + forward probe — the
manifest IS the contract between research and serving), runs the same
preprocessing as training (no augmentation), applies the fitted temperature,
and produces the StudyResult payload defined in docs/plan/05-BACKEND-DESIGN.md.

Supports both single-model checkpoints and 5-member ensembles (e.g. NeuroSight-E5)
with exact mathematical aggregation and temperature scaling parity.

TorchScript export is intentionally deferred: variable-length visit sequences
make ``torch.jit.trace`` unsound for these models. The manifest pins
architecture + weights + temperature instead; ONNX with dynamic axes is the
proper export path and is documented as future work.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurosight.config import ModelConfig
from neurosight.data.modes import METADATA_DIM
from neurosight.data.preprocessing import preprocess_2d
from neurosight.models.registry import build_model, family
from neurosight.utils.io import load_checkpoint
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


class StudyCancelledError(RuntimeError):
    """Raised between visits when the caller's ``should_cancel`` fires."""


def apply_temperature(prob: float, temperature: float) -> float:
    """Exact binary temperature rescaling from the softmax ratio."""
    t = max(temperature, 0.05)
    p1 = min(max(prob, 1e-9), 1 - 1e-9)
    z = np.log(p1 / (1 - p1)) / t
    return float(1.0 / (1.0 + np.exp(-z)))


def risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "High"
    if prob >= 0.5:
        return "Moderate"
    return "Low"


def progression_trend(per_visit_probs: list[float]) -> str:
    """Stable | Declining | Rapidly Declining by the largest consecutive rise."""
    if len(per_visit_probs) < 2:
        return "Stable"
    max_rise = max((b - a for a, b in pairwise(per_visit_probs)), default=0.0)
    if max_rise > 0.15:
        return "Rapidly Declining"
    if max_rise > 0.05:
        return "Declining"
    return "Stable"


class NeuroSightPredictor:
    """Manifest-driven predictor shared by the API and offline scripts."""

    def __init__(
        self,
        manifest: dict,
        models_dir: str | Path = "models",
        num_slices: int = 32,
        img_size: int = 128,
    ):
        self.manifest = manifest
        self.model_key: str = manifest.get("model_key") or manifest.get("active_model", "")
        if not self.model_key:
            raise KeyError("manifest missing 'model_key' or 'active_model'")
        self.active_system: str = manifest.get("active_system", "")
        self.input_mode: str = manifest.get("input_mode", "mri_only")
        self.temperature: float = float(manifest.get("temperature", 1.0))
        sha = str(manifest.get("git_sha", "dev"))[:8]
        if self.active_system:
            self.model_version = f"{self.active_system}@{sha}[{self.input_mode}]"
        else:
            self.model_version = f"{self.model_key}@{sha}[{self.input_mode}]"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_slices = num_slices
        self.img_size = img_size

        if family(self.model_key) != "2d":
            raise ValueError(
                f"serving v1 supports 2d models only (got {self.model_key}); "
                "3D serving is documented future work"
            )

        members = manifest.get("members")
        self.models: list[torch.nn.Module] = []
        if members and isinstance(members, dict):
            # Ensemble mode (e.g. NeuroSight-E5 across seeds 42-46)
            for member_key, ckpt_name in members.items():
                m = build_model(
                    self.model_key,
                    ModelConfig(model_key=self.model_key, input_mode=self.input_mode),
                )
                ckpt_path = Path(ckpt_name)
                if not ckpt_path.is_absolute() or not ckpt_path.exists():
                    ckpt_path = Path(models_dir) / ckpt_path.name
                if not ckpt_path.exists() and (Path(models_dir) / ckpt_name).exists():
                    ckpt_path = Path(models_dir) / ckpt_name
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"ensemble member checkpoint not found: {ckpt_name}")
                load_checkpoint(ckpt_path, model=m, map_location="cpu")
                m.to(self.device).eval()

                with torch.no_grad():
                    probe = m(
                        torch.zeros(1, 1, self.num_slices, self.img_size, self.img_size).to(
                            self.device
                        ),
                        torch.zeros(1, 1, METADATA_DIM).to(self.device),
                        torch.ones(1, 1).to(self.device),
                    )
                if probe.shape != (1, 2):
                    raise RuntimeError(
                        f"forward probe failed on member {member_key}: got {tuple(probe.shape)}"
                    )
                self.models.append(m)

            self.model = self.models[0]
            logger.info(
                "ensemble predictor ready: %s (%d members, T=%.3f, device=%s)",
                self.model_version,
                len(self.models),
                self.temperature,
                self.device,
            )
        else:
            # Single checkpoint mode
            m = build_model(
                self.model_key,
                ModelConfig(model_key=self.model_key, input_mode=self.input_mode),
            )
            ckpt_ref = str(manifest.get("checkpoint", ""))
            checkpoint = (
                Path(ckpt_ref) if Path(ckpt_ref).is_absolute() else Path(models_dir) / ckpt_ref
            )
            if not checkpoint.exists() and manifest.get("checkpoint"):
                checkpoint = Path(manifest["checkpoint"])
            load_checkpoint(checkpoint, model=m, map_location="cpu")
            m.to(self.device).eval()

            # forward probe: refuse to serve a broken model
            with torch.no_grad():
                probe = m(
                    torch.zeros(1, 1, self.num_slices, self.img_size, self.img_size).to(
                        self.device
                    ),
                    torch.zeros(1, 1, METADATA_DIM).to(self.device),
                    torch.ones(1, 1).to(self.device),
                )
            if probe.shape != (1, 2):
                raise RuntimeError(f"forward probe failed: got {tuple(probe.shape)}")
            self.model = m
            self.models = [m]
            logger.info(
                "predictor ready: %s (T=%.3f, device=%s)",
                self.model_version,
                self.temperature,
                self.device,
            )

    @classmethod
    def from_manifest_file(cls, manifest_path: str | Path, **kwargs) -> NeuroSightPredictor:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"model manifest not found: {path}")
        models_dir = kwargs.pop("models_dir", path.parent)
        return cls(json.loads(path.read_text(encoding="utf-8")), models_dir=models_dir, **kwargs)

    # ── inference ────────────────────────────────────────────────────────────

    def _visit_stack(self, paths: list[str]) -> np.ndarray:
        stacks = [preprocess_2d(_load(p), self.num_slices, self.img_size, "middle") for p in paths]
        return np.mean(stacks, axis=0) if len(stacks) > 1 else stacks[0]

    # Metadata field order for clinical modes (see neurosight/data/modes.py)
    _DEMOGRAPHIC_ORDER: tuple[str, ...] = (
        "sex",
        "age",
        "educ",
        "ses",
        "mmse",
        "etiv",
        "nwbv",
        "asf",
    )

    def predict_study(
        self,
        visits: list[list[str]],
        demographics: dict[str, float] | None = None,
        visit_labels: list[str] | None = None,
        include_gradcam: bool = False,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Predict on one study: ``visits[t]`` = list of scan paths for visit t.

        Per-visit probabilities come from running the model on each visit
        PREFIX (visit 1; visits 1-2; ...) — how the prediction evolves as
        visits accumulate is the honest longitudinal signal.

        Demographics are accepted but masked per the model's input mode: an
        ``mri_only`` model never sees them (the mask is the leakage contract).

        ``should_cancel`` (when given) is polled BETWEEN visits — inside the
        longitudinal prefix loop and the Grad-CAM loop — so a cancellation is
        honoured promptly instead of after all inference work; it raises
        :class:`StudyCancelledError`.
        """
        if not visits:
            raise ValueError("empty study: no visits supplied")

        stacks = np.stack([self._visit_stack(paths) for paths in visits])
        meta = np.zeros((len(visits), METADATA_DIM), dtype=np.float32)
        if demographics and self.input_mode != "mri_only":
            for i, key in enumerate(self._DEMOGRAPHIC_ORDER):
                if key in demographics:
                    meta[:, i] = float(demographics[key])
        mask = torch.ones(1, len(visits))

        scans = torch.from_numpy(stacks[np.newaxis].astype(np.float32)).to(self.device)
        meta_t = torch.from_numpy(meta[np.newaxis]).to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            member_logits = [m(scans, meta_t, mask) for m in self.models]
            # Softmax per member, then average across ensemble (parity with bakeoff_e5)
            member_probs = [torch.softmax(lg, dim=1) for lg in member_logits]
            raw = torch.stack(member_probs).mean(dim=0)[0].cpu().numpy()

        calib = np.array([apply_temperature(float(p), self.temperature) for p in raw])
        calib = calib / calib.sum()  # renormalise after temperature

        # longitudinal: prediction after each accumulating prefix of visits
        per_visit: list[float] = []
        if len(visits) > 1:
            with torch.no_grad():
                for t in range(1, len(visits) + 1):
                    self._check_cancel(should_cancel)
                    p_mask = torch.ones(1, t, device=self.device)
                    prefix_p1_members = [
                        torch.softmax(m(scans[:, :t], meta_t[:, :t], p_mask), dim=1)[0, 1].item()
                        for m in self.models
                    ]
                    p_mean = float(np.mean(prefix_p1_members))
                    per_visit.append(apply_temperature(p_mean, self.temperature))

        result: dict[str, Any] = {
            "predicted_class": int(calib.argmax()),
            "class_labels": ["Non-Demented", "Demented"],
            "probabilities": {
                "Non-Demented": float(calib[0]),
                "Demented": float(calib[1]),
            },
            "raw_probabilities": {"Non-Demented": float(raw[0]), "Demented": float(raw[1])},
            "confidence": float(calib.max()),
            "confidence_band": "adequate" if calib.max() >= 0.6 else "low",
            "risk_level": risk_level(float(calib[1])),
            "input_mode": self.input_mode,
            "model_version": self.model_version,
            "visit_count": len(visits),
            "visit_labels": visit_labels or [f"MR{i + 1}" for i in range(len(visits))],
            "longitudinal": {
                "trend": progression_trend(per_visit) if len(visits) > 1 else "Stable",
                "per_visit_probability": per_visit,
            },
            "conversion_risk": {"available": False},
            "caveats": ["research_only", "small_test_set"],
        }

        if include_gradcam:
            from neurosight.evaluation.gradcam import KEY_SLICES, GradCAM, overlay_heatmap

            self._check_cancel(should_cancel)
            heatmaps_list = []
            for m in self.models:
                self._check_cancel(should_cancel)
                cam = GradCAM(m)
                try:
                    heatmaps_list.append(cam.generate(scans, meta_t, mask))
                finally:
                    cam.remove_hooks()
            # Ensemble-averaged saliency across all 5 member checkpoints
            heatmaps = np.mean(heatmaps_list, axis=0)

            gradcam_out = []
            labels = result["visit_labels"]
            # iterate VISITS: a caller-supplied label list longer than the
            # study used to IndexError here; short lists fall back to MR<n>
            for t in range(len(visits)):
                self._check_cancel(should_cancel)
                label = labels[t] if t < len(labels) else f"MR{t + 1}"
                images = {
                    str(s): _to_b64(overlay_heatmap(stacks[t, s], heatmaps[t, s]))
                    for s in KEY_SLICES
                    if s < heatmaps.shape[1]
                }
                gradcam_out.append({"visit": label, "slices": sorted(images), "images": images})
            result["gradcam"] = gradcam_out
        return result

    @staticmethod
    def _check_cancel(should_cancel: Callable[[], bool] | None) -> None:
        if should_cancel is not None and should_cancel():
            raise StudyCancelledError("study cancelled between visits")


def _load(path: str) -> np.ndarray:
    import nibabel as nib

    image = nib.load(path)  # type: ignore[no-untyped-call]
    return np.asarray(image.get_fdata(), dtype=np.float32)  # type: ignore[attr-defined]


def _to_b64(rgb: np.ndarray) -> str:
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, format="PNG")
    plt.close("all")
    return base64.b64encode(buf.getvalue()).decode("ascii")


__all__ = [
    "NeuroSightPredictor",
    "StudyCancelledError",
    "apply_temperature",
    "progression_trend",
    "risk_level",
]
