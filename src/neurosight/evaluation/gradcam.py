"""gradcam.py — Grad-CAM and saliency for the CNN variants (ported).

Hooks ``cnn.conv3`` (last conv block before the FC head) on the 2D family.
Grad-CAM answers "which regions drove this prediction" — the clinician-facing
evidence surface (PRD FR-7 / plan T5.3).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.ndimage import zoom as scipy_zoom

from neurosight.utils.logging import get_logger

logger = get_logger(__name__)

KEY_SLICES = [10, 16, 24]  # brain-core slices used in reports and the API


class GradCAM:
    """Grad-CAM for the 2D model family.

    Resolves the target layer automatically: the last Conv2d under the ``cnn``
    submodule (its activations are the highest-resolution spatial map the
    classifier sees). A specific ``target_layer`` name overrides the search —
    attention variants nest their conv blocks differently, so the hardcoded
    ``cnn.conv3`` path broke on the champion (7_dual_attn).
    """

    def __init__(self, model: torch.nn.Module, target_layer: str | None = None):
        self.model = model
        self.model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        if target_layer is None:
            target_layer = self._find_last_conv()
            logger.info("Grad-CAM target layer resolved: %s", target_layer)
        self.target_layer = target_layer
        module = self._resolve(target_layer)
        self._handles = [
            module.register_forward_hook(self._save_activation),
            module.register_full_backward_hook(self._save_gradient),
        ]

    def _find_last_conv(self) -> str:
        # CBAM's SpatialAttention owns a 1-channel pre-sigmoid conv whose output
        # the classifier never consumes; named_modules() registration order used
        # to make cnn.sattn.conv the last "cnn*" Conv2d match on the attention
        # variants (4_spatial_attn / 7_dual_attn), so Grad-CAM latched onto it.
        # Imported here (not at module level) to avoid a circular import.
        from neurosight.models.attention import SpatialAttention

        attn_prefixes = tuple(
            f"{name}."
            for name, mod in self.model.named_modules()
            if isinstance(mod, SpatialAttention)
        )
        last_conv = None
        for name, mod in self.model.named_modules():
            if not name.startswith("cnn") or not isinstance(mod, torch.nn.Conv2d):
                continue
            if attn_prefixes and name.startswith(attn_prefixes):
                continue
            last_conv = name
        if last_conv is None:
            raise AttributeError("no Conv2d found under 'cnn' for Grad-CAM")
        return last_conv

    def _resolve(self, dotted: str) -> Any:
        module: Any = self.model
        for part in dotted.split("."):
            module = getattr(module, part)
        return module

    def _save_activation(self, _m, _i, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _m, _gi, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        scans: torch.Tensor,
        meta: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute heatmaps for a single study.

        Args:
            scans: (1, T, S, H, W)
        Returns:
            (T, S, H, W) float32 heatmaps in [0, 1]
        """
        if self.activations is not None:
            self.activations = None  # fresh forward, fresh capture
        device = next(self.model.parameters()).device
        scans = scans.to(device)
        meta_t = meta.to(device) if meta is not None else None
        mask_t = mask.to(device) if mask is not None else None

        self.model.zero_grad()
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False  # deterministic backward for CAM
        try:
            logits = self.model(scans, meta_t, mask_t)
            if self.activations is None:
                raise RuntimeError("forward hook did not fire — check target_layer")
            cls = int(logits.argmax(1).item()) if target_class is None else target_class
            one_hot = torch.zeros(1, logits.size(1), device=device)
            one_hot[0, cls] = 1.0
            logits.backward(gradient=one_hot, retain_graph=True)
        finally:
            # process-wide flag: a failed request must never leave cuDNN
            # disabled for every later inference in the process
            torch.backends.cudnn.enabled = cudnn_enabled

        if self.activations is None or self.gradients is None:
            raise RuntimeError("no activations/gradients captured")

        # activations: (B*T*S, C, h, w); gradient-weighted sum over channels
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))  # (B*T*S, h, w)
        B, T, S, H, W = scans.shape
        cam = cam.view(B, T, S, *cam.shape[-2:])[0].cpu().numpy()

        heatmaps = np.zeros((T, S, H, W), dtype=np.float32)
        for t in range(T):
            for s in range(S):
                c = cam[t, s]
                span = c.max() - c.min()
                if span > 1e-8:
                    c = (c - c.min()) / span
                heatmaps[t, s] = scipy_zoom(c, (H / c.shape[0], W / c.shape[1]), order=3)
        return np.clip(heatmaps, 0, 1)

    def __del__(self):  # hook hygiene
        try:
            self.remove_hooks()
        except Exception:
            pass


def overlay_heatmap(slice_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Jet-colored heatmap blended over the normalized grayscale slice."""
    sn = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
    rgb = np.stack([sn] * 3, axis=-1)
    cmap = plt.colormaps["jet"]
    heat_rgb = cmap(np.clip(heatmap, 0, 1))[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat_rgb, 0, 1)


def save_gradcam_grid(
    scans: torch.Tensor,
    heatmaps: np.ndarray,
    patient_id: str,
    pred_class: int,
    true_label: int | None,
    save_dir: str | Path,
    visit_labels: list[str] | None = None,
) -> Path:
    """Rows = visits, columns = original / heatmap / overlay at the middle slice."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    T, S = heatmaps.shape[0], heatmaps.shape[1]
    mid = S // 2

    fig, axes = plt.subplots(min(T, 4), 3, figsize=(10, 3 * min(T, 4)), squeeze=False)
    for t in range(min(T, 4)):
        slice_img = (
            scans[0, t, mid].cpu().numpy() if scans.dim() == 5 else scans[t, mid].cpu().numpy()
        )
        overlay = overlay_heatmap(slice_img, heatmaps[t, mid])
        label = visit_labels[t] if visit_labels and t < len(visit_labels) else f"Visit {t + 1}"
        axes[t][0].imshow(slice_img, cmap="gray")
        axes[t][0].set_title(f"{label} — slice {mid + 1}", fontsize=9)
        axes[t][1].imshow(heatmaps[t, mid], cmap="jet", vmin=0, vmax=1)
        axes[t][1].set_title("Grad-CAM", fontsize=9)
        axes[t][2].imshow(overlay)
        axes[t][2].set_title("Overlay", fontsize=9)
        for c in range(3):
            axes[t][c].axis("off")
    pred = "Demented" if pred_class else "Non-Demented"
    true = "?" if true_label is None else ("Demented" if true_label else "Non-Demented")
    fig.suptitle(f"{patient_id} | true={true} pred={pred}", fontsize=12)
    fig.tight_layout()
    path = out / f"gradcam_{patient_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Grad-CAM grid saved: %s", path)
    return path


__all__ = ["KEY_SLICES", "GradCAM", "overlay_heatmap", "save_gradcam_grid"]
