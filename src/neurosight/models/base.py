"""base.py — shared pieces for all NeuroSight model variants.

Mask discipline (the predecessor's silent bug): every temporal model must
honour the visit mask — the old variants read ``out[:, -1]``, i.e. the last
*padding* step for short sequences. All variants here take ``mask`` and use
the helpers below.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def last_valid(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather the last valid timestep per sample.

    Args:
        features: (B, T, F)
        mask: (B, T) with 1.0 = real visit, 0.0 = padding
    """
    lengths = mask.sum(dim=1).long().clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, features.size(-1))
    return features.gather(1, idx).squeeze(1)


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid timesteps only: (B, T, F) + (B, T) -> (B, F)."""
    m = mask.unsqueeze(-1)
    return (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class FeatureExtractorMixin(nn.Module):
    """Uniform two-part interface: ``extract`` returns pre-classifier features,
    ``forward`` classifies them. Lets the conversion head (heads.py) reuse any
    variant's encoder without duplicating its body."""

    def extract(
        self, x: torch.Tensor, meta: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, meta: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # cls is a nn.Linear assigned by every concrete variant (mypy can't see
        # it on the mixin — this is the mixin's whole contract)
        classifier = self.cls  # type: ignore[attr-defined]
        return classifier(self.extract(x, meta, mask))  # type: ignore[operator]


def pack_lstm(lstm: nn.LSTM, features: torch.Tensor, mask: torch.Tensor | None):
    """Run an LSTM with proper packing when a mask is present.

    Returns (output_padded_to_T, lengths). Without a mask the full sequence is
    processed (all steps valid) — the caller guarantees that case.
    """
    if mask is None:
        out, _ = lstm(features)
        return out, torch.full((features.size(0),), features.size(1), dtype=torch.long)
    lengths = mask.sum(dim=1).long().cpu().clamp(min=1)
    packed = nn.utils.rnn.pack_padded_sequence(
        features, lengths, batch_first=True, enforce_sorted=False
    )
    out, _ = lstm(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=features.size(1))
    return out, lengths


__all__ = ["FeatureExtractorMixin", "last_valid", "masked_mean", "pack_lstm"]
