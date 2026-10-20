"""cnn3d_lstm.py — 3D CNN per visit → LSTM over visits (ported, mask-aware).

Input (B, T, 1, D=64, H=128, W=128)."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurosight.models.base import FeatureExtractorMixin, pack_lstm
from neurosight.models.cnn3d import _Conv3DBlock


class CNN3DEncoder(nn.Module):
    """Shared volumetric encoder: (N, 1, 64, 128, 128) → (N, 256)."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.conv1 = _Conv3DBlock(1, 32)
        self.conv2 = _Conv3DBlock(32, 64)
        self.conv3 = _Conv3DBlock(64, 128)
        self.conv4 = _Conv3DBlock(128, 256, pool=False)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        return self.drop(self.gap(h).flatten(1))


class CNN3D_LSTM(FeatureExtractorMixin, nn.Module):
    """3D CNN per volume → packed LSTM → last valid step → FC2."""

    def __init__(self, dropout: float = 0.5, lstm_hidden: int = 64, **_):
        super().__init__()
        self.encoder = CNN3DEncoder(dropout)
        self.lstm = nn.LSTM(256, lstm_hidden, batch_first=True)
        self.cls = nn.Linear(lstm_hidden, 2)

    def extract(self, x, meta=None, mask=None):
        B, T = x.shape[0], x.shape[1]
        flat = x.reshape(B * T, *x.shape[2:])
        if flat.dim() == 4:  # (N, D, H, W) channelless batches — add the channel
            flat = flat.unsqueeze(1)
        f = self.encoder(flat).view(B, T, -1)
        out, _ = pack_lstm(self.lstm, f, mask)
        if mask is None:
            return out[:, -1]
        lengths = mask.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        return out.gather(1, idx).squeeze(1)


__all__ = ["CNN3D_LSTM", "CNN3DEncoder"]
