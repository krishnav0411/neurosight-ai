"""attention.py — CBAM spatial attention and masked temporal attention (ported).

TemporalAttention normalises over valid visits only when a mask is supplied —
the ported version softmaxed over padding too.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurosight.models.base import FeatureExtractorMixin, pack_lstm


class SpatialAttention(nn.Module):
    """CBAM spatial attention: avg+max pool → Conv2d → sigmoid → scale."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class TemporalAttention(nn.Module):
    """Additive attention over visits; returns (context, weights).

    With a mask, padding positions get score -inf before the softmax so they
    receive exactly zero weight.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        scores = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask <= 0, float("-inf"))
        weights = F.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1), weights


class _AttentionCNN(nn.Module):
    """Same conv stack as PlainCNN plus CBAM spatial attention before the FC."""

    def __init__(self, dropout: float):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.sattn = SpatialAttention()
        self.fc = nn.Sequential(nn.Linear(32 * 16 * 16, 64), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)
        return self.fc(self.sattn(f).flatten(1))


class CNN_LSTM_SpatialAttn(FeatureExtractorMixin, nn.Module):
    """CNN + SpatialAttention → LSTM (packed) → last valid step → FC2."""

    def __init__(self, dropout: float = 0.5, lstm_hidden: int = 32, **_):
        super().__init__()
        self.cnn = _AttentionCNN(dropout)
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True)
        self.cls = nn.Linear(lstm_hidden, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        out, _ = pack_lstm(self.lstm, f, mask)
        if mask is None:
            return out[:, -1]
        lengths = mask.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        return out.gather(1, idx).squeeze(1)


class CNN_LSTM_TemporalAttn(FeatureExtractorMixin, nn.Module):
    """CNN → LSTM (packed) → masked TemporalAttention → FC2."""

    def __init__(self, dropout: float = 0.5, lstm_hidden: int = 32, **_):
        super().__init__()
        from neurosight.models.cnn_lstm import PlainCNN

        self.cnn = PlainCNN(dropout)
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True)
        self.tattn = TemporalAttention(lstm_hidden)
        self.cls = nn.Linear(lstm_hidden, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        out, _ = pack_lstm(self.lstm, f, mask)
        ctx, _ = self.tattn(out, mask)
        return ctx


class CNN_LSTM_DualAttn(FeatureExtractorMixin, nn.Module):
    """CNN + SpatialAttn → LSTM → masked TemporalAttn → FC2."""

    def __init__(self, dropout: float = 0.5, lstm_hidden: int = 32, **_):
        super().__init__()
        self.cnn = _AttentionCNN(dropout)
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True)
        self.tattn = TemporalAttention(lstm_hidden)
        self.cls = nn.Linear(lstm_hidden, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        out, _ = pack_lstm(self.lstm, f, mask)
        ctx, _ = self.tattn(out, mask)
        return ctx


__all__ = [
    "CNN_LSTM_DualAttn",
    "CNN_LSTM_SpatialAttn",
    "CNN_LSTM_TemporalAttn",
    "SpatialAttention",
    "TemporalAttention",
]
