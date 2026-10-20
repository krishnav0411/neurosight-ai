"""se_cnn.py — Squeeze-and-Excitation CNN variants (ported, mask-aware)."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurosight.models.attention import TemporalAttention
from neurosight.models.base import FeatureExtractorMixin, pack_lstm


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: GAP → FC(↓r) → ReLU → FC(↑) → sigmoid → scale."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        r = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, r, bias=False),
            nn.ReLU(),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


def _se_conv_block(cin: int, cout: int, dropout_unused: None = None) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        SEBlock(cout),
        nn.MaxPool2d(2),
    )


class _SECNNBackbone(nn.Module):
    """Conv(1→8→16→32, each + SE) → FC(8192→64)."""

    def __init__(self, dropout: float):
        super().__init__()
        self.blocks = nn.Sequential(
            _se_conv_block(1, 8), _se_conv_block(8, 16), _se_conv_block(16, 32)
        )
        self.fc = nn.Sequential(nn.Linear(32 * 16 * 16, 64), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.blocks(x).flatten(1))


class SECNN(FeatureExtractorMixin, nn.Module):
    """CNN+SE → mean(slices) → mean(visits) → FC2 (no temporal modelling)."""

    def __init__(self, dropout: float = 0.5, **_):
        super().__init__()
        self.cnn = _SECNNBackbone(dropout)
        self.cls = nn.Linear(64, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        if mask is None:
            return f.mean(1)
        m = mask.unsqueeze(-1)
        return (f * m).sum(1) / m.sum(1).clamp(min=1.0)


class CNN_LSTM_SE(FeatureExtractorMixin, nn.Module):
    """CNN+SE → LSTM (packed) → last valid step → FC2."""

    def __init__(self, dropout: float = 0.5, lstm_hidden: int = 32, **_):
        super().__init__()
        self.cnn = _SECNNBackbone(dropout)
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


class CNN_LSTM_SE_TemporalAttn(FeatureExtractorMixin, nn.Module):
    """CNN+SE → LSTM → masked TemporalAttention → FC2."""

    def __init__(self, dropout: float = 0.5, lstm_hidden: int = 32, **_):
        super().__init__()
        self.cnn = _SECNNBackbone(dropout)
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True)
        self.tattn = TemporalAttention(lstm_hidden)
        self.cls = nn.Linear(lstm_hidden, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        out, _ = pack_lstm(self.lstm, f, mask)
        ctx, _ = self.tattn(out, mask)
        return ctx


__all__ = ["CNN_LSTM_SE", "SECNN", "CNN_LSTM_SE_TemporalAttn", "SEBlock"]
