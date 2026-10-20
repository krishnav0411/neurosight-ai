"""cnn_lstm.py — plain CNN baseline variants (ported, mask-aware).

Per-slice CNN: Conv2d(1→8→16→32) + BN + ReLU + MaxPool x3 → FC(8192→64).
All variants expose ``extract``/``forward`` (see base.FeatureExtractorMixin).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neurosight.models.base import FeatureExtractorMixin, last_valid, masked_mean, pack_lstm


class PlainCNN(nn.Module):
    """3x Conv2d(1→8→16→32)+BN+ReLU+MaxPool → FC(8192→64). Input (N,1,128,128)."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x.flatten(1))


class CNN_Avg(FeatureExtractorMixin, nn.Module):
    """CNN → mean(slices) → mean(visits) → FC2. No temporal modelling (baseline)."""

    def __init__(self, dropout: float = 0.5, **_):
        super().__init__()
        self.cnn = PlainCNN(dropout)
        self.cls = nn.Linear(64, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        if mask is not None:
            return masked_mean(f, mask)
        return f.mean(1)


class CNN_Last(FeatureExtractorMixin, nn.Module):
    """CNN → mean(slices) → LAST VALID visit → FC2."""

    def __init__(self, dropout: float = 0.5, **_):
        super().__init__()
        self.cnn = PlainCNN(dropout)
        self.cls = nn.Linear(64, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        if mask is not None:
            return last_valid(f, mask)
        return f[:, -1]


class CNN_LSTM(FeatureExtractorMixin, nn.Module):
    """CNN → mean(slices) → LSTM (packed) → last valid step → FC2."""

    def __init__(self, dropout: float = 0.5, lstm_hidden: int = 32, **_):
        super().__init__()
        self.cnn = PlainCNN(dropout)
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True)
        self.cls = nn.Linear(lstm_hidden, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        out, _lengths = pack_lstm(self.lstm, f, mask)
        if mask is None:
            return out[:, -1]
        return last_valid(out, mask)


__all__ = ["CNN_LSTM", "CNN_Avg", "CNN_LSTM_Masked", "CNN_Last", "PlainCNN"]

# Backward-compatible alias: the masked behaviour is now the only behaviour
CNN_LSTM_Masked = CNN_LSTM
