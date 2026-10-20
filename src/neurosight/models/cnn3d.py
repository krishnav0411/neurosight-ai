"""cnn3d.py — volumetric 3D CNN (ported). Input (B, 1, D=64, H=128, W=128)."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurosight.models.base import FeatureExtractorMixin


class _Conv3DBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv3d(cin, cout, k, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN3D(FeatureExtractorMixin, nn.Module):
    """4x Conv3d (32→64→128→256) → GAP → FC(2). Single-volume baseline."""

    def __init__(self, dropout: float = 0.5, **_):
        super().__init__()
        self.conv1 = _Conv3DBlock(1, 32)
        self.conv2 = _Conv3DBlock(32, 64)
        self.conv3 = _Conv3DBlock(64, 128)
        self.conv4 = _Conv3DBlock(128, 256, pool=False)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(256, 2)

    def extract(self, x, meta=None, mask=None):
        # x: (B, 1, D, H, W)
        h = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        return self.drop(self.gap(h).flatten(1))


__all__ = ["CNN3D"]
