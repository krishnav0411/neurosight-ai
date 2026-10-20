"""transformer.py — CNN + Transformer temporal encoder and ResNet transfer (ported, mask-aware)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from neurosight.models.attention import TemporalAttention
from neurosight.models.base import FeatureExtractorMixin, last_valid, masked_mean, pack_lstm
from neurosight.models.cnn_lstm import PlainCNN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe  # registered buffer; mypy sees Tensor | Module
        assert isinstance(pe, torch.Tensor)
        return x + pe[:, : x.size(1)]


class CNN_Transformer(FeatureExtractorMixin, nn.Module):
    """CNN → mean(slices) → PositionalEncoding → TransformerEncoder (key-padding
    mask) → masked TemporalAttn → FC2."""

    def __init__(
        self, dropout: float = 0.5, d_model: int = 64, nhead: int = 4, num_layers: int = 2, **_
    ):
        super().__init__()
        self.cnn = PlainCNN(dropout)
        self.proj = nn.Linear(64, d_model) if d_model != 64 else nn.Identity()
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.tattn = TemporalAttention(d_model)
        self.cls = nn.Linear(d_model, 2)

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        f = self.cnn(x.view(B * T * S, 1, H, W)).view(B * T, S, -1).mean(1).view(B, T, -1)
        f = self.pos_enc(self.proj(f))
        key_padding = None if mask is None else (mask <= 0)
        f = self.transformer(f, src_key_padding_mask=key_padding)
        ctx, _ = self.tattn(f, mask)
        return ctx


class ResNet_LSTM(FeatureExtractorMixin, nn.Module):
    """Transfer learning: pretrained ResNet18 backbone → LSTM over visits → FC2.

    Grayscale slices are replicated to 3 channels; ImageNet weights are reused
    (frozen by default) because 32-slice MRI texture transfer beats training the
    small custom CNN from scratch only sometimes — the ablation measures it.
    """

    def __init__(
        self, dropout: float = 0.5, lstm_hidden: int = 32, freeze_backbone: bool = True, **_
    ):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet18(weights="DEFAULT" if hasattr(models, "ResNet18_Weights") else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (N, 512, 4, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True)
        self.cls = nn.Linear(lstm_hidden, 2)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> ResNet_LSTM:
        """Keep the frozen backbone in eval mode through train().

        ``requires_grad=False`` stops gradients, NOT BatchNorm running-statistic
        updates: without this override the frozen backbone's BN state drifts
        every epoch (the trainer calls ``model.train(True)``).
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def extract(self, x, meta=None, mask=None):
        B, T, S, H, W = x.shape
        x_in = x.view(B * T * S, 1, H, W).repeat(1, 3, 1, 1)
        f = self.gap(self.backbone(x_in)).flatten(1)
        f = self.fc(f).view(B * T, S, -1).mean(1).view(B, T, -1)
        out, _ = pack_lstm(self.lstm, f, mask)
        if mask is None:
            return out[:, -1]
        return last_valid(out, mask)


__all__ = ["CNN_Transformer", "PositionalEncoding", "ResNet_LSTM", "masked_mean"]
