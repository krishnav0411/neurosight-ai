"""registry.py — the 11-variant ablation registry (ported from the predecessor's
ablation.py) with family mapping and config-driven construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from neurosight.config import ModelConfig
from neurosight.models.attention import (
    CNN_LSTM_DualAttn,
    CNN_LSTM_SpatialAttn,
    CNN_LSTM_TemporalAttn,
)
from neurosight.models.cnn3d import CNN3D
from neurosight.models.cnn3d_lstm import CNN3D_LSTM
from neurosight.models.cnn_lstm import CNN_LSTM, CNN_Avg, CNN_Last
from neurosight.models.se_cnn import CNN_LSTM_SE
from neurosight.models.transformer import CNN_Transformer, ResNet_LSTM

Family = str  # "2d" | "3d" | "3d_lstm"


@dataclass(frozen=True)
class VariantSpec:
    cls: Callable[..., nn.Module]
    family: Family
    desc: str


ABLATION_VARIANTS: dict[str, VariantSpec] = {
    "1_cnn_avg": VariantSpec(CNN_Avg, "2d", "CNN-avg (no LSTM)"),
    "2_cnn_last": VariantSpec(CNN_Last, "2d", "CNN-last (single visit)"),
    "3_cnn_lstm": VariantSpec(CNN_LSTM, "2d", "CNN+LSTM"),
    "4_spatial_attn": VariantSpec(CNN_LSTM_SpatialAttn, "2d", "CNN+LSTM+SpatialAttn"),
    "5_temporal_attn": VariantSpec(CNN_LSTM_TemporalAttn, "2d", "CNN+LSTM+TemporalAttn"),
    "6_se": VariantSpec(CNN_LSTM_SE, "2d", "CNN+LSTM+SE"),
    "7_dual_attn": VariantSpec(CNN_LSTM_DualAttn, "2d", "CNN+LSTM+DualAttn"),
    "8_transformer": VariantSpec(CNN_Transformer, "2d", "CNN+Transformer"),
    "9_3d_cnn": VariantSpec(CNN3D, "3d", "3D-CNN"),
    "10_3d_lstm": VariantSpec(CNN3D_LSTM, "3d_lstm", "3D-CNN+LSTM"),
    "11_resnet_lstm": VariantSpec(ResNet_LSTM, "2d", "ResNet18+LSTM"),
}


def build_model(key: str, config: ModelConfig) -> nn.Module:
    """Instantiate a registered variant from a ModelConfig.

    Every variant accepts the full hyperparameter set via ``**_``, so the same
    kwargs go to all of them; each uses what it needs (``freeze_backbone``
    matters only for ResNet, ``d_model``/``nhead`` only for the Transformer).
    """
    if key not in ABLATION_VARIANTS:
        raise KeyError(f"unknown model key '{key}'; registered: {sorted(ABLATION_VARIANTS)}")
    if config.num_classes != 2:
        raise ValueError(
            f"registry variants are binary classifiers (num_classes=2), got {config.num_classes}"
        )
    spec = ABLATION_VARIANTS[key]
    return spec.cls(
        dropout=config.dropout,
        lstm_hidden=config.lstm_hidden,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        freeze_backbone=config.freeze_backbone,
    )


def family(key: str) -> Family:
    if key not in ABLATION_VARIANTS:
        raise KeyError(f"unknown model key '{key}'")
    return ABLATION_VARIANTS[key].family


__all__ = ["ABLATION_VARIANTS", "VariantSpec", "build_model", "family"]
