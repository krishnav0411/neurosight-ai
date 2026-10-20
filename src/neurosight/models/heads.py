"""heads.py — NeuroSightModel: registry variant + optional conversion head.

The status head (T1) is each variant's own classifier. The conversion head (T2)
shares the variant encoder: it applies a linear map to the same pre-classifier
features, so one forward produces both outputs and the encoder learns both
signals. With ``use_conversion_head=False`` (or ``mri_only`` single-task runs)
``forward`` returns ``(status_logits, None)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neurosight.models.base import FeatureExtractorMixin
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


class NeuroSightModel(nn.Module):
    """Wrap any registry variant to expose status + conversion logits."""

    def __init__(self, variant: FeatureExtractorMixin, use_conversion_head: bool = True):
        super().__init__()
        self.variant = variant
        classifier = variant.cls
        if not isinstance(classifier, nn.Linear):
            raise TypeError(f"variant {type(variant).__name__} has no Linear cls head")
        self.classifier = classifier  # the status head (T1)
        self.conversion: nn.Linear | None = (
            nn.Linear(classifier.in_features, 2) if use_conversion_head else None
        )
        if use_conversion_head:
            logger.info(
                "NeuroSightModel: conversion head attached (%d features)",
                classifier.in_features,
            )

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.variant.extract(x, meta, mask)
        status_logits = self.classifier(features)
        conversion_logits = self.conversion(features) if self.conversion is not None else None
        return status_logits, conversion_logits


__all__ = ["NeuroSightModel"]
