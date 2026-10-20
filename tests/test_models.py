"""Tests for the model registry, mask correctness, and the conversion wrapper (plan T3.1, T3.2).

2D variants require 128x128 input (their FC expects 32*16*16); 3D variants run
on reduced volumes to keep CPU test time sane — the architectures are
input-size agnostic thanks to global pooling.
"""

from __future__ import annotations

import pytest
import torch

from neurosight.config import ModelConfig
from neurosight.models import ABLATION_VARIANTS, NeuroSightModel, build_model, family
from neurosight.models.attention import TemporalAttention
from neurosight.models.base import last_valid, masked_mean

ALL_KEYS = sorted(ABLATION_VARIANTS)
CONFIG = ModelConfig(model_key="3_cnn_lstm", input_mode="mri_only")


def test_registry_has_11_variants() -> None:
    assert len(ALL_KEYS) == 11
    assert family("9_3d_cnn") == "3d"
    assert family("10_3d_lstm") == "3d_lstm"
    assert family("3_cnn_lstm") == "2d"
    with pytest.raises(KeyError, match="unknown model key"):
        family("99_bogus")


def test_build_model_rejects_unknown_and_nonbinary() -> None:
    with pytest.raises(KeyError):
        build_model("nope", CONFIG)
    with pytest.raises(ValueError, match="binary"):
        build_model("3_cnn_lstm", ModelConfig(num_classes=3))


@pytest.mark.parametrize("key", ALL_KEYS)
def test_forward_shapes_all_variants(key: str) -> None:
    model = build_model(key, CONFIG)
    model.eval()
    fam = family(key)
    if fam == "2d":
        x = torch.randn(1, 2, 32, 128, 128)
    elif fam == "3d":
        x = torch.randn(1, 1, 64, 32, 32)
    else:  # 3d_lstm
        x = torch.randn(1, 2, 1, 64, 32, 32)
    meta = torch.randn(1, 2, 8)
    mask = torch.tensor([[1.0, 1.0]])
    with torch.no_grad():
        out = model(x, meta, mask)
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_param_counts_match_reference() -> None:
    """Reference counts from the predecessor codebase / final report (±5%).
    Total parameters (frozen ResNet backbone included) — trainable counts are
    reported separately by the evaluator."""
    refs = {
        "1_cnn_avg": 530_000,
        "3_cnn_lstm": 543_000,
        "6_se": 543_000,
        "9_3d_cnn": 1_160_000,
        "10_3d_lstm": 1_250_000,
        "11_resnet_lstm": 11_000_000,
    }
    for key, ref in refs.items():
        model = build_model(key, CONFIG)
        n = sum(p.numel() for p in model.parameters())
        assert abs(n - ref) / ref < 0.05, f"{key}: {n} vs reference {ref}"


def test_masked_lstm_ignores_padding() -> None:
    """With a mask, padding visits must not change the output.

    Row 0: two real visits. Row 1: same first visit, garbage in step 2 masked out.
    Both rows must equal the single-visit result.
    """
    model = build_model("3_cnn_lstm", CONFIG)
    model.eval()
    v1 = torch.randn(1, 1, 32, 128, 128)
    v2 = torch.randn(1, 1, 32, 128, 128)
    with torch.no_grad():
        single = model(v1, None, None)  # full-sequence path (T=1)

    x = torch.cat([torch.cat([v1, v2], dim=1), torch.cat([v1, torch.zeros_like(v2)], dim=1)])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    with torch.no_grad():
        out = model(x, None, mask)
    # row 1 = v1 + masked padding -> must equal the single-visit result
    assert torch.allclose(out[1], single[0], atol=1e-4), "padding leaked into the output"
    # row 0 = v1, v2 both real -> must differ from v1-only (LSTM consumed visit 2)
    assert not torch.allclose(out[0], single[0], atol=1e-4)


def test_masked_mean_and_last_valid() -> None:
    f = torch.tensor([[[1.0, 1.0], [3.0, 3.0]], [[10.0, 10.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    assert masked_mean(f, mask).tolist() == [[2.0, 2.0], [10.0, 10.0]]
    assert last_valid(f, mask).tolist() == [[3.0, 3.0], [10.0, 10.0]]


def test_temporal_attention_respects_mask() -> None:
    attn = TemporalAttention(8)
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    ctx, weights = attn(x, mask)
    assert weights[0].sum().item() == pytest.approx(1.0)
    assert weights[0, 3].item() == 0.0
    assert weights[1, 0].item() == pytest.approx(1.0)
    assert torch.isfinite(ctx).all()


def test_neurosight_model_wrapper() -> None:
    variant = build_model("3_cnn_lstm", CONFIG)
    wrapped = NeuroSightModel(variant, use_conversion_head=True)
    x = torch.randn(2, 2, 32, 128, 128)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    status, conv = wrapped(x, None, mask)
    assert status.shape == (2, 2)
    assert conv is not None and conv.shape == (2, 2)

    single = NeuroSightModel(build_model("3_cnn_lstm", CONFIG), use_conversion_head=False)
    status, conv = single(x, None, mask)
    assert conv is None


def test_conversion_head_gradients_flow() -> None:
    wrapped = NeuroSightModel(build_model("3_cnn_lstm", CONFIG), use_conversion_head=True)
    x = torch.randn(1, 2, 32, 128, 128)
    mask = torch.ones(1, 2)
    status, conv = wrapped(x, None, mask)
    (status.sum() + conv.sum()).backward()
    enc_grads = [p.grad for n, p in wrapped.named_parameters() if "cnn" in n and p.grad is not None]
    conv_grads = [
        p.grad for n, p in wrapped.named_parameters() if "conversion" in n and p.grad is not None
    ]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)
    assert conv_grads and any(g.abs().sum() > 0 for g in conv_grads)


def test_resnet_frozen_backbone_stays_eval_in_train_mode() -> None:
    """requires_grad=False does not freeze BatchNorm running statistics: after
    model.train() the frozen ResNet backbone must remain in eval mode, or its
    BN state drifts during training (trainer calls model.train(True) per epoch)."""
    import torch.nn as nn

    model = build_model("11_resnet_lstm", CONFIG)
    model.train()
    assert model.training  # the rest of the network trains
    assert not model.backbone.training, "frozen backbone entered train mode: BN drift"
    bns = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)

    unfrozen = build_model(
        "11_resnet_lstm", ModelConfig(model_key="11_resnet_lstm", freeze_backbone=False)
    )
    unfrozen.train()
    assert unfrozen.backbone.training  # a trainable backbone must train normally
    unfrozen.eval()
    assert not unfrozen.backbone.training
