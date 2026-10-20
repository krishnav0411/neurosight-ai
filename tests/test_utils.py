"""Tests for logging, seeding, device, and io utilities (plan T0.3)."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neurosight.utils.device import get_device
from neurosight.utils.io import ensure_dir, load_checkpoint, save_checkpoint
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed


def test_set_seed_reproducible() -> None:
    set_seed(42)
    torch_a, np_a, py_a = torch.randn(3), np.random.rand(3), random.random()
    set_seed(42)
    torch_b, np_b, py_b = torch.randn(3), np.random.rand(3), random.random()
    assert torch.equal(torch_a, torch_b)
    assert np.array_equal(np_a, np_b)
    assert py_a == py_b  # random.seed() is pinned too — np.random can't prove that


def test_set_seed_pins_cudnn_flags() -> None:
    set_seed(7, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    set_seed(7, deterministic=False)
    assert torch.backends.cudnn.benchmark is True
    set_seed(7, deterministic=True)  # restore for other tests


def test_get_logger_writes_rotating_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NEUROSIGHT_LOG_DIR", str(tmp_path))
    name = "neurosight.test.file"
    logger = get_logger(name)
    logger.warning("hello-from-test")
    log_file = tmp_path / "neurosight.log"
    assert log_file.exists()
    assert "hello-from-test" in log_file.read_text(encoding="utf-8")
    assert "[WARNING]" in log_file.read_text(encoding="utf-8")


def test_get_logger_idempotent_handlers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NEUROSIGHT_LOG_DIR", str(tmp_path))
    name = "neurosight.test.idem"
    logger = get_logger(name)
    n_handlers = len(logger.handlers)
    again = get_logger(name)
    assert again is logger
    assert len(logger.handlers) == n_handlers
    assert logger.propagate is False


def test_get_logger_format_has_module_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NEUROSIGHT_LOG_DIR", str(tmp_path))
    logger = get_logger("neurosight.test.fmt")
    logger.error("boom")
    text = (tmp_path / "neurosight.log").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if "boom" in ln)
    assert line.startswith("[") and "][ERROR][" in line and "neurosight.test.fmt" in line


def test_log_level_environment_fallback() -> None:
    # Without NEUROSIGHT_LOG_DIR set, the default 'logs/' dir must be created lazily.
    import os

    old = os.environ.pop("NEUROSIGHT_LOG_DIR", None)
    try:
        logger = get_logger("neurosight.test.defaultdir")
        assert isinstance(logger, logging.Logger)
    finally:
        if old is not None:
            os.environ["NEUROSIGHT_LOG_DIR"] = old


def test_get_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "cpu"}


def test_ensure_dir(tmp_path: Path) -> None:
    out = ensure_dir(tmp_path / "a" / "b")
    assert out.is_dir()
    assert ensure_dir(out) == out  # idempotent


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model_a = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model_a.parameters(), lr=0.1)
    path = tmp_path / "ck" / "best.pth"
    save_checkpoint(path, model_a, optimizer=optimizer, epoch=3, metrics={"acc": 0.5})

    model_b = nn.Linear(4, 2)
    payload = load_checkpoint(path, model=model_b)
    assert payload["epoch"] == 3
    assert payload["metrics"] == {"acc": 0.5}
    for a, b in zip(model_a.state_dict().values(), model_b.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_load_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "nope.pth")


def test_get_logger_unwritable_log_dir_falls_back_to_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only (or blocked) log directory must never break module imports:
    get_logger falls back to console-only logging instead of raising."""
    from logging.handlers import RotatingFileHandler

    blocker = tmp_path / "blocked"  # a FILE where the log dir must be created
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("NEUROSIGHT_LOG_DIR", str(blocker))
    logger = get_logger("neurosight.test.readonly")
    assert logger.handlers, "console logging must survive"
    assert not any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    logger.warning("console-only line")  # must not raise
