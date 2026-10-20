"""logging.py — central logger factory: rotating file + console.

Every NeuroSight module logs via :func:`get_logger`; ``print()`` is banned inside
the package (ruff rule T201). Log format: ``[TIMESTAMP][LEVEL][module] message``.
File output rotates at 10 MB with 3 backups under ``$NEUROSIGHT_LOG_DIR`` (default ``logs/``).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUPS = 3

# Names already wired with handlers — get_logger must stay idempotent so modules
# can call it at import time without duplicating output.
_CONFIGURED: set[str] = set()


def get_logger(
    name: str,
    level: int | str | None = None,
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """Return a logger writing to console and ``<log_dir>/neurosight.log``.

    Idempotent: repeated calls with the same ``name`` never add handlers.
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        if level is not None:
            logger.setLevel(level)
        return logger

    if log_dir is not None:
        resolved_dir = Path(log_dir)
    else:
        resolved_dir = Path(os.getenv("NEUROSIGHT_LOG_DIR", "logs"))

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    logger.setLevel(level if level is not None else logging.INFO)

    try:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            resolved_dir / "neurosight.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUPS,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # a read-only working directory must never break module imports:
        # degrade to console-only logging instead of raising
        logging.getLogger(__name__).warning(
            "file logging unavailable (%s) — logging to console only", exc
        )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    _CONFIGURED.add(name)
    return logger


__all__ = ["get_logger"]
