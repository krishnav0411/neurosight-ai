"""preprocess.py — CLI: preprocess all sessions of the configured dataset into the cache.

Usage:
    python scripts/preprocess.py                  # 2D cache only
    python scripts/preprocess.py --include-3d     # also build the 3D volume cache
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from neurosight.config import load_config
from neurosight.data.cache import build_session_cache
from neurosight.data.loader import OASIS2Loader
from neurosight.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--config-dir", default="configs", show_default=True)
@click.option("--include-3d", is_flag=True, help="Also build the (1, D, H, W) volume cache.")
def main(config_dir: str, include_3d: bool) -> None:
    cfg = load_config(config_dir)
    loader = OASIS2Loader(cfg.data)
    stats = build_session_cache(
        loader,
        processed_dir=cfg.data.processed_dir,
        num_slices=cfg.data.num_slices,
        img_size=cfg.data.slice_size,
        slice_strategy=cfg.data.slice_strategy,
        vol_depth=cfg.data.vol_depth,
        include_3d=include_3d,
    )
    out = Path("reports/preprocess_stats.json")
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    click.echo(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
