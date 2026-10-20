"""search.py — CLI: bounded hyperparameter search for a champion family.

python scripts/search.py
"""

from __future__ import annotations

import click

from neurosight.config import load_config
from neurosight.data.loader import OASIS2Loader
from neurosight.training.search import run_search
from neurosight.utils.logging import get_logger
from neurosight.utils.seed import set_seed

logger = get_logger(__name__)


@click.command()
@click.option("--config-dir", default="configs", show_default=True)
@click.option("--model", default="3_cnn_lstm", show_default=True)
def main(config_dir: str, model: str) -> None:
    cfg = load_config(config_dir)
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
    loader = OASIS2Loader(cfg.data)
    results = run_search(cfg, loader, model)
    logger.info("winner: %s", results[0])


if __name__ == "__main__":
    main()
