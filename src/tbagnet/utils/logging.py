from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str | Path) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_dir / "train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
