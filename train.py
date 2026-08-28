from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for item in (str(SRC), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from tbagnet.training_runner import read_yaml, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TBAGNet with a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", required=True, help="Path to the selected dataset root.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--force", action="store_true", help="Overwrite the selected output directory after local audit.")
    args = parser.parse_args()
    config = read_yaml(args.config)
    run_training(config, data_root=args.data_root, output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
