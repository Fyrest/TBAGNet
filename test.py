from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for item in (str(SRC), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from tbagnet.training_runner import evaluate_checkpoint, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TBAGNet using the YAML file as the sole model configuration source."
    )
    parser.add_argument("--config", required=True, help="YAML configuration used to construct the model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint produced by the training script.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="results/reevaluation")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    # Checkpoint metadata never changes the model structure; it supplies weights only.
    config = read_yaml(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    metrics = evaluate_checkpoint(config, checkpoint_path, args.data_root, args.output_dir, device=args.device)
    print(metrics)


if __name__ == "__main__":
    main()
