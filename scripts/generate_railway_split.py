from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for item in (str(ROOT), str(ROOT / "src")):
    if item not in sys.path:
        sys.path.insert(0, item)

from data.railway_split import DEFAULT_RATIOS, generate_grouped_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Railway-DAS split after grouping byte-equivalent model-input signals."
    )
    parser.add_argument("--data-root", required=True, help="Railway-DAS root containing the class directories.")
    parser.add_argument("--output-dir", required=True, help="New, empty directory for generated split files.")
    parser.add_argument(
        "--class-mapping",
        default=str(ROOT / "data/splits/railway_das/class_mapping.json"),
        help="JSON class-ID mapping.",
    )
    parser.add_argument(
        "--window-catalog",
        default=None,
        help="Optional manifest whose start/end selections are preserved; its existing split column is ignored.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=DEFAULT_RATIOS,
    )
    parser.add_argument("--data-key", default="signal")
    args = parser.parse_args()
    report = generate_grouped_split(
        data_root=args.data_root,
        class_mapping_path=args.class_mapping,
        output_dir=args.output_dir,
        window_catalog=args.window_catalog,
        seed=args.seed,
        window_length=args.window_length,
        stride=args.stride,
        ratios=tuple(args.ratios),
        data_key=args.data_key,
        protected_output_dir=ROOT / "data/splits/railway_das",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
