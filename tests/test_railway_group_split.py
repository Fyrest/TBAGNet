from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
for item in (str(ROOT), str(ROOT / "src")):
    if item not in sys.path:
        sys.path.insert(0, item)

from data.railway_split import generate_grouped_split, validate_grouped_manifest


def _write_signal(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"signal": values[:, None].astype(np.float64)})


class RailwayGroupedSplitTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        data_root = root / "railway"
        mapping_path = root / "mapping.json"
        mapping_path.write_text(json.dumps({"0": "Class_A"}), encoding="utf-8")
        base = np.arange(2048, dtype=np.float32)
        _write_signal(data_root / "Class_A/a.mat", base)
        _write_signal(data_root / "Class_A/a_copy.mat", base)
        _write_signal(
            data_root / "Class_A/shared_head.mat",
            np.concatenate([base[:1024], base[1024:] + 100]),
        )
        for index in range(1, 5):
            _write_signal(data_root / f"Class_A/unique_{index}.mat", base + index)
        return data_root, mapping_path

    def test_identical_sources_are_kept_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, mapping_path = self._fixture(root)
            output_dir = root / "generated"
            report = generate_grouped_split(
                data_root=data_root,
                class_mapping_path=mapping_path,
                output_dir=output_dir,
                seed=42,
                window_length=1024,
                stride=512,
            )
            manifest = pd.read_csv(output_dir / "manifest.csv")
            duplicate_rows = manifest[
                manifest["relative_path"].isin(["Class_A/a.mat", "Class_A/a_copy.mat"])
            ]
            self.assertEqual(duplicate_rows["source_signal_sha256"].nunique(), 1)
            self.assertEqual(duplicate_rows["split"].nunique(), 1)
            self.assertEqual(
                report["integrity"]["source_hash_intersections"],
                {"train_val": 0, "train_test": 0, "val_test": 0},
            )
            self.assertEqual(
                report["integrity"]["window_hash_intersections"],
                {"train_val": 0, "train_test": 0, "val_test": 0},
            )

    def test_validator_rejects_cross_split_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, mapping_path = self._fixture(root)
            output_dir = root / "generated"
            generate_grouped_split(
                data_root=data_root,
                class_mapping_path=mapping_path,
                output_dir=output_dir,
                seed=42,
            )
            manifest = pd.read_csv(output_dir / "manifest.csv")
            digest = manifest.loc[
                manifest["relative_path"] == "Class_A/a.mat", "source_signal_sha256"
            ].iloc[0]
            indices = manifest.index[manifest["source_signal_sha256"] == digest].tolist()
            manifest.loc[indices[0], "split"] = "train"
            manifest.loc[indices[-1], "split"] = "test"
            with self.assertRaisesRegex(RuntimeError, "Source-signal hash isolation failed"):
                validate_grouped_manifest(manifest, data_root=data_root)

    def test_validator_rejects_cross_split_exact_window_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, mapping_path = self._fixture(root)
            output_dir = root / "generated"
            generate_grouped_split(
                data_root=data_root,
                class_mapping_path=mapping_path,
                output_dir=output_dir,
                seed=42,
            )
            manifest = pd.read_csv(output_dir / "manifest.csv")
            base_digest = manifest.loc[
                manifest["relative_path"] == "Class_A/a.mat", "source_signal_sha256"
            ].iloc[0]
            shared_digest = manifest.loc[
                manifest["relative_path"] == "Class_A/shared_head.mat", "source_signal_sha256"
            ].iloc[0]
            manifest.loc[manifest["source_signal_sha256"] == base_digest, "split"] = "train"
            manifest.loc[manifest["source_signal_sha256"] == shared_digest, "split"] = "test"
            with self.assertRaisesRegex(RuntimeError, "Exact-window hash isolation failed"):
                validate_grouped_manifest(manifest, data_root=data_root)


if __name__ == "__main__":
    unittest.main()
