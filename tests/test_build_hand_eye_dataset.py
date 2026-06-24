from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from camera.build_hand_eye_dataset import build_dataset, find_sample_pairs, main


def transform(tx: float = 0.0) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[0, 3] = tx
    return matrix.tolist()


def zed_settings(depth_mode: str = "NEURAL") -> dict[str, object]:
    return {
        "resolution": "HD720",
        "fps": 30,
        "depth_mode": depth_mode,
        "coordinate_units": "METER",
        "coordinate_system": "IMAGE",
        "left_view": "LEFT",
        "left_image_rectification": "rectified",
        "intrinsics_file": "calibration/zed_intrinsics.yaml",
    }


class BuildHandEyeDatasetTests(unittest.TestCase):
    def test_find_sample_pairs_matches_shared_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pose_001_base_ee.yaml").write_text("{}", encoding="utf-8")
            (root / "pose_001_marker.yaml").write_text("{}", encoding="utf-8")
            (root / "pose_002_base_ee.yaml").write_text("{}", encoding="utf-8")

            pairs = find_sample_pairs(root)

        self.assertEqual([stem for stem, _files in pairs], ["pose_001"])

    def test_build_dataset_extracts_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pose_001_base_ee.yaml").write_text(
                yaml.safe_dump({"T_base_ee": transform(0.1), "zed_settings": zed_settings()}),
                encoding="utf-8",
            )
            (root / "pose_001_marker.yaml").write_text(
                yaml.safe_dump(
                    {
                        "image": "pose_001.png",
                        "marker_id": 0,
                        "zed_settings": zed_settings(),
                        "T_cam_marker": transform(0.2),
                    }
                ),
                encoding="utf-8",
            )

            samples = build_dataset(root)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["id"], "pose_001")
        self.assertEqual(samples[0]["image"], "pose_001.png")
        self.assertEqual(samples[0]["marker_id"], 0)
        self.assertEqual(samples[0]["zed_settings"]["depth_mode"], "NEURAL")
        np.testing.assert_allclose(samples[0]["T_base_ee"], transform(0.1))
        np.testing.assert_allclose(samples[0]["T_cam_marker"], transform(0.2))

    def test_build_dataset_rejects_non_hand_marker_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pose_001_base_ee.yaml").write_text(
                yaml.safe_dump({"T_base_ee": transform(0.1)}),
                encoding="utf-8",
            )
            (root / "pose_001_marker.yaml").write_text(
                yaml.safe_dump({"marker_id": 2, "T_cam_marker": transform(0.2)}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected hand marker 0"):
                build_dataset(root)

    def test_build_dataset_rejects_zed_settings_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pose_001_base_ee.yaml").write_text(
                yaml.safe_dump({"T_base_ee": transform(0.1), "zed_settings": zed_settings()}),
                encoding="utf-8",
            )
            (root / "pose_001_marker.yaml").write_text(
                yaml.safe_dump(
                    {
                        "marker_id": 0,
                        "T_cam_marker": transform(0.2),
                        "zed_settings": zed_settings("QUALITY"),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "different zed_settings"):
                build_dataset(root)

    def test_cli_writes_hand_eye_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "hand_eye_samples.yaml"
            (root / "pose_001_base_ee.yaml").write_text(
                yaml.safe_dump({"T_base_ee": transform(0.1)}),
                encoding="utf-8",
            )
            (root / "pose_001_marker.yaml").write_text(
                yaml.safe_dump({"marker_id": 0, "T_cam_marker": transform(0.2)}),
                encoding="utf-8",
            )

            result = main(["--input-dir", str(root), "--output", str(output)])
            document = yaml.safe_load(output.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(document["sample_count"], 1)
        self.assertEqual(document["samples"][0]["id"], "pose_001")
        self.assertIsNone(document["zed_settings"])


if __name__ == "__main__":
    unittest.main()
