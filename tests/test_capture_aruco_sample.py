from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from camera.aruco_pose import load_config
from camera.capture_aruco_sample import marker_pose_document, next_sample_index, next_sample_paths


class CaptureArucoSampleTests(unittest.TestCase):
    def test_next_sample_index_uses_pngs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pose_001.png").write_bytes(b"")
            (root / "pose_003.png").write_bytes(b"")
            (root / "pose_999_marker.yaml").write_text("{}", encoding="utf-8")

            self.assertEqual(next_sample_index(root), 4)

    def test_next_sample_paths_share_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = next_sample_paths(Path(tmpdir))

        self.assertEqual(paths["index"], 1)
        self.assertEqual(paths["image"].name, "pose_001.png")
        self.assertEqual(paths["marker"].name, "pose_001_marker.yaml")
        self.assertEqual(paths["base_ee"].name, "pose_001_base_ee.yaml")

    def test_marker_pose_document_contains_fit_inputs(self) -> None:
        image = Path("aruco_markers/aruco_0.png")
        if not image.exists():
            self.skipTest("generated ArUco marker image is not present")
        config = load_config(Path("calibration/aruco_config.yaml"))

        document = marker_pose_document(
            image_path=image,
            config=config,
            dictionary_name="DICT_4X4_50",
            marker_length_m=0.045,
            marker_id=0,
            ignore_distortion=False,
            camera_frame="zed_left_camera_optical_frame",
            sample_index=7,
        )

        self.assertEqual(document["sample_index"], 7)
        self.assertEqual(document["marker_id"], 0)
        self.assertEqual(document["marker_length_m"], 0.045)
        self.assertIn("T_cam_marker", document)
        self.assertIn("T_marker_cam", document)
        self.assertEqual(len(document["T_cam_marker"]), 4)
        yaml.safe_dump(document)


if __name__ == "__main__":
    unittest.main()
