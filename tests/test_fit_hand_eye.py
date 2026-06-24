from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from camera.fit_hand_eye import fit_hand_eye, invert_transform, load_samples, main, make_transform


ZED_SETTINGS = {
    "sdk_version": "5.3.1",
    "resolution": "HD720",
    "fps": 30,
    "depth_mode": "NEURAL",
    "coordinate_units": "METER",
    "coordinate_system": "IMAGE",
    "left_view": "LEFT",
    "left_image_rectification": "rectified",
    "intrinsics_file": "calibration/zed_intrinsics.yaml",
}


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def synthetic_samples() -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    t_base_cam = make_transform(rot_z(0.2) @ rot_y(0.3), [0.3, 0.1, 0.2])
    t_ee_marker = make_transform(rot_x(0.4) @ rot_z(-0.1), [0.02, -0.03, 0.04])
    samples = []
    for index in range(8):
        rotation = rot_z(0.2 * index) @ rot_y(0.1 * index) @ rot_x(0.05 * index)
        translation = [
            0.25 + 0.02 * index,
            -0.1 + 0.03 * np.sin(index),
            0.2 + 0.01 * np.cos(index),
        ]
        t_base_ee = make_transform(rotation, translation)
        t_cam_marker = invert_transform(t_base_cam) @ t_base_ee @ t_ee_marker
        samples.append({"T_base_ee": t_base_ee, "T_cam_marker": t_cam_marker})
    return samples, t_base_cam, t_ee_marker


class FitHandEyeTests(unittest.TestCase):
    def test_fit_hand_eye_recovers_synthetic_transforms(self) -> None:
        samples, t_base_cam, t_ee_marker = synthetic_samples()

        result = fit_hand_eye(samples)

        np.testing.assert_allclose(result["T_base_cam"], t_base_cam, atol=1e-10)
        np.testing.assert_allclose(result["T_ee_marker"], t_ee_marker, atol=1e-10)
        self.assertLess(result["residuals"]["translation_m"]["max"], 1e-10)
        self.assertEqual(len(result["tested_conventions"]), 4)

    def test_load_samples_from_yaml(self) -> None:
        samples, _t_base_cam, _t_ee_marker = synthetic_samples()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "samples": [
                            {
                                "T_base_ee": sample["T_base_ee"].tolist(),
                                "T_cam_marker": sample["T_cam_marker"].tolist(),
                                "zed_settings": ZED_SETTINGS,
                            }
                            for sample in samples[:3]
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_samples(path)

        self.assertEqual(len(loaded), 3)
        np.testing.assert_allclose(loaded[0]["T_base_ee"], samples[0]["T_base_ee"])
        self.assertEqual(loaded[0]["zed_settings"]["depth_mode"], "NEURAL")

    def test_cli_writes_result_files(self) -> None:
        samples, _t_base_cam, _t_ee_marker = synthetic_samples()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples_path = root / "samples.yaml"
            base_cam_path = root / "base_to_camera.yaml"
            ee_marker_path = root / "ee_marker_estimated.yaml"
            samples_path.write_text(
                yaml.safe_dump(
                    {
                        "samples": [
                            {
                                "T_base_ee": sample["T_base_ee"].tolist(),
                                "T_cam_marker": sample["T_cam_marker"].tolist(),
                                "zed_settings": ZED_SETTINGS,
                            }
                            for sample in samples
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = main(
                [
                    "--samples",
                    str(samples_path),
                    "--base-to-camera-output",
                    str(base_cam_path),
                    "--ee-marker-output",
                    str(ee_marker_path),
                ]
            )
            base_cam = yaml.safe_load(base_cam_path.read_text(encoding="utf-8"))
            ee_marker = yaml.safe_load(ee_marker_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertTrue(base_cam["calibrated"])
        self.assertTrue(ee_marker["calibrated"])
        self.assertEqual(base_cam["sample_count"], len(samples))
        self.assertEqual(base_cam["calibration"]["sample_count"], len(samples))
        self.assertEqual(base_cam["zed"]["depth_mode"], "NEURAL")
        self.assertEqual(base_cam["zed"]["coordinate_system"], "IMAGE")
        self.assertIn("translation_rms_m", base_cam["results"])
        self.assertIn("rotation_rms_deg", base_cam["results"])
        self.assertIn("transform_matrix", base_cam)
        self.assertIn("T_ee_marker", ee_marker)
        self.assertEqual(ee_marker["zed"]["depth_mode"], "NEURAL")
        self.assertIn("selected_convention", base_cam)
        self.assertIn("tested_conventions", ee_marker)


if __name__ == "__main__":
    unittest.main()
