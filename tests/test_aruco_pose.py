from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from camera.aruco_pose import (
    camera_parameters,
    invert_transform,
    load_transform_matrix,
    make_transform,
    marker_object_points,
    save_transforms,
)


class ArucoPoseTests(unittest.TestCase):
    def test_camera_parameters_builds_matrix(self) -> None:
        params = camera_parameters(
            {
                "camera": {
                    "fx": 772.11,
                    "fy": 771.955,
                    "cx": 626.175,
                    "cy": 377.329,
                    "distortion_vector": [-0.0302337, 0.00629367, 0.0175478, -0.0115979],
                }
            }
        )
        self.assertEqual(params["camera_matrix"].shape, (3, 3))
        self.assertEqual(params["distortion"].shape, (4, 1))
        self.assertAlmostEqual(params["camera_matrix"][0, 0], 772.11)

    def test_marker_object_points_are_centered(self) -> None:
        points = marker_object_points(0.04)
        np.testing.assert_allclose(points.mean(axis=0), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(points[0], [-0.02, 0.02, 0.0])

    def test_make_transform_places_translation(self) -> None:
        transform = make_transform(np.eye(3), np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])

    def test_invert_transform(self) -> None:
        transform = make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
        inverse = invert_transform(transform)
        np.testing.assert_allclose(inverse[:3, 3], [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(transform @ inverse, np.eye(4))

    def test_load_transform_matrix_from_keyed_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ee_marker.yaml"
            matrix = np.eye(4)
            matrix[:3, 3] = [0.1, 0.2, 0.3]
            path.write_text(yaml.safe_dump({"T_ee_marker": matrix.tolist()}), encoding="utf-8")

            loaded = load_transform_matrix(path, preferred_key="T_ee_marker")

        np.testing.assert_allclose(loaded, matrix)

    def test_save_transforms_composes_base_to_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ee_marker_path = root / "ee_marker.yaml"
            base_ee_path = root / "base_ee.yaml"
            output_path = root / "transforms.yaml"

            t_cam_marker = make_transform(np.eye(3), [0.0, 0.0, 1.0])
            t_marker_cam = invert_transform(t_cam_marker)
            t_ee_marker = make_transform(np.eye(3), [0.0, 2.0, 0.0])
            t_base_ee = make_transform(np.eye(3), [3.0, 0.0, 0.0])
            ee_marker_path.write_text(
                yaml.safe_dump({"T_ee_marker": t_ee_marker.tolist()}),
                encoding="utf-8",
            )
            base_ee_path.write_text(
                yaml.safe_dump({"T_base_ee": t_base_ee.tolist()}),
                encoding="utf-8",
            )

            save_transforms(
                output_path,
                {"marker_to_camera": t_cam_marker, "camera_to_marker": t_marker_cam},
                ee_marker_path,
                base_ee_path,
                {
                    "marker_id": 0,
                    "marker_length_m": 0.05,
                    "camera_frame": "zed_left_camera_optical_frame",
                },
            )
            saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))

        np.testing.assert_allclose(saved["transforms"]["T_cam_marker"], t_cam_marker)
        np.testing.assert_allclose(saved["transforms"]["T_ee_marker"], t_ee_marker)
        np.testing.assert_allclose(saved["transforms"]["T_base_ee"], t_base_ee)
        np.testing.assert_allclose(
            saved["transforms"]["T_base_cam"],
            t_base_ee @ t_ee_marker @ t_marker_cam,
        )


if __name__ == "__main__":
    unittest.main()
