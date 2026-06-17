from __future__ import annotations

import unittest

import numpy as np

from camera.aruco_pose import camera_parameters, invert_transform, make_transform, marker_object_points


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


if __name__ == "__main__":
    unittest.main()
