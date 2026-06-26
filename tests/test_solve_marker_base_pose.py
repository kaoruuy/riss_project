from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from camera.aruco_pose import make_transform
from camera.solve_marker_base_pose import marker_base_pose_document


class SolveMarkerBasePoseTests(unittest.TestCase):
    def test_marker_base_pose_document_contains_expected_transform(self) -> None:
        t_base_marker = make_transform(np.eye(3), [0.1, 0.2, 0.3])
        t_cam_marker = make_transform(np.eye(3), [0.0, 0.0, 0.5])

        document = marker_base_pose_document(
            {"marker_id": 1, "T_base_marker": t_base_marker, "T_cam_marker": t_cam_marker},
            image_path=Path("calibration/table_marker_references.png"),
            base_to_camera_path=Path("calibration/base_to_camera.yaml"),
            dictionary_name="DICT_4X4_50",
            marker_length_m=0.045,
            base_frame="base",
            camera_frame="zed_left_camera_optical_frame",
        )

        self.assertEqual(document["marker_id"], 1)
        self.assertEqual(document["calculation"], "T_base_marker = T_base_cam @ T_cam_marker")
        np.testing.assert_allclose(document["T_base_marker"], t_base_marker)
        self.assertAlmostEqual(document["translation_m"]["x"], 0.1)


if __name__ == "__main__":
    unittest.main()
