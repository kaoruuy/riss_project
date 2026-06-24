from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np

from camera.aruco_live_viewer import print_transform
from camera.zed_config import camera_parameters_from_config


class ArucoLiveViewerTests(unittest.TestCase):
    def test_live_camera_parameters_reads_aruco_config_shape(self) -> None:
        params = camera_parameters_from_config(
            {
                "camera": {
                    "fx": 772.11,
                    "fy": 771.955,
                    "cx": 626.175,
                    "cy": 377.329,
                    "distortion_vector": [-0.0302337, 0.00629367, 0.0175478, -0.0115979],
                }
            },
            resolution="HD720",
            eye="left",
            ignore_distortion=False,
        )

        self.assertEqual(params["camera_matrix"].shape, (3, 3))
        self.assertEqual(params["distortion"].shape, (4, 1))
        self.assertAlmostEqual(params["camera_matrix"][0, 0], 772.11)

    def test_live_camera_parameters_reads_zed_intrinsics_shape(self) -> None:
        params = camera_parameters_from_config(
            {
                "resolutions": {
                    "HD720": {
                        "left": {
                            "fx": 772.11,
                            "fy": 771.955,
                            "cx": 626.175,
                            "cy": 377.329,
                            "distortion_vector": [-0.0302337, 0.00629367, 0.0175478, -0.0115979],
                        }
                    }
                }
            },
            resolution="HD720",
            eye="left",
            ignore_distortion=True,
        )

        self.assertEqual(params["camera_matrix"].shape, (3, 3))
        np.testing.assert_allclose(params["distortion"], np.zeros((4, 1)))

    def test_print_transform_outputs_marker_payload(self) -> None:
        matrix = np.eye(4)
        matrix[:3, 3] = [0.1, 0.2, 0.3]
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            print_transform(
                marker_id=0,
                dictionary_name="DICT_4X4_50",
                marker_length_m=0.05,
                t_cam_marker=matrix,
            )

        self.assertIn('"marker_id": 0', output.getvalue())
        self.assertIn('"T_cam_marker"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
