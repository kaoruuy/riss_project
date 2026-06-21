from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from camera.zed_pointcloud import (
    ColoredPointCloud,
    Intrinsics,
    point_cloud_from_rgbd,
    rgb_from_image,
    write_point_cloud,
)


class ZedPointCloudTests(unittest.TestCase):
    def test_rgb_from_bgra_image(self) -> None:
        image = np.array([[[10, 20, 30, 255]]], dtype=np.uint8)
        rgb = rgb_from_image(image, "bgra")
        np.testing.assert_array_equal(rgb, [[[30, 20, 10]]])

    def test_point_cloud_projection(self) -> None:
        depth = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
        rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        cloud = point_cloud_from_rgbd(
            depth,
            rgb,
            Intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            min_depth_m=0.1,
            max_depth_m=5.0,
        )
        np.testing.assert_allclose(cloud.points, [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0], [4.0, 4.0, 4.0]])
        np.testing.assert_array_equal(cloud.colors, [[255, 0, 0], [0, 255, 0], [255, 255, 255]])

    def test_write_ascii_ply(self) -> None:
        cloud = ColoredPointCloud(
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            colors=np.array([[4, 5, 6]], dtype=np.uint8),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            write_point_cloud(path, cloud)
            text = path.read_text(encoding="utf-8")
        self.assertIn("element vertex 1", text)
        self.assertIn("1.000000 2.000000 3.000000 4 5 6", text)


if __name__ == "__main__":
    unittest.main()
