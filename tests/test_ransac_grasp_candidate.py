from __future__ import annotations

import argparse
import unittest

import numpy as np

from camera.ransac_grasp_candidate import (
    cluster_extent,
    cluster_is_valid,
    fit_plane_ransac,
    largest_grid_cluster,
    object_center,
    point_cloud_from_depth,
    points_within_workspace,
    signed_plane_distances,
    transform_points,
    within_workspace,
)
from camera.zed_pointcloud import Intrinsics


class RansacGraspCandidateTests(unittest.TestCase):
    def test_point_cloud_from_depth_uses_image_convention(self) -> None:
        depth = np.array([[1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        intrinsics = Intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)

        points, pixels = point_cloud_from_depth(
            depth,
            intrinsics,
            min_depth_m=0.1,
            max_depth_m=3.0,
            stride=1,
            max_points=0,
        )

        self.assertEqual(points.shape, (4, 3))
        np.testing.assert_allclose(points[1], [1.0, 0.0, 1.0])
        np.testing.assert_allclose(points[2], [0.0, 1.0, 1.0])
        np.testing.assert_allclose(pixels[3], [1.0, 1.0])

    def test_fit_plane_ransac_recovers_table_plane(self) -> None:
        x, y = np.meshgrid(np.linspace(-0.2, 0.2, 12), np.linspace(-0.2, 0.2, 12))
        table = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.1)))
        outliers = np.array([[0.0, 0.0, 0.16], [0.05, 0.02, 0.15]])
        points = np.vstack((table, outliers))

        plane = fit_plane_ransac(
            points,
            iterations=80,
            threshold_m=0.003,
            rng=np.random.default_rng(1),
        )

        self.assertIsNotNone(plane)
        assert plane is not None
        self.assertGreater(plane["normal"][2], 0.99)
        self.assertLess(np.abs(signed_plane_distances(table, plane["normal"], plane["d"])).max(), 1e-6)

    def test_largest_grid_cluster_selects_largest_component(self) -> None:
        small = np.array([[0.0, 0.0], [0.01, 0.0]])
        large = np.array([[1.0, 1.0], [1.01, 1.0], [1.02, 1.0], [1.03, 1.0]])
        points = np.vstack((small, large))

        indices = largest_grid_cluster(
            points,
            cell_size=0.03,
            min_points=3,
            max_width_m=0.2,
            max_depth_m=0.2,
            max_area_m2=0.04,
        )

        self.assertEqual(set(indices.tolist()), {2, 3, 4, 5})

    def test_largest_grid_cluster_rejects_overly_wide_component(self) -> None:
        wide = np.column_stack((np.linspace(0.0, 0.5, 20), np.zeros(20)))
        compact = np.array([[1.0, 1.0], [1.01, 1.0], [1.0, 1.01], [1.01, 1.01]])
        points = np.vstack((wide, compact))

        indices = largest_grid_cluster(
            points,
            cell_size=0.03,
            min_points=3,
            max_width_m=0.12,
            max_depth_m=0.12,
            max_area_m2=0.02,
        )

        self.assertEqual(set(indices.tolist()), {20, 21, 22, 23})

    def test_cluster_validity_uses_extent_and_density(self) -> None:
        compact = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]])
        wide = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]])

        self.assertTrue(
            cluster_is_valid(
                compact,
                cell_size=0.03,
                min_points=3,
                max_width_m=0.12,
                max_depth_m=0.12,
                max_area_m2=0.02,
                min_density=0.0,
            )
        )
        self.assertFalse(
            cluster_is_valid(
                wide,
                cell_size=0.03,
                min_points=3,
                max_width_m=0.12,
                max_depth_m=0.12,
                max_area_m2=0.02,
                min_density=0.0,
            )
        )
        np.testing.assert_allclose(cluster_extent(compact), [0.01, 0.01])

    def test_object_center_can_use_top_height(self) -> None:
        points = np.array([[0.0, 0.0, 0.1], [0.2, 0.0, 0.2]])

        center = object_center(points, z_mode="top")

        np.testing.assert_allclose(center, [0.1, 0.0, 0.2])

    def test_transform_points_applies_homogeneous_transform(self) -> None:
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]

        points = transform_points(transform, np.array([[0.1, 0.2, 0.3]]))

        np.testing.assert_allclose(points[0], [1.1, 2.2, 3.3])

    def test_within_workspace_checks_bounds(self) -> None:
        args = argparse.Namespace(
            workspace_x=[0.0, 1.0],
            workspace_y=[-1.0, 1.0],
            workspace_z=[0.0, 0.5],
        )

        self.assertTrue(within_workspace(np.array([0.2, 0.0, 0.3]), args))
        self.assertFalse(within_workspace(np.array([1.2, 0.0, 0.3]), args))

    def test_points_within_workspace_filters_outside_candidates(self) -> None:
        args = argparse.Namespace(
            workspace_x=[0.0, 1.0],
            workspace_y=[-1.0, 1.0],
            workspace_z=[0.0, 0.5],
        )
        points = np.array(
            [
                [0.2, 0.0, 0.3],
                [1.2, 0.0, 0.3],
                [0.2, -1.2, 0.3],
                [0.2, 0.0, 0.8],
            ]
        )

        mask = points_within_workspace(points, args)

        np.testing.assert_array_equal(mask, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
