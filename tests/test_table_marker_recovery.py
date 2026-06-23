from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from camera.aruco_pose import make_transform
from camera.table_marker_recovery import (
    compute_table_references,
    load_table_references,
    recover_base_to_camera,
    recovered_base_to_camera_document,
    table_references_document,
)


class TableMarkerRecoveryTests(unittest.TestCase):
    def test_recover_base_to_camera_from_references(self) -> None:
        t_base_cam = make_transform(np.eye(3), [0.8, 0.1, 0.5])
        marker_poses = {
            1: make_transform(np.eye(3), [0.2, 0.0, 1.0]),
            2: make_transform(np.eye(3), [0.0, 0.1, 1.1]),
        }
        references = compute_table_references(t_base_cam, marker_poses)

        result = recover_base_to_camera(references, marker_poses, min_markers=2)

        np.testing.assert_allclose(result["T_base_cam"], t_base_cam, atol=1e-12)
        self.assertEqual(result["used_marker_ids"], [1, 2])
        self.assertLess(result["residuals"]["translation_m"]["max"], 1e-12)

    def test_recover_base_to_camera_uses_visible_subset(self) -> None:
        t_base_cam = make_transform(np.eye(3), [0.8, 0.1, 0.5])
        marker_poses = {1: make_transform(np.eye(3), [0.2, 0.0, 1.0])}
        references = compute_table_references(t_base_cam, marker_poses)
        references[2] = make_transform(np.eye(3), [1.0, 2.0, 3.0])

        result = recover_base_to_camera(references, marker_poses, min_markers=1)

        np.testing.assert_allclose(result["T_base_cam"], t_base_cam, atol=1e-12)
        self.assertEqual(result["used_marker_ids"], [1])

    def test_table_references_document_round_trips(self) -> None:
        t_base_cam = make_transform(np.eye(3), [0.8, 0.1, 0.5])
        marker_poses = {1: make_transform(np.eye(3), [0.2, 0.0, 1.0])}
        references = compute_table_references(t_base_cam, marker_poses)
        document = table_references_document(
            references,
            marker_poses,
            image_path=Path("calibration/table_marker_references.png"),
            dictionary_name="DICT_4X4_50",
            marker_length_m=0.045,
            base_to_camera_path=Path("calibration/base_to_camera.yaml"),
            base_frame="base",
            camera_frame="zed_left_camera_optical_frame",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table_marker_references.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            loaded = load_table_references(path)

        np.testing.assert_allclose(loaded[1], references[1])

    def test_recovered_document_contains_transform_matrix(self) -> None:
        t_base_cam = make_transform(np.eye(3), [0.8, 0.1, 0.5])
        result = {
            "T_base_cam": t_base_cam,
            "used_marker_ids": [1],
            "per_marker_estimates": {1: t_base_cam},
            "residuals": {"translation_m": {}, "rotation_deg": {}, "per_marker": []},
        }

        document = recovered_base_to_camera_document(
            result,
            image_path=Path("calibration/table_marker_recovery.png"),
            references_path=Path("calibration/table_marker_references.yaml"),
            base_frame="base",
            camera_frame="zed_left_camera_optical_frame",
        )

        np.testing.assert_allclose(document["transform_matrix"], t_base_cam)
        self.assertEqual(document["used_marker_ids"], [1])


if __name__ == "__main__":
    unittest.main()
