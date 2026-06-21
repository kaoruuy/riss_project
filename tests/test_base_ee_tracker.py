from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from arm.base_ee_tracker import (
    base_ee_document,
    load_pose_sample,
    main,
    parse_pose_data,
    quaternion_to_rotation,
    sample_from_xarm_tcp_pose,
)


class BaseEeTrackerTests(unittest.TestCase):
    def test_quaternion_pose_builds_transform(self) -> None:
        sample = parse_pose_data(
            {
                "position_m": {"x": 0.1, "y": 0.2, "z": 0.3},
                "orientation_quaternion_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        )

        np.testing.assert_allclose(sample["matrix"][:3, :3], np.eye(3))
        np.testing.assert_allclose(sample["matrix"][:3, 3], [0.1, 0.2, 0.3])

    def test_direct_matrix_input_is_accepted(self) -> None:
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]

        sample = parse_pose_data({"T_base_ee": matrix.tolist()})

        np.testing.assert_allclose(sample["matrix"], matrix)

    def test_xarm_tcp_pose_converts_mm_and_degrees(self) -> None:
        sample = sample_from_xarm_tcp_pose([100.0, 200.0, 300.0, 0.0, 0.0, 90.0])

        np.testing.assert_allclose(sample["matrix"][:3, 3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            sample["matrix"][:3, :3],
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            atol=1e-12,
        )

    def test_quaternion_to_rotation_normalizes_input(self) -> None:
        rotation = quaternion_to_rotation(np.array([0.0, 0.0, 0.0, 2.0]))

        np.testing.assert_allclose(rotation, np.eye(3))

    def test_document_contains_transform_and_frames(self) -> None:
        sample = parse_pose_data(
            {
                "position_m": [0.1, 0.2, 0.3],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )

        document = base_ee_document(sample, "base", "tool0")

        self.assertEqual(document["frames"], {"parent": "base", "child": "tool0"})
        np.testing.assert_allclose(document["T_base_ee"][:3], sample["matrix"][:3])

    def test_one_shot_cli_writes_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "pose.yaml"
            output = root / "base_ee.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "position_m": [0.1, 0.2, 0.3],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ),
                encoding="utf-8",
            )

            result = main(
                [
                    "--source",
                    "file",
                    "--pose-file",
                    str(source),
                    "--output",
                    str(output),
                    "--once",
                ]
            )
            saved = load_pose_sample(output)

        self.assertEqual(result, 0)
        np.testing.assert_allclose(saved["matrix"][:3, 3], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
