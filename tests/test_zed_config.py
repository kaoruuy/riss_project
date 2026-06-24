from __future__ import annotations

import argparse
import unittest

from camera.zed_config import (
    DEFAULT_DEPTH_MODE,
    ZedRuntimeConfig,
    add_zed_runtime_args,
    config_from_args,
    left_view_rectification,
)


class ZedConfigTests(unittest.TestCase):
    def test_default_metadata_captures_runtime_contract(self) -> None:
        metadata = ZedRuntimeConfig().metadata()

        self.assertEqual(metadata["resolution"], "HD720")
        self.assertEqual(metadata["depth_mode"], DEFAULT_DEPTH_MODE)
        self.assertEqual(metadata["coordinate_units"], "METER")
        self.assertEqual(metadata["coordinate_system"], "IMAGE")
        self.assertEqual(metadata["left_view"], "LEFT")
        self.assertEqual(metadata["left_image_rectification"], "rectified")
        self.assertEqual(metadata["intrinsics_file"], "calibration/zed_intrinsics.yaml")

    def test_metadata_with_sdk_includes_sdk_version(self) -> None:
        class Camera:
            @staticmethod
            def get_sdk_version() -> str:
                return "5.3.1"

        class Sl:
            pass

        Sl.Camera = Camera

        metadata = ZedRuntimeConfig().metadata_with_sdk(Sl)

        self.assertEqual(metadata["sdk_version"], "5.3.1")

    def test_config_from_args_uses_shared_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        add_zed_runtime_args(parser)
        args = parser.parse_args([])

        config = config_from_args(args)

        self.assertEqual(config.depth_mode, "NEURAL")
        self.assertEqual(config.resolution, "HD720")

    def test_left_view_rectification_marks_raw_views(self) -> None:
        self.assertEqual(left_view_rectification("LEFT"), "rectified")
        self.assertEqual(left_view_rectification("LEFT_UNRECTIFIED"), "raw_unrectified")


if __name__ == "__main__":
    unittest.main()
