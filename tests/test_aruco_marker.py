from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

from camera.aruco_generator import generate_marker, main


class ArucoMarkerTests(unittest.TestCase):
    def test_generate_marker_requires_non_negative_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            generate_marker(-1)

    def test_generate_marker_requires_positive_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "size-px"):
            generate_marker(0, size_px=0)

    def test_generate_marker_requires_non_negative_border(self) -> None:
        with self.assertRaisesRegex(ValueError, "border-px"):
            generate_marker(0, border_px=-1)

    def test_cli_reports_missing_opencv(self) -> None:
        with patch("camera.aruco_generator.import_cv2", side_effect=ImportError("missing cv2")):
            with redirect_stderr(StringIO()):
                self.assertEqual(main(["0"]), 1)


if __name__ == "__main__":
    unittest.main()
