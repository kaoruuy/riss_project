from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import yaml

from camera.diagnose_hand_eye import diagnose_samples, main
from camera.fit_hand_eye import fit_hand_eye
from tests.test_fit_hand_eye import synthetic_samples


class DiagnoseHandEyeTests(unittest.TestCase):
    def test_diagnose_samples_returns_sorted_rows(self) -> None:
        samples, _t_base_cam, _t_ee_marker = synthetic_samples()
        result = fit_hand_eye(samples)
        raw_document = {"samples": [{"id": f"pose_{index:03d}"} for index in range(1, len(samples) + 1)]}

        rows = diagnose_samples(samples, result, raw_document)

        self.assertEqual(len(rows), len(samples))
        self.assertLess(rows[0]["translation_error_m"], 1e-10)
        self.assertGreaterEqual(rows[0]["translation_error_m"], rows[-1]["translation_error_m"])

    def test_cli_prints_report(self) -> None:
        samples, _t_base_cam, _t_ee_marker = synthetic_samples()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "samples": [
                            {
                                "id": f"pose_{index:03d}",
                                "T_base_ee": sample["T_base_ee"].tolist(),
                                "T_cam_marker": sample["T_cam_marker"].tolist(),
                            }
                            for index, sample in enumerate(samples, start=1)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                result = main(["--samples", str(path), "--top", "3"])

        self.assertEqual(result, 0)
        self.assertIn("Selected convention:", stdout.getvalue())
        self.assertIn("Top 3 samples", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
