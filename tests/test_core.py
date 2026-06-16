from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from property_estimation.cli import main
from property_estimation.fusion import EstimateFusion
from property_estimation.models import ObservationEstimate, PROPERTY_NAMES
from property_estimation.openai_vlm import _output_text
from property_estimation.schema import response_schema
from property_estimation.zed_camera import DEFAULT_OPEN_TIMEOUT, ZedCamera, _summarize_depth


def estimate(score: float, confidence: float) -> ObservationEstimate:
    return ObservationEstimate.from_dict(
        {
            "object_name": "cup",
            "material_candidates": ["ceramic"],
            "properties": {
                name: {"score": score, "confidence": confidence, "evidence": "visible cue"}
                for name in PROPERTY_NAMES
            },
            "estimated_mass_kg_min": 0.2,
            "estimated_mass_kg_max": 0.4,
            "mass_confidence": confidence,
            "grasp_force": "gentle",
            "manipulation_advice": ["grasp gently"],
            "uncertainty_notes": ["contents unknown"],
        }
    )


class CoreTests(unittest.TestCase):
    def test_camera_open_timeout_defaults_to_30_seconds(self) -> None:
        self.assertEqual(DEFAULT_OPEN_TIMEOUT, 30.0)
        self.assertEqual(ZedCamera().open_timeout, 30.0)

    def test_camera_rejects_non_positive_open_timeout(self) -> None:
        with self.assertRaises(ValueError):
            ZedCamera(open_timeout=0)

    def test_schema_is_strict(self) -> None:
        schema = response_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]["properties"]["required"]), set(PROPERTY_NAMES))

    def test_response_text_extraction(self) -> None:
        payload = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
        self.assertEqual(_output_text(payload), "{}")

    def test_fusion_weights_confident_observation(self) -> None:
        fusion = EstimateFusion()
        fusion.add(estimate(0.0, 0.1))
        fused = fusion.add(estimate(1.0, 0.9))
        self.assertAlmostEqual(fused.properties["softness"].score, 0.9)

    def test_consistent_observations_increase_confidence(self) -> None:
        fusion = EstimateFusion()
        first = fusion.add(estimate(0.5, 0.5))
        second = fusion.add(estimate(0.5, 0.5))
        self.assertGreater(
            second.properties["softness"].confidence,
            first.properties["softness"].confidence,
        )

    def test_depth_summary(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")
        summary = _summarize_depth(np.ones((10, 10), dtype=float))
        self.assertIn("median=1.000", summary)

    def test_model_round_trip(self) -> None:
        original = estimate(0.4, 0.7)
        restored = ObservationEstimate.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(original, restored)

    def test_cli_reports_missing_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main(["--observations", "1"]), 1)


if __name__ == "__main__":
    unittest.main()
