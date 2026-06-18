"""Minimal OpenAI Responses API client for visual property estimation."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

from .models import ObservationEstimate
from .schema import response_schema
from .zed_camera import VisualObservation


SYSTEM_PROMPT = """You estimate latent physical properties for robotic manipulation.
Infer only from visible cues, stereo views, depth statistics, and common-sense priors.
Scores mean: 0 = very low/absent, 1 = very high.
Treat mass and hidden material as uncertain. Never claim visual inference is a measurement.
Give conservative grasp advice: avoid damage and slipping when confidence is low."""


class OpenAIVisionEstimator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.4-mini",
        api_base: str = "https://api.openai.com/v1",
        timeout: int = 90,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY before running inference")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def estimate(self, observation: VisualObservation) -> ObservationEstimate:
        payload = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Estimate the physical properties of the primary object. "
                                f"{observation.depth_summary}"
                            ),
                        },
                        _image_content(observation.left_image, observation.mime_type),
                        _image_content(observation.right_image, observation.mime_type),
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "physical_property_estimate",
                    "strict": True,
                    "schema": response_schema(),
                }
            },
        }
        request = urllib.request.Request(
            f"{self.api_base}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

        return ObservationEstimate.from_dict(json.loads(_output_text(result)))


def _image_content(image: bytes, mime_type: str) -> dict[str, str]:
    encoded = base64.b64encode(image).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type};base64,{encoded}",
        "detail": "high",
    }


def _output_text(response: dict) -> str:
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content["text"]
            if content.get("type") == "refusal":
                raise RuntimeError(f"Model refused the request: {content.get('refusal')}")
    raise RuntimeError("OpenAI response did not contain output text")
