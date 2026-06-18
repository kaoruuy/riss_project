"""Strict JSON schema used for model output."""

from __future__ import annotations

from .models import PROPERTY_NAMES


def response_schema() -> dict:
    property_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
        },
        "required": ["score", "confidence", "evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "object_name": {"type": "string"},
            "material_candidates": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
            "properties": {
                "type": "object",
                "properties": {name: property_schema for name in PROPERTY_NAMES},
                "required": list(PROPERTY_NAMES),
                "additionalProperties": False,
            },
            "estimated_mass_kg_min": {"type": "number", "minimum": 0},
            "estimated_mass_kg_max": {"type": "number", "minimum": 0},
            "mass_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "grasp_force": {
                "type": "string",
                "enum": ["very_gentle", "gentle", "moderate", "firm", "unknown"],
            },
            "manipulation_advice": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
            "uncertainty_notes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
        },
        "required": [
            "object_name",
            "material_candidates",
            "properties",
            "estimated_mass_kg_min",
            "estimated_mass_kg_max",
            "mass_confidence",
            "grasp_force",
            "manipulation_advice",
            "uncertainty_notes",
        ],
        "additionalProperties": False,
    }
