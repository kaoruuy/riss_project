"""Data models and validation for physical-property estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PROPERTY_NAMES = (
    "softness",
    "rigidity",
    "roughness",
    "slipperiness",
    "deformability",
    "fragility",
)


@dataclass(frozen=True)
class PropertyEstimate:
    score: float
    confidence: float
    evidence: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PropertyEstimate":
        return cls(
            score=_unit_value(data["score"], "score"),
            confidence=_unit_value(data["confidence"], "confidence"),
            evidence=str(data["evidence"]),
        )


@dataclass(frozen=True)
class ObservationEstimate:
    object_name: str
    material_candidates: list[str]
    properties: dict[str, PropertyEstimate]
    estimated_mass_kg_min: float
    estimated_mass_kg_max: float
    mass_confidence: float
    grasp_force: str
    manipulation_advice: list[str]
    uncertainty_notes: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationEstimate":
        properties = {
            name: PropertyEstimate.from_dict(data["properties"][name])
            for name in PROPERTY_NAMES
        }
        mass_min = max(0.0, float(data["estimated_mass_kg_min"]))
        mass_max = max(0.0, float(data["estimated_mass_kg_max"]))
        if mass_min > mass_max:
            mass_min, mass_max = mass_max, mass_min

        return cls(
            object_name=str(data["object_name"]),
            material_candidates=[str(value) for value in data["material_candidates"]],
            properties=properties,
            estimated_mass_kg_min=mass_min,
            estimated_mass_kg_max=mass_max,
            mass_confidence=_unit_value(data["mass_confidence"], "mass_confidence"),
            grasp_force=str(data["grasp_force"]),
            manipulation_advice=[str(value) for value in data["manipulation_advice"]],
            uncertainty_notes=[str(value) for value in data["uncertainty_notes"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unit_value(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {number}")
    return number
