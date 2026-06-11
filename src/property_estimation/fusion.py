"""Confidence-weighted fusion of repeated visual estimates."""

from __future__ import annotations

import math
import statistics
from collections import Counter

from .models import ObservationEstimate, PROPERTY_NAMES, PropertyEstimate


class EstimateFusion:
    def __init__(self) -> None:
        self._observations: list[ObservationEstimate] = []

    def add(self, estimate: ObservationEstimate) -> ObservationEstimate:
        self._observations.append(estimate)
        return self.current()

    def current(self) -> ObservationEstimate:
        if not self._observations:
            raise RuntimeError("No estimates have been added")

        properties = {}
        for name in PROPERTY_NAMES:
            values = [item.properties[name] for item in self._observations]
            weights = [max(value.confidence, 0.05) for value in values]
            properties[name] = PropertyEstimate(
                score=_weighted_mean([value.score for value in values], weights),
                confidence=_combined_confidence(values),
                evidence=values[-1].evidence,
            )

        mass_weights = [max(item.mass_confidence, 0.05) for item in self._observations]
        latest = self._observations[-1]
        return ObservationEstimate(
            object_name=_most_common(item.object_name for item in self._observations),
            material_candidates=_top_materials(self._observations),
            properties=properties,
            estimated_mass_kg_min=_weighted_mean(
                [item.estimated_mass_kg_min for item in self._observations], mass_weights
            ),
            estimated_mass_kg_max=_weighted_mean(
                [item.estimated_mass_kg_max for item in self._observations], mass_weights
            ),
            mass_confidence=_accumulated_confidence(
                [item.mass_confidence for item in self._observations]
            ),
            grasp_force=_most_common(item.grasp_force for item in self._observations),
            manipulation_advice=latest.manipulation_advice,
            uncertainty_notes=latest.uncertainty_notes,
        )


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _combined_confidence(values: list[PropertyEstimate]) -> float:
    accumulated = _accumulated_confidence([value.confidence for value in values])
    disagreement = statistics.pstdev(value.score for value in values)
    return max(0.0, min(0.99, accumulated * (1.0 - disagreement)))


def _accumulated_confidence(confidences: list[float]) -> float:
    return min(0.99, 1.0 - math.prod(1.0 - confidence for confidence in confidences))


def _most_common(values: object) -> str:
    return Counter(values).most_common(1)[0][0]


def _top_materials(observations: list[ObservationEstimate]) -> list[str]:
    counts = Counter(
        material for observation in observations for material in observation.material_candidates
    )
    return [material for material, _count in counts.most_common(5)]
