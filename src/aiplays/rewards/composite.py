from __future__ import annotations

from typing import Any

from aiplays.rewards.base import RewardComponent
from aiplays.rewards.exploration import CoordinateDiscovery, StepPenalty, StuckPenalty
from aiplays.rewards.pokemon_red import DeltaField


class CompositeReward:
    def __init__(self, weights: dict[str, float]):
        components: list[RewardComponent] = [CoordinateDiscovery(), StepPenalty(), StuckPenalty()]
        components.extend(
            [
                DeltaField("new_map", "map_id"),
                DeltaField("badge", "badges"),
                DeltaField("party_level", "party_level_total"),
            ]
        )
        self.components = {component.name: component for component in components}
        self.weights = weights
        self.last_breakdown: dict[str, float] = {}

    def reset(self, info: dict[str, Any]) -> None:
        for component in self.components.values():
            component.reset(info)
        self.last_breakdown = {}

    def value(self, info: dict[str, Any]) -> float:
        self.last_breakdown = {
            name: component.value(info) * self.weights.get(name, 0.0)
            for name, component in self.components.items()
        }
        return float(sum(self.last_breakdown.values()))
