from __future__ import annotations

from typing import Any


class CoordinateDiscovery:
    name = "new_coordinate"

    def __init__(self) -> None:
        self.visited: set[tuple[int, int, int]] = set()

    def reset(self, info: dict[str, Any]) -> None:
        self.visited.clear()
        self.value(info)

    def value(self, info: dict[str, Any]) -> float:
        ram = info.get("ram", {})
        if not {"map_id", "x", "y"}.issubset(ram):
            return 0.0
        location = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
        if location in self.visited:
            return 0.0
        self.visited.add(location)
        return 1.0


class StepPenalty:
    name = "step"

    def reset(self, info: dict[str, Any]) -> None:
        del info

    def value(self, info: dict[str, Any]) -> float:
        del info
        return 1.0


class StuckPenalty:
    name = "stuck"

    def __init__(self, threshold: int = 12) -> None:
        self.threshold = threshold
        self.previous: tuple[int, int, int] | None = None
        self.count = 0

    def reset(self, info: dict[str, Any]) -> None:
        self.previous = None
        self.count = 0
        self.value(info)

    def value(self, info: dict[str, Any]) -> float:
        ram = info.get("ram", {})
        if not {"map_id", "x", "y"}.issubset(ram):
            return 0.0
        place = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
        self.count = self.count + 1 if place == self.previous else 0
        self.previous = place
        return 1.0 if self.count >= self.threshold else 0.0
