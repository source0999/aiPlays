from __future__ import annotations

from typing import Any


class DeltaField:
    """Only rewards an increase, preventing a reward every step for current progress."""

    def __init__(self, name: str, field: str) -> None:
        self.name, self.field, self.previous = name, field, 0.0

    def reset(self, info: dict[str, Any]) -> None:
        self.previous = float(info.get("ram", {}).get(self.field, 0.0))

    def value(self, info: dict[str, Any]) -> float:
        current = float(info.get("ram", {}).get(self.field, self.previous))
        delta = max(0.0, current - self.previous)
        self.previous = current
        return delta
