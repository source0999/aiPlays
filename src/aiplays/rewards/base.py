from __future__ import annotations

from typing import Any, Protocol


class RewardComponent(Protocol):
    name: str

    def reset(self, info: dict[str, Any]) -> None: ...

    def value(self, info: dict[str, Any]) -> float: ...
