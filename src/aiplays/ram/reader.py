from __future__ import annotations

from typing import Protocol


class MemoryReader(Protocol):
    def read(self, memory: object) -> dict[str, int | list[int]]: ...
