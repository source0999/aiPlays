from __future__ import annotations

from typing import Protocol

import numpy as np


class EmulatorBackend(Protocol):
    screen: object
    memory: object

    def tick(self) -> bool: ...
    def button_press(self, input: str) -> None: ...
    def button_release(self, input: str) -> None: ...
    def stop(self) -> None: ...
    def load_state(self, file_like_object: object) -> None: ...


class ScreenBackend(Protocol):
    ndarray: np.ndarray


def framebuffer(backend: EmulatorBackend) -> np.ndarray:
    screen: ScreenBackend = backend.screen  # type: ignore[assignment]
    return np.asarray(screen.ndarray, dtype=np.uint8).copy()
