from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces


class FakeScreen:
    def __init__(self) -> None:
        self.ndarray = np.zeros((144, 160, 4), dtype=np.uint8)
        self.ndarray[..., 3] = 255


class FakeBackend:
    def __init__(self) -> None:
        self.screen = FakeScreen()
        self.memory = bytearray(0x10000)
        self.events: list[tuple[str, str]] = []
        self.ticks = 0
        self.stopped = False
        self.running = True

    def tick(self) -> bool:
        self.ticks += 1
        self.screen.ndarray[..., 0] = self.ticks % 255
        return self.running

    def button_press(self, input: str) -> None:
        self.events.append(("press", input))

    def button_release(self, input: str) -> None:
        self.events.append(("release", input))

    def load_state(self, file_like_object: object) -> None:
        del file_like_object

    def stop(self) -> None:
        self.stopped = True


class FakeTrainingEnv(Env[np.ndarray, int]):
    metadata = {"render_modes": [None]}

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 255, shape=(4,), dtype=np.uint8)
        self.step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        self.step_count = 0
        return np.zeros(4, dtype=np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.step_count += 1
        return (
            np.full(4, action, dtype=np.uint8),
            float(action == 1),
            False,
            self.step_count >= 8,
            {},
        )
