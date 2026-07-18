from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from gymnasium import spaces

from aiplays.config import ObservationConfig


class ObservationProcessor:
    """Turns PyBoy's mutable RGBA buffer into stable uint8 Gym observations."""

    def __init__(self, config: ObservationConfig):
        self.config = config
        channels = 1 if config.grayscale else 3
        self.pixel_space = spaces.Box(
            0,
            255,
            shape=(channels * config.frame_stack, config.height, config.width),
            dtype=np.uint8,
        )
        self.frames: deque[np.ndarray] = deque(maxlen=config.frame_stack)

    def reset(self, rgba: np.ndarray) -> np.ndarray:
        frame = self._process(rgba)
        self.frames.clear()
        self.frames.extend(frame.copy() for _ in range(self.config.frame_stack))
        return self.value()

    def append(self, rgba: np.ndarray) -> np.ndarray:
        self.frames.append(self._process(rgba))
        return self.value()

    def value(self) -> np.ndarray:
        if len(self.frames) != self.config.frame_stack:
            raise RuntimeError("frame stack has not been initialized")
        observation = np.concatenate(tuple(self.frames), axis=0).astype(np.uint8, copy=False)
        if not self.pixel_space.contains(observation):
            raise RuntimeError("pixel observation does not match its declared space")
        return observation

    def _process(self, rgba: np.ndarray) -> np.ndarray:
        image = np.asarray(rgba, dtype=np.uint8).copy()  # PyBoy's framebuffer is reused every tick.
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Expected an RGBA/RGB framebuffer, got shape {image.shape}")
        rgb = image[..., :3]
        if self.config.grayscale:
            resized = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                (self.config.width, self.config.height),
                interpolation=cv2.INTER_AREA,
            )
            return resized[None, ...]
        resized = cv2.resize(
            rgb, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA
        )
        return np.moveaxis(resized, -1, 0)
