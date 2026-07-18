from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aiplays.actions import Action
from aiplays.config import AppConfig
from aiplays.envs.base import EmulatorBackend, framebuffer
from aiplays.observations import ObservationProcessor
from aiplays.paths import resolve_from_root
from aiplays.ram.pokemon_red import PokemonRedRamReader
from aiplays.rewards.composite import CompositeReward


class PyBoyEnv(gym.Env[Any, Any]):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 60}

    def __init__(
        self,
        config: AppConfig,
        render_mode: str | None = None,
        backend_factory: Callable[[], EmulatorBackend] | None = None,
    ):
        self.config, self.render_mode, self.backend_factory = config, render_mode, backend_factory
        self.processor = ObservationProcessor(config.observation)
        self.ram_reader = PokemonRedRamReader() if config.game.adapter == "pokemon_red" else None
        self.actions = config.emulator.enabled_actions
        self.action_space = spaces.Discrete(len(self.actions))
        ram_space = spaces.Box(
            0.0, 1.0, shape=(PokemonRedRamReader.feature_space,), dtype=np.float32
        )
        if config.observation.mode == "pixels":
            self.observation_space = self.processor.pixel_space
        elif config.observation.mode == "ram":
            self.observation_space = ram_space
        else:
            self.observation_space = spaces.Dict(
                {"pixels": self.processor.pixel_space, "ram": ram_space}
            )
        self.reward = CompositeReward(config.rewards.weights)
        self.backend: EmulatorBackend | None = None
        self.steps = 0
        self.closed = False

    def _create_backend(self) -> EmulatorBackend:
        if self.backend_factory:
            return self.backend_factory()
        rom_path = resolve_from_root(self.config.game.rom_path)
        if not rom_path.is_file():
            raise FileNotFoundError(
                f"ROM not found: {rom_path}. Supply your legal ROM or set AIPLAYS_ROM_PATH."
            )
        from pyboy import PyBoy

        window = "SDL2" if self.render_mode == "human" else "null"
        backend = PyBoy(str(rom_path), window=window, sound=False)
        backend.set_emulation_speed(self.config.emulator.speed)
        return backend

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.close()
        self.closed = False
        self.backend = self._create_backend()
        state_path = self.config.game.state_path
        if state_path:
            path = resolve_from_root(state_path)
            if not path.is_file():
                self.close()
                raise FileNotFoundError(f"Configured PyBoy state is missing: {path}")
            try:
                with path.open("rb") as handle:
                    self.backend.load_state(handle)
            except Exception as exc:
                self.close()
                raise RuntimeError(
                    f"Could not load PyBoy state {path}; it may not match this ROM."
                ) from exc
        self.steps = 0
        info = self._info()
        self.reward.reset(info)
        return self._observation(), info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.backend is None or self.closed:
            raise RuntimeError("Call reset() before step()")
        action_value = self.actions[int(action)]
        try:
            if action_value is not Action.NOOP:
                self.backend.button_press(action_value.value)
            for _ in range(
                max(self.config.emulator.frame_skip, self.config.emulator.action_hold_frames)
            ):
                self.backend.tick()
            if action_value is not Action.NOOP:
                self.backend.button_release(action_value.value)
            for _ in range(self.config.emulator.release_frames):
                self.backend.tick()
        except Exception:
            self._release_all()
            raise
        self.steps += 1
        info = self._info()
        value = self.reward.value(info)
        info["action"] = action_value.value
        info["reward_components"] = self.reward.last_breakdown
        if self.config.dashboard.enabled:
            info["_frame"] = framebuffer(self.backend)
        truncated = self.steps >= self.config.emulator.max_episode_steps
        return self._observation(), value, False, truncated, info

    def _observation(self) -> Any:
        assert self.backend is not None
        pixels = (
            self.processor.reset(framebuffer(self.backend))
            if self.steps == 0
            else self.processor.append(framebuffer(self.backend))
        )
        if self.ram_reader is None:
            return pixels
        ram = self.ram_reader.features(self.ram_reader.read(self.backend.memory))
        if self.config.observation.mode == "pixels":
            return pixels
        if self.config.observation.mode == "ram":
            return ram
        return {"pixels": pixels, "ram": ram}

    def _info(self) -> dict[str, Any]:
        if self.backend is None or self.ram_reader is None:
            return {"ram": {}}
        return {"ram": self.ram_reader.read(self.backend.memory)}

    def render(self) -> np.ndarray | None:
        if self.backend is None:
            return None
        return framebuffer(self.backend) if self.render_mode == "rgb_array" else None

    def manual_tick(self) -> dict[str, Any]:
        """Advance one PyBoy frame without taking an RL action or consuming an episode step."""
        if self.backend is None or self.closed:
            raise RuntimeError("Call reset() before manual_tick()")
        self.backend.tick()
        return self._info()

    def set_emulation_speed(self, speed: int) -> None:
        """Set a PyBoy speed when supported; fake test backends need not implement it."""
        if self.backend is None or self.closed:
            raise RuntimeError("Call reset() before set_emulation_speed()")
        setter = getattr(self.backend, "set_emulation_speed", None)
        if callable(setter):
            setter(speed)

    def _release_all(self) -> None:
        if self.backend:
            for action in self.actions:
                if action is not Action.NOOP:
                    try:
                        self.backend.button_release(action.value)
                    except Exception:
                        pass

    def close(self) -> None:
        if self.backend is not None and not self.closed:
            self._release_all()
            self.backend.stop()
        self.backend = None
        self.closed = True
