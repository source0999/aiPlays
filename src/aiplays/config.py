from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aiplays.actions import DEFAULT_ACTIONS, Action, parse_actions
from aiplays.paths import resolve_from_root


@dataclass
class GameConfig:
    adapter: str = "generic"
    rom_path: str = "roms/game.gb"
    state_path: str | None = None


@dataclass
class EmulatorConfig:
    window: str | None = None
    speed: int = 0
    frame_skip: int = 8
    action_hold_frames: int = 4
    release_frames: int = 1
    max_episode_steps: int = 20_000
    enabled_actions: tuple[Action, ...] = DEFAULT_ACTIONS


@dataclass
class ObservationConfig:
    mode: str = "pixels"
    grayscale: bool = True
    width: int = 84
    height: int = 84
    frame_stack: int = 4


@dataclass
class TrainingConfig:
    algorithm: str = "ppo"
    device: str = "auto"
    seed: int = 42
    num_envs: int = 1
    total_timesteps: int = 1_000_000
    checkpoint_frequency: int = 25_000
    learning_rate: float = 0.00025
    n_steps: int = 128
    batch_size: int = 64


@dataclass
class DashboardConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class PathsConfig:
    models_dir: str = "models"
    logs_dir: str = "logs"
    runs_dir: str = "runs"


@dataclass
class RewardsConfig:
    profile: str = "exploration"
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class AppConfig:
    game: GameConfig = field(default_factory=GameConfig)
    emulator: EmulatorConfig = field(default_factory=EmulatorConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = resolve_from_root(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    game = GameConfig(**_section(raw, "game"))
    emulator_data = _section(raw, "emulator")
    emulator_data["enabled_actions"] = parse_actions(
        emulator_data.get("enabled_actions", [action.value for action in DEFAULT_ACTIONS])
    )
    config = AppConfig(
        game=game,
        emulator=EmulatorConfig(**emulator_data),
        observation=ObservationConfig(**_section(raw, "observation")),
        training=TrainingConfig(**_section(raw, "training")),
        dashboard=DashboardConfig(**_section(raw, "dashboard")),
        paths=PathsConfig(**_section(raw, "paths")),
        rewards=RewardsConfig(**_section(raw, "rewards")),
    )
    _apply_environment(config)
    validate_config(config)
    return config


def _apply_environment(config: AppConfig) -> None:
    overrides = {
        "AIPLAYS_ROM_PATH": (config.game, "rom_path"),
        "AIPLAYS_STATE_PATH": (config.game, "state_path"),
        "AIPLAYS_DEVICE": (config.training, "device"),
        "AIPLAYS_MODELS_DIR": (config.paths, "models_dir"),
        "AIPLAYS_LOGS_DIR": (config.paths, "logs_dir"),
        "AIPLAYS_RUNS_DIR": (config.paths, "runs_dir"),
        "AIPLAYS_DASHBOARD_HOST": (config.dashboard, "host"),
    }
    for variable, (target, attribute) in overrides.items():
        if value := os.getenv(variable):
            setattr(target, attribute, value)
    if value := os.getenv("AIPLAYS_DASHBOARD_PORT"):
        config.dashboard.port = int(value)


def validate_config(config: AppConfig) -> None:
    if config.observation.mode not in {"pixels", "ram", "multimodal"}:
        raise ValueError("observation.mode must be pixels, ram, or multimodal")
    if config.observation.width <= 0 or config.observation.height <= 0:
        raise ValueError("observation dimensions must be positive")
    if config.observation.frame_stack <= 0:
        raise ValueError("observation.frame_stack must be positive")
    if config.training.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda")
    for name in ("frame_skip", "action_hold_frames", "release_frames", "max_episode_steps"):
        if getattr(config.emulator, name) < 0:
            raise ValueError(f"emulator.{name} must be non-negative")
    if config.emulator.max_episode_steps == 0 or config.training.num_envs <= 0:
        raise ValueError("max_episode_steps and num_envs must be positive")
    if config.training.n_steps <= 0 or config.training.batch_size <= 0:
        raise ValueError("training.n_steps and training.batch_size must be positive")
