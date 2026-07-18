from __future__ import annotations

import json
import signal
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from aiplays.checkpoints import update_latest
from aiplays.config import AppConfig
from aiplays.devices import device_description, select_device
from aiplays.envs.pyboy_env import PyBoyEnv
from aiplays.paths import ensure_output_directories, resolve_from_root
from aiplays.telemetry import Telemetry, TelemetryWriter
from aiplays.training.callbacks import LatestCheckpointCallback, TelemetryCallback


def policy_for(config: AppConfig) -> str:
    return (
        "MultiInputPolicy"
        if config.observation.mode == "multimodal"
        else "CnnPolicy"
        if config.observation.mode == "pixels"
        else "MlpPolicy"
    )


def make_vector_env(
    config: AppConfig,
    render: bool,
    factory: Callable[[], Env[Any, Any]] | None = None,
) -> VecEnv:
    def make_one() -> Env[Any, Any]:
        return Monitor(
            factory() if factory else PyBoyEnv(config, render_mode="human" if render else None)
        )

    if config.training.num_envs == 1 or render:
        return DummyVecEnv([make_one])
    try:
        return SubprocVecEnv(
            [make_one for _ in range(config.training.num_envs)], start_method="spawn"
        )
    except Exception:
        return DummyVecEnv([make_one for _ in range(config.training.num_envs)])


def train(
    config: AppConfig,
    timesteps: int | None = None,
    render: bool = False,
    resume: Path | None = None,
    factory: Callable[[], Env[Any, Any]] | None = None,
) -> Path:
    ensure_output_directories(config)
    device = select_device(config.training.device)
    print(f"Selected Torch device: {device_description(device)}")
    if render:
        print(
            "Visible training uses one SDL2 environment and is much slower than headless training."
        )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = resolve_from_root(config.paths.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            config,
            default=lambda item: item.value if hasattr(item, "value") else item.__dict__,
            indent=2,
        ),
        encoding="utf-8",
    )
    models_dir = resolve_from_root(config.paths.models_dir)
    writer = TelemetryWriter(resolve_from_root(config.paths.runs_dir) / "telemetry.json")
    env = make_vector_env(config, render, factory)
    model = (
        PPO.load(str(resume), env=env, device=device)
        if resume
        else PPO(
            policy_for(config),
            env,
            device=device,
            seed=config.training.seed,
            verbose=1,
            learning_rate=config.training.learning_rate,
            n_steps=config.training.n_steps,
            batch_size=config.training.batch_size,
            tensorboard_log=str(resolve_from_root(config.paths.logs_dir) / run_id),
        )
    )
    callbacks = CallbackList(
        [
            LatestCheckpointCallback(config.training.checkpoint_frequency, run_dir / "checkpoints"),
            TelemetryCallback(writer, "latest.zip"),
        ]
    )
    interrupted = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_interrupt(signum: int, frame: object) -> None:
        nonlocal interrupted
        del signum, frame
        interrupted = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        model.learn(
            total_timesteps=timesteps or config.training.total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=resume is None,
        )
    except KeyboardInterrupt:
        print("Interrupted; writing a final checkpoint.")
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        final = run_dir / "final_model"
        model.save(str(final))
        latest = update_latest(final.with_suffix(".zip"), models_dir)
        writer.publish(
            Telemetry(
                status="interrupted" if interrupted else "completed",
                total_steps=model.num_timesteps,
                model=str(latest),
            )
        )
        env.close()
    return latest
