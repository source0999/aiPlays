from __future__ import annotations

import argparse
import importlib.metadata
import socket
import sys
from pathlib import Path

import psutil
import torch

from aiplays.checkpoints import newest_checkpoint
from aiplays.config import AppConfig, load_config
from aiplays.devices import device_description, select_device
from aiplays.envs.pyboy_env import PyBoyEnv
from aiplays.paths import ensure_output_directories, repository_root, resolve_from_root
from aiplays.training.train import train


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default="configs/pokemon_red.yaml", help="YAML configuration path"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiplays", description="Local Game Boy RL with PyBoy and PPO"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="diagnose local installation")
    _config_argument(doctor)
    doctor.add_argument("--require-rom", action="store_true")
    manual = sub.add_parser("manual", aliases=["manual-play"], help="open a visible PyBoy window")
    _config_argument(manual)
    manual.add_argument(
        "--speed",
        type=int,
        help="PyBoy speed for manual play (default: 1x; 0 means unlimited)",
    )
    training = sub.add_parser("train", help="train PPO")
    _config_argument(training)
    training.add_argument("--timesteps", type=int)
    training.add_argument("--num-envs", type=int)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"])
    training.add_argument("--render", action="store_true")
    training.add_argument("--resume", type=Path)
    watch = sub.add_parser("watch", help="watch a trained checkpoint")
    _config_argument(watch)
    watch.add_argument("--model", type=Path)
    watch.add_argument("--stochastic", action="store_true")
    watch.add_argument("--episodes", type=int, default=1)
    dashboard = sub.add_parser("dashboard", help="serve local dashboard")
    _config_argument(dashboard)
    verify = sub.add_parser("verify-ram", help="print decoded RAM while the emulator runs")
    _config_argument(verify)
    return parser


def doctor(config: AppConfig, require_rom: bool) -> int:
    ensure_output_directories(config)
    rom = resolve_from_root(config.game.rom_path)
    state = resolve_from_root(config.game.state_path) if config.game.state_path else None
    port_free = False
    with socket.socket() as sock:
        try:
            sock.bind((config.dashboard.host, config.dashboard.port))
            port_free = True
        except OSError:
            pass
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {sys.platform}; CPU logical cores: {psutil.cpu_count()}")
    print(f"Repository: {repository_root()}")
    print(f"Virtual environment: {'yes' if sys.prefix != sys.base_prefix else 'no'}")
    for package in ("pyboy", "gymnasium", "stable-baselines3", "torch", "fastapi"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: MISSING")
    print(
        f"Torch CUDA: {torch.cuda.is_available()}; selected device: {device_description(select_device(config.training.device))}"
    )
    print("PyBoy import: OK")
    print(
        "Headless emulator readiness: pending legal ROM"
        if not rom.is_file()
        else "Headless emulator readiness: ROM found"
    )
    print("SDL2 visible-window readiness: test with `aiplays manual` after adding a ROM")
    print(
        f"ROM: {'found' if rom.is_file() else 'missing (expected until you add a legal ROM)'} — {rom}"
    )
    print(
        f"State: {'found' if state and state.is_file() else 'not configured/missing'}"
        + (f" — {state}" if state else "")
    )
    print(
        f"Dashboard port {config.dashboard.host}:{config.dashboard.port}: {'available' if port_free else 'in use'}"
    )
    print("Config: valid; output directories: writable")
    return 1 if require_rom and not rom.is_file() else 0


def manual(config: AppConfig, verify_ram: bool = False, speed: int | None = None) -> int:
    env = PyBoyEnv(config, render_mode="human")
    try:
        _, info = env.reset(seed=config.training.seed)
        manual_speed = 1 if speed is None else speed
        env.set_emulation_speed(manual_speed)
        print(
            "Window open at "
            f"{manual_speed}x speed. PyBoy SDL2 receives keyboard controls directly; "
            "close its window or press Ctrl+C to exit."
        )
        print(
            "Controls: arrows = D-pad, A = A, S = B, Enter = Start, Backspace = Select. "
            "PyBoy reserves Z to save and X to load its optional <rom>.state sidecar."
        )
        frame_count = 0
        while True:
            info = env.manual_tick()
            frame_count += 1
            if verify_ram and frame_count % 30 == 0:
                print(info.get("ram", {}), end="\r")
    except KeyboardInterrupt:
        return 0
    finally:
        env.close()


def watch(config: AppConfig, model_path: Path | None, stochastic: bool, episodes: int) -> int:
    from stable_baselines3 import PPO

    path = model_path or newest_checkpoint(resolve_from_root(config.paths.models_dir))
    if path is None or not path.is_file():
        print(
            "No valid checkpoint found. Train first or pass --model models\\latest.zip.",
            file=sys.stderr,
        )
        return 2
    env = PyBoyEnv(config, render_mode="human")
    try:
        model = PPO.load(str(path), device=select_device(config.training.device))
        for episode in range(episodes):
            observation, _ = env.reset(seed=config.training.seed + episode)
            total = 0.0
            while True:
                action, _ = model.predict(observation, deterministic=not stochastic)
                observation, reward, terminated, truncated, info = env.step(int(action))
                total += reward
                print(f"action={info['action']} reward={reward:.3f} total={total:.3f}", end="\r")
                if terminated or truncated:
                    print()
                    break
    except KeyboardInterrupt:
        return 0
    finally:
        env.close()
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "doctor":
        code = doctor(config, args.require_rom)
    elif args.command in {"manual", "manual-play"}:
        code = manual(config, speed=args.speed)
    elif args.command == "verify-ram":
        code = manual(config, verify_ram=True)
    elif args.command == "train":
        if args.num_envs:
            config.training.num_envs = args.num_envs
        if args.device:
            config.training.device = args.device
        train(config, args.timesteps, args.render, args.resume)
        code = 0
    elif args.command == "watch":
        code = watch(config, args.model, args.stochastic, args.episodes)
    else:
        import uvicorn

        from aiplays.dashboard.server import create_app

        print(f"Dashboard: http://{config.dashboard.host}:{config.dashboard.port}")
        uvicorn.run(
            create_app(resolve_from_root(config.paths.runs_dir) / "telemetry.json"),
            host=config.dashboard.host,
            port=config.dashboard.port,
        )
        code = 0
    raise SystemExit(code)
