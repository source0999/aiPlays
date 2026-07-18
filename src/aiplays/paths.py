from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiplays.config import AppConfig


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root() / path


def ensure_output_directories(config: AppConfig) -> None:
    paths = config.paths
    for value in (paths.models_dir, paths.logs_dir, paths.runs_dir):
        resolve_from_root(value).mkdir(parents=True, exist_ok=True)
