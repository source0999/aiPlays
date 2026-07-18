from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Telemetry:
    status: str = "idle"
    total_steps: int = 0
    episode: int = 0
    episode_reward: float = 0.0
    action: str = "noop"
    fps: float = 0.0
    elapsed_seconds: float = 0.0
    model: str | None = None
    frame_data_url: str | None = None
    ram: dict[str, Any] = field(default_factory=dict)
    reward_components: dict[str, float] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


class TelemetryWriter:
    """Best-effort atomic snapshot writer; training never depends on dashboard availability."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, data: Telemetry | dict[str, Any]) -> None:
        payload = asdict(data) if isinstance(data, Telemetry) else dict(data)
        payload["updated_at"] = time.time()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="telemetry-", suffix=".json", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, default=str)
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def read_telemetry(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return asdict(Telemetry())
