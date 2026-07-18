from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from aiplays.checkpoints import update_latest
from aiplays.telemetry import Telemetry, TelemetryWriter


class LatestCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: Path) -> None:
        super().__init__(
            save_freq=max(1, save_freq), save_path=str(save_path), name_prefix="checkpoint"
        )
        self.models_dir = save_path.parent

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            checkpoint = Path(self.save_path) / f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            if checkpoint.exists():
                update_latest(checkpoint, self.models_dir)
        return result


class TelemetryCallback(BaseCallback):
    def __init__(self, writer: TelemetryWriter, model_name: str) -> None:
        super().__init__()
        self.writer, self.model_name = writer, model_name

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        frame = info.get("_frame")
        self.writer.publish(
            Telemetry(
                status="training",
                total_steps=self.num_timesteps,
                action=str(info.get("action", "unknown")),
                model=self.model_name,
                frame_data_url=self._frame_data_url(frame),
                ram=info.get("ram", {}),
                reward_components=info.get("reward_components", {}),
            )
        )
        return True

    @staticmethod
    def _frame_data_url(frame: object) -> str | None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
            return None
        encoded, data = cv2.imencode(".jpg", cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR))
        if not encoded:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode("ascii")
