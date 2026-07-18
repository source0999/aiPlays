from pathlib import Path

from aiplays.telemetry import read_telemetry


class DashboardState:
    def __init__(self, telemetry_path: Path):
        self.telemetry_path = telemetry_path

    def snapshot(self) -> dict[str, object]:
        return read_telemetry(self.telemetry_path)
