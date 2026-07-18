from fastapi.testclient import TestClient

from aiplays.dashboard.server import create_app
from aiplays.telemetry import Telemetry, TelemetryWriter


def test_dashboard_serves_atomic_telemetry(tmp_path) -> None:
    telemetry = tmp_path / "telemetry.json"
    TelemetryWriter(telemetry).publish(Telemetry(status="training", total_steps=3))
    client = TestClient(create_app(telemetry))
    assert client.get("/api/telemetry").json()["total_steps"] == 3
    assert client.get("/").status_code == 200
