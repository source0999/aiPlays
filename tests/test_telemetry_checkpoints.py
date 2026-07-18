from aiplays.checkpoints import newest_checkpoint
from aiplays.telemetry import Telemetry, TelemetryWriter, read_telemetry


def test_atomic_telemetry_and_checkpoint_discovery(tmp_path: object) -> None:
    path = tmp_path / "status.json"
    TelemetryWriter(path).publish(Telemetry(status="training", total_steps=4))
    assert read_telemetry(path)["total_steps"] == 4
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")
    assert newest_checkpoint(tmp_path) == model
