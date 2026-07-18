from pathlib import Path

import pytest

from aiplays.config import load_config


def test_config_loads() -> None:
    config = load_config("configs/default.yaml")
    assert config.observation.width == 84


def test_missing_config_is_actionable() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("missing.yaml"))
