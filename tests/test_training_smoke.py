from stable_baselines3 import PPO

from aiplays.config import load_config
from aiplays.testing import FakeTrainingEnv
from aiplays.training.train import train


def test_tiny_ppo_run_saves_and_reloads(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config("D:/aiPlayer/configs/default.yaml")
    config.observation.mode = "ram"
    config.training.n_steps = 8
    config.training.batch_size = 8
    config.training.checkpoint_frequency = 8
    config.paths.models_dir = str(tmp_path / "models")
    config.paths.logs_dir = str(tmp_path / "logs")
    config.paths.runs_dir = str(tmp_path / "runs")
    path = train(config, timesteps=16, factory=FakeTrainingEnv)
    assert path.is_file()
    model = PPO.load(str(path))
    action, _ = model.predict(FakeTrainingEnv().reset()[0])
    assert action in (0, 1, 2)
