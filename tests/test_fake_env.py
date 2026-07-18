from stable_baselines3.common.env_checker import check_env

from aiplays.testing import FakeTrainingEnv


def test_fake_environment_passes_sb3_checker() -> None:
    check_env(FakeTrainingEnv(), warn=True)
