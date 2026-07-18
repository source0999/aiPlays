from aiplays.config import load_config
from aiplays.envs.pyboy_env import PyBoyEnv
from aiplays.testing import FakeBackend


def test_pyboy_environment_sequences_input_and_closes() -> None:
    config = load_config("configs/pokemon_red.yaml")
    config.game.state_path = None
    config.observation.mode = "pixels"
    backend = FakeBackend()
    env = PyBoyEnv(config, backend_factory=lambda: backend)
    observation, _ = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    observation, _, _, _, _ = env.step(5)
    assert env.observation_space.contains(observation)
    assert backend.events[:2] == [("press", "a"), ("release", "a")]
    env.close()
    assert backend.stopped


def test_manual_ticks_do_not_consume_episode_steps() -> None:
    config = load_config("configs/pokemon_red.yaml")
    config.game.state_path = None
    backend = FakeBackend()
    env = PyBoyEnv(config, backend_factory=lambda: backend)
    env.reset(seed=7)
    env.manual_tick()
    assert backend.ticks == 1
    assert env.steps == 0
    assert backend.events == []
    env.close()
