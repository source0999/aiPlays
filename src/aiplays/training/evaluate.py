from __future__ import annotations

from typing import cast

from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv


def evaluate(
    model: PPO, env: Env[object, object] | VecEnv, episodes: int = 5
) -> tuple[float, float]:
    return cast(tuple[float, float], evaluate_policy(model, env, n_eval_episodes=episodes))
