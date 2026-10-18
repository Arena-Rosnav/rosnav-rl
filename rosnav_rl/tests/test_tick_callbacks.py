"""Tests for tick-driven curriculum staging and checkpoints."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rosnav_rl.utils.stable_baselines3.tick_callbacks import TickCheckpointCallback, stage_at


class _Drift(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        self.t += 1
        return np.zeros(1, dtype=np.float32), float(-abs(action[0])), self.t >= 16, False, {}


def test_stage_follows_the_tick_schedule():
    schedule = [0, 50000, 100000, 175000]

    assert [stage_at(schedule, t) for t in (0, 49999, 50000, 120000, 175000, 999999)] == [0, 0, 1, 2, 3, 3]


def test_checkpoints_land_once_per_tick_counted_in_control_ticks(tmp_path):
    model = PPO("MlpPolicy", DummyVecEnv([_Drift, _Drift]), n_steps=32, batch_size=32, n_epochs=1, device="cpu", seed=0)

    model.learn(total_timesteps=128, callback=TickCheckpointCallback(tmp_path, [100, 200, 5000], ticks_per_step=2))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["tick_100.zip", "tick_200.zip"]


def test_a_run_resumed_from_a_tick_checkpoint_keeps_its_count_and_skips_passed_checkpoints(tmp_path):
    first = tmp_path / "first"
    model = PPO("MlpPolicy", DummyVecEnv([_Drift, _Drift]), n_steps=32, batch_size=32, n_epochs=1, device="cpu", seed=0)
    model.learn(total_timesteps=64, callback=TickCheckpointCallback(first, [100], ticks_per_step=2))
    resumed_dir = tmp_path / "resumed"
    resumed = PPO.load(first / "tick_100", env=DummyVecEnv([_Drift, _Drift]), device="cpu")
    done = resumed.num_timesteps

    resumed.learn(
        total_timesteps=192 - done,
        reset_num_timesteps=False,
        callback=TickCheckpointCallback(resumed_dir, [100, 300], ticks_per_step=2),
    )

    assert done == 50
    assert 192 <= resumed.num_timesteps < 192 + 64
    assert sorted(p.name for p in resumed_dir.iterdir()) == ["tick_300.zip"]
