"""Regression test for StableBaselinesEnv.stack() (P3.3, audit 2026-07-04).

stack() used to rebuild a fresh `dones` array and `infos` list on every call
before forwarding them to VecFrameStack's StackedObservations.update(). Since
this call site always passes an all-False `dones` (no episode-end state is
threaded through here), the update() call never mutates the corresponding
info dict, so caching and reusing `dones`/`infos` across calls is safe. This
test pins the exact stacked output over two steps to guard against a future
change to that assumption.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rosnav_rl.model.stable_baselines3.sb3_model import StableBaselinesEnv
from rosnav_rl.utils.stable_baselines3.vec_frame_stack import VecFrameStack


class _DictObsEnv(gym.Env):
    """Minimal gym.Env with a Dict observation space."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "a": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
                "b": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def _make_env():
    venv = DummyVecEnv([_DictObsEnv])
    stacked_venv = VecFrameStack(venv, n_stack=2)
    return StableBaselinesEnv(stacked_venv)


def test_stack_matches_pinned_output_over_two_steps():
    env = _make_env()

    obs1 = {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([4.0, 5.0], dtype=np.float32),
    }
    obs2 = {
        "a": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "b": np.array([40.0, 50.0], dtype=np.float32),
    }

    out1, infos1 = env.stack(obs1)
    out2, infos2 = env.stack(obs2)

    np.testing.assert_array_equal(
        out1["a"], np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out1["b"], np.array([[0.0, 0.0, 4.0, 5.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out2["a"], np.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out2["b"], np.array([[4.0, 5.0, 40.0, 50.0]], dtype=np.float32)
    )
    assert infos1 == [{}]
    assert infos2 == [{}]


def test_stack_dones_infos_reused_not_rebuilt():
    env = _make_env()
    obs = {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([4.0, 5.0], dtype=np.float32),
    }

    env.stack(obs)
    dones_ref_1 = env._stack_dones
    infos_ref_1 = env._stack_infos
    env.stack(obs)
    dones_ref_2 = env._stack_dones
    infos_ref_2 = env._stack_infos

    assert dones_ref_1 is dones_ref_2
    assert infos_ref_1 is infos_ref_2
    assert not dones_ref_2.any()
    assert infos_ref_2 == [{}]
