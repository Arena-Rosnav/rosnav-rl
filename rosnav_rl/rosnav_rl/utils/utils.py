import json
import os
import random
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rosnav_rl.spaces.space_manager.base_space_manager import BaseSpaceManager
from rosnav_rl.utils.stable_baselines3.vec_frame_stack import VecFrameStack


def load_json(file_path: str) -> dict:
    with open(file_path) as file:
        return json.load(file)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


class _SpaceManagerEnv(gym.Env):
    """Minimal gymnasium.Env backed by a SpaceManager's obs/action spaces.

    Used by make_mock_env() to create the DummyVecEnv needed for frame
    stacking and normalization at inference time.  No actual simulation is
    performed — step() always returns a zero observation and reward.
    """

    def __init__(self, space_manager: BaseSpaceManager):
        super().__init__()
        self.observation_space = space_manager.observation_space
        self.action_space = space_manager.action_space

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, False, False, {}


def make_mock_env(
    ns: str, space_manager: BaseSpaceManager, stack_size: int = 1
) -> DummyVecEnv:
    """Create a minimal DummyVecEnv from a SpaceManager for inference-time
    stacking and normalization support.

    The environment produces zero-valued observations and rewards and is never
    used for actual simulation — it exists solely to satisfy the SB3 VecEnv
    API required for VecFrameStack / VecNormalize wrappers.

    Args:
        ns: Unused namespace argument (kept for API compatibility).
        space_manager: The SpaceManager that defines obs/action spaces.
        stack_size: Number of frames to stack.  When > 1 the returned
            DummyVecEnv is wrapped with VecFrameStack so that the
            ``has_stack_wrapper`` flag is True and ``stack()`` works.
    """
    env = DummyVecEnv([lambda: _SpaceManagerEnv(space_manager)])
    if stack_size > 1:
        env = VecFrameStack(env, n_stack=stack_size, channels_order="first")
    return env


def wrap_vec_framestack(env: DummyVecEnv, stack_size: int) -> VecFrameStack:
    return VecFrameStack(env, n_stack=stack_size, channels_order="first")


def load_vec_normalize(path: str, venv: VecEnv = None) -> VecNormalize:
    return VecNormalize.load(path, venv)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
