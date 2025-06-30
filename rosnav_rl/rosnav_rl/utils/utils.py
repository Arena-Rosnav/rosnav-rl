import json
import os
import random
from typing import Tuple

import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

# from task_generator.utils import Utils

from rosnav_rl.spaces.space_manager.base_space_manager import BaseSpaceManager


def load_json(file_path: str) -> dict:
    with open(file_path) as file:
        return json.load(file)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def make_mock_env(ns: str, space_manager: BaseSpaceManager) -> DummyVecEnv:
    # TODO: Make mock env
    # import rl_utils.envs.flatland_gymnasium_env as arena_flatland_gym_env
    # import rl_utils.envs.unity as arena_unity_env

    def _init_flatland_env():
        return arena_flatland_gym_env.FlatlandEnv(
            ns=ns,
            space_manager=space_manager,
            init_by_call=True,
        )

    def _init_arena_unity_env():
        return arena_unity_env.UnityEnv(
            ns=ns,
            space_manager=space_manager,
            init_by_call=True,
        )

    sim = Utils.get_simulator()
    if sim == Simulator.UNITY:
        return DummyVecEnv([_init_arena_unity_env])
    elif sim == Simulator.FLATLAND:
        return DummyVecEnv([_init_flatland_env])
    else:
        raise RuntimeError(
            f"Training environemnts only supported for simulators Arena Unity and Flatland but got {sim}"
        )


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
