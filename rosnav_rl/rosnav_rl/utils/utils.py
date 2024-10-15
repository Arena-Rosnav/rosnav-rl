import json
import os
import random
from typing import Tuple

import numpy as np
import rospkg
import rospy
import torch
import yaml
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from task_generator.constants import Constants
from task_generator.utils import Utils


def get_robot_yaml_path(robot_model: str = None) -> str:
    robot_model = rospy.get_param(os.path.join(rospy.get_namespace(), "model"))

    simulation_setup_path = rospkg.RosPack().get_path("arena_simulation_setup")
    return os.path.join(
        simulation_setup_path, "entities", "robots", robot_model, f"model_params.yaml"
    )


def get_laser_from_robot_yaml(robot_model: str = None) -> Tuple[int, int, int, int]:
    robot_yaml_path = get_robot_yaml_path(robot_model)

    with open(robot_yaml_path, "r") as fd:
        robot_data = yaml.safe_load(fd)
        laser_data = robot_data["laser"]

        rospy.set_param(
            os.path.join(rospy.get_namespace(), "laser/num_beams"),
            laser_data["num_beams"],
        )

        return (
            laser_data["num_beams"],
            laser_data["angle"]["min"],
            laser_data["angle"]["max"],
            laser_data["angle"]["increment"],
        )


def get_actions_from_robot_yaml(robot_model: str = None):
    robot_yaml_path = get_robot_yaml_path(robot_model)

    with open(robot_yaml_path, "r") as fd:
        robot_data = yaml.safe_load(fd)
        action_data = robot_data["actions"]

    return action_data


def load_json(file_path: str) -> dict:
    with open(file_path) as file:
        return json.load(file)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def make_mock_env(ns: str, agent_description) -> DummyVecEnv:
    import rl_utils.envs.flatland_gymnasium_env as flatland_gym_env
    import rl_utils.envs.unity as arena_unity_env

    def _init_flatland_env():
        return flatland_gym_env.FlatlandEnv(
            ns=ns,
            agent_description=agent_description,
            reward_fnc=None,
            init_by_call=False,
        )

    def _init_arena_unity_env():
        return arena_unity_env.UnityEnv(
            ns=ns,
            agent_description=agent_description,
            reward_fnc=None,
            init_by_call=False,
        )

    sim = Utils.get_simulator()
    if sim == Constants.Simulator.UNITY:
        return DummyVecEnv([_init_arena_unity_env])
    elif sim == Constants.Simulator.FLATLAND:
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
