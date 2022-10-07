import os
from typing import Tuple

import numpy as np
import rospkg
import rospy
import yaml
from gym import spaces
from rosnav.utils.constants import RosnavEncoder


def get_robot_yaml_path(robot_model: str = None) -> str:
    if not robot_model:
        robot_model = rospy.get_param("model")

    simulation_setup_path = rospkg.RosPack().get_path("arena-simulation-setup")
    return os.path.join(
        simulation_setup_path, "robot", robot_model, f"{robot_model}.model.yaml"
    )


def get_laser_from_robot_yaml(robot_model: str = None) -> Tuple[int, int, int, int]:
    robot_yaml_path = get_robot_yaml_path(robot_model)

    with open(robot_yaml_path, "r") as fd:
        robot_data = yaml.safe_load(fd)

        for plugin in robot_data["plugins"]:
            if plugin["type"] == "Laser":
                laser_angle_min = plugin["angle"]["min"]
                laser_angle_max = plugin["angle"]["max"]
                laser_angle_increment = plugin["angle"]["increment"]

                _L = int(
                    round((laser_angle_max - laser_angle_min) / laser_angle_increment)
                )

                # Because RosnavEncoder ist weird...
                rospy.set_param("laser/num_beams", _L)

                return _L, laser_angle_min, laser_angle_max, laser_angle_increment


def get_observation_space_from_file(robot_model: str = None) -> Tuple[int, int]:
    actions_in_obs = rospy.get_param("/actions_in_obs", True)
    robot_state_size, action_state_size = 2, 3 if actions_in_obs else 0
    num_beams, _, _, _ = get_laser_from_robot_yaml(robot_model)

    return num_beams, action_state_size + robot_state_size


def get_robot_space_encoder() -> str:
    return rospy.get_param("space_encoder", "RobotSpecificEncoder")


def get_observation_space() -> Tuple[int, int]:
    observation_space = RosnavEncoder[get_robot_space_encoder()]

    return observation_space["lasers"], observation_space["meta"]


def stack_spaces(*ss) -> spaces.Box:
    low = []
    high = []

    for space in ss:
        low.extend(space.low.tolist())
        high.extend(space.high.tolist())

    return spaces.Box(np.array(low).flatten(), np.array(high).flatten())
