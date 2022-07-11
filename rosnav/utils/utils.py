from rosnav.utils.constants import RosnavEncoder
import rospy
import rospkg
import yaml
import os
from gym import spaces
import numpy as np


def get_robot_yaml_path():
    robot_model = rospy.get_param("model")

    simulation_setup_path = rospkg.RosPack().get_path("arena-simulation-setup")
    return os.path.join(
        simulation_setup_path, "robot", robot_model, f"{robot_model}.model.yaml"
    )

def get_robot_space_encoder():
    return rospy.get_param("space_encoder", "RobotSpecificEncoder")


def get_observation_space():
    observation_space = RosnavEncoder[get_robot_space_encoder()]

    return observation_space["lasers"], observation_space["meta"]


def stack_spaces(*ss):
    low = []
    high = []

    for space in ss:
        low.extend(space.low.tolist())
        high.extend(space.high.tolist())
    
    return spaces.Box(np.array(low).flatten(), np.array(high).flatten())