import rospy
import rospkg
import os
import sys
import traceback
import json
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device

from rosnav.srv import GetAction, GetActionResponse
from rosnav.rosnav_space_manager.rosnav_space_manager import RosnavSpaceManager


from rosnav import *
sys.modules["rl_agent"] = sys.modules["rosnav"]
sys.modules["rl_utils.rl_utils.utils"] = sys.modules["rosnav.utils"]


class RosnavNode:
    def __init__(self):
        # Agent name and path
        self.agent_name = rospy.get_param("agent_name")
        self.agent_path = self._get_model_path(self.agent_name)

        assert os.path.isdir(self.agent_path), f"Model cannot be found at {self.agent_path}"

        # Load hyperparams
        self._hyperparams = self._load_hyperparams(self.agent_path)
        # rospy.set_param("/actions_in_obs", self._hyperparams.get("actions_in_observationspace", False))

        self._obs_structure = self._get_observation_space_structure(self._hyperparams)

        # Set RosnavSpaceEncoder as Middleware
        self._encoder = RosnavSpaceManager()

        # Load the model
        self._agent = self._get_model(self.agent_path)

        self._get_next_action_srv = rospy.Service(
            "rosnav/get_action", GetAction, self._handle_next_action_srv
        )

    def _handle_next_action_srv(self, request):
        observation = self._encoder.encode_observation({
            "goal_in_robot_frame": request.goal_in_robot_frame,
            "laser_scan": request.laser_scan,
            "last_action": request.last_action
        }, self._obs_structure)

        action = self._agent.predict(observation, deterministic=True)[0]

        decoded_action = self._encoder.decode_action(action)

        response = GetActionResponse()
        response.action = decoded_action

        return response

    def _get_model(self, agent_path):
        action_state_sizes = [0, 3]

        for size in action_state_sizes:
            rospy.set_param(rospy.get_namespace() + "action_state_size", size)
            try:
                return PPO.load(os.path.join(agent_path, "best_model.zip")).policy
            except:
                pass

        rospy.signal_shutdown("")

    def _get_model_path(self, model_name):
        return os.path.join(
            rospkg.RosPack().get_path("rosnav"),
            "agents",
            model_name
        )

    def _load_hyperparams(self, agent_path):
        with open(os.path.join(agent_path, "hyperparameters.json")) as file:
            return json.load(file)

    def _get_observation_space_structure(self, hyperparams):
        structure = hyperparams.get("observation_space", ["laser_scan", "goal_in_robot_frame"])

        return structure


if __name__ == "__main__":
    rospy.init_node("rosnav_node")

    node = RosnavNode()

    while not rospy.is_shutdown():
        rospy.spin()