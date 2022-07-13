import rospy
import rospkg
import os
import sys
from stable_baselines3 import PPO

from rosnav.srv import GetAction, GetActionResponse
from rosnav.rosnav_space_manager.rosnav_space_manager import RosnavSpaceManager


from rosnav import *
sys.modules["rl_agent"] = sys.modules["rosnav"]


class RosnavNode:
    def __init__(self):
        # Set RosnavSpaceEncoder as Middleware
        self._encoder = RosnavSpaceManager()

        # Load the model
        self._agent = self._get_model()

        self._get_next_action_srv = rospy.Service(
            "rosnav/get_action", GetAction, self._handle_next_action_srv
        )

    def _handle_next_action_srv(self, request):
        observation = self._encoder.encode_observation({
            "goal_in_robot_frame": request.goal_in_robot_frame,
            "laser_scan": request.laser_scan,
            "last_action": request.last_action
        })

        action = self._agent.predict(observation, deterministic=True)[0]

        decoded_action = self._encoder.decode_action(action)

        response = GetActionResponse()
        response.action = decoded_action

        return response

    def _get_model(self):
        agent_name = rospy.get_param("agent_name")
        agent_path = self._get_model_path(agent_name)

        assert os.path.isfile(agent_path), f"Model cannot be found at {agent_path}"

        return PPO.load(agent_path).policy

    def _get_model_path(self, model_name):
        return os.path.join(
            rospkg.RosPack().get_path("rosnav"),
            "agents",
            model_name,
            "best_model.zip"
        )


if __name__ == "__main__":
    rospy.init_node("rosnav_node")

    node = RosnavNode()

    while not rospy.is_shutdown():
        rospy.spin()