from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import rospy
from std_msgs.msg import Int16

from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.srv import GetAction, GetActionResponse
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.type_aliases import ObservationDict


class ObservationCollector(Protocol):
    def get_observations(self, *args, **kwargs) -> ObservationDict: ...


class ActionServer(ABC):
    """ActionServer is an abstract base class for a ROS action server that interacts with a reinforcement learning agent.

    Attributes:
        agent (RL_Agent): The reinforcement learning agent.
        observation_collector (ObservationCollector): The observation collector for gathering environment observations.
    """

    agent: RL_Agent = None
    observation_collector: ObservationCollector = None

    def __init__(self, agent_name: str, namespace: str = "") -> None:
        """
        Initializes the BaseServer.

        Args:
            model_path (str): The path to the model file.
            namespace (str, optional): The namespace for the server. Defaults to an empty string.
        """
        self.agent_name = agent_name
        self.namespace = Namespace(namespace)

    @abstractmethod
    def _initialize_agent(self) -> RL_Agent: ...

    @abstractmethod
    def _initialize_observation_collector(self) -> ObservationCollector: ...

    def _initialize_ros(self):
        """
        Initializes ROS services and subscribers for the action server.

        This method sets up the following ROS components:
        - A service to get the next action, which is handled by `__handle_next_action_srv`.
        - A subscriber to reset the stacked observations, which listens to the "/scenario_reset" topic and calls `__on_scene_reset`.

        Returns:
            None
        """
        self._get_next_action_srv = rospy.Service(
            str(self.namespace("rosnav/get_action")),
            GetAction,
            self.__handle_next_action_srv,
        )
        self._sub_reset_stacked_obs = rospy.Subscriber(
            "/scenario_reset", Int16, self.__on_scene_reset
        )

    def __handle_next_action_srv(self, request: GetAction):
        """
        Handles the service request to get the next action.

        Args:
            request (GetAction): The service request.

        Returns:
            GetActionResponse: The service response containing the next action.
        """
        response = GetActionResponse()
        response.action = np.array([0, 0, 0])

        if self.agent is None:
            rospy.loginfo("Agent not initialized yet.")
            return response

        action = self.agent.get_action(self.observation_collector.get_observations())
        response.action = action

        return response

    def __on_scene_reset(self, request: Int16):
        """
        Resets the last action and stacked observations.

        Args:
            request (Int16): The reset request.

        Returns:
            None
        """
        if self.agent is None:
            rospy.loginfo("Agent not initialized yet.")
            return
        self.agent.model.reset()

    def start(self):
        """
        Starts the ROS node and initializes the agent and observation collector.

        This method performs the following steps:
        1. Initializes ROS-related components.
        2. Initializes the agent.
        3. Initializes the observation collector.
        4. Enters a loop that keeps the node running until ROS is shut down.
        """
        self._initialize_ros()
        rospy.loginfo("[Rosnav-RL | Action Server] ROS services initialized.")
        self.agent = self._initialize_agent()
        rospy.loginfo("[Rosnav-RL | Action Server] Agent initialized.")
        self.observation_collector = self._initialize_observation_collector()
        rospy.loginfo("[Rosnav-RL | Action Server] Observation collector initialized.")

        rospy.loginfo("[Rosnav-RL | Action Server] Spinning...")
        while not rospy.is_shutdown():
            rospy.spin()
