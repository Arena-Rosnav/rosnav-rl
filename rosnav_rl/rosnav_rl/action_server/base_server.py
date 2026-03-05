from abc import ABC, abstractmethod
from typing import Protocol

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosnav_rl_msgs.srv import GetCommand
from std_msgs.msg import Int16

from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.type_aliases import ObservationDict


class ObservationCollector(Protocol):
    def get_observations(self, *args, **kwargs) -> ObservationDict: ...


class ActionServer(ABC):
    """ActionServer is an abstract base class for a ROS2 action server that interacts with a reinforcement learning agent.

    Attributes:
        agent (RL_Agent): The reinforcement learning agent.
        observation_collector (ObservationCollector): The observation collector for gathering environment observations.
    """

    agent: RL_Agent = None
    observation_collector: ObservationCollector = None

    def __init__(self, agent_name: str, namespace: str = "", node: Node = None) -> None:
        """
        Initializes the BaseServer.

        Args:
            agent_name (str): The name of the trained agent to load.
            namespace (str, optional): The namespace for the server. Defaults to an empty string.
            node (Node, optional): An existing ROS2 node. If None, a new one is created.
        """
        self.agent_name = agent_name
        self.namespace = Namespace(namespace)
        self.node = (
            node
            if node is not None
            else rclpy.create_node(
                "rosnav_action_server",
                namespace=str(self.namespace),
            )
        )
        self.logger = self.node.get_logger()

    @abstractmethod
    def _initialize_agent(self) -> RL_Agent: ...

    @abstractmethod
    def _initialize_observation_collector(self) -> ObservationCollector: ...

    def _initialize_ros(self):
        """
        Initializes ROS2 services and subscribers for the action server.

        Sets up:
        - A service to get the next action (GetCommand).
        - A subscriber to reset stacked observations on scenario reset.
        """
        self._get_next_action_srv = self.node.create_service(
            GetCommand,
            "get_command",
            self.__handle_next_action_srv,
        )
        #TODO: Adjust episode reset trigger
        self._sub_reset_stacked_obs = self.node.create_subscription(
            Int16,
            "/scenario_reset",
            self.__on_scene_reset,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

    def __handle_next_action_srv(
        self, request: GetCommand.Request, response: GetCommand.Response
    ):
        """
        Handles the service request to get the next action.

        Args:
            request (GetCommand.Request): The service request.
            response (GetCommand.Response): The service response.

        Returns:
            GetCommand.Response: The service response containing the next action.
        """
        cmd_vel = Twist()

        if self.agent is None:
            self.logger.info("Agent not initialized yet.")
            response.twist = cmd_vel
            return response

        try:
            action = self.agent.get_action(
                self.observation_collector.get_observations()
            )
        except Exception as exc:
            # Gracefully handle transient observation failures (e.g. missing
            # topic data at startup) instead of crashing the whole node.
            self.logger.warn(
                f"[Rosnav-RL] get_action failed — returning zero velocity. "
                f"Reason: {type(exc).__name__}: {exc}"
            )
            response.twist = cmd_vel
            return response

        # Action is a numpy array with [linear.x, linear.y, angular.z]
        cmd_vel.linear.x = float(action[0])
        cmd_vel.linear.y = float(action[1])
        cmd_vel.angular.z = float(action[2])

        response.twist = cmd_vel

        return response

    def __on_scene_reset(self, msg: Int16):
        """
        Resets the last action and stacked observations on scenario reset.

        Args:
            msg (Int16): The reset message.
        """
        if self.agent is None:
            self.logger.info("Agent not initialized yet.")
            return
        self.agent.model.reset()

    def start(self):
        """
        Starts the ROS2 node and initializes the agent and observation collector.

        This method performs the following steps:
        1. Initializes ROS-related components.
        2. Initializes the agent.
        3. Initializes the observation collector.
        4. Spins the node until shutdown.
        """
        self._initialize_ros()
        self.logger.info("[Rosnav-RL | Action Server] ROS services initialized.")
        self.agent = self._initialize_agent()
        self.logger.info("[Rosnav-RL | Action Server] Agent initialized.")
        self.observation_collector = self._initialize_observation_collector()
        self.logger.info(
            "[Rosnav-RL | Action Server] Observation collector initialized."
        )

        self.logger.info("[Rosnav-RL | Action Server] Spinning...")
        rclpy.spin(self.node)
