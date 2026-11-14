from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosnav_rl_msgs.srv import GetCommand
from std_msgs.msg import Int16

from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.type_aliases import ObservationDict


class ObservationCollector(Protocol):
    def get_observations(self) -> ObservationDict: ...


class ActionServer(ABC):
    """ActionServer is an abstract base class for a ROS2 action server that interacts with a reinforcement learning agent.

    Attributes:
        agent (RL_Agent): The reinforcement learning agent.
        observation_collector (ObservationCollector): The observation collector for gathering environment observations.
    """

    agent: RL_Agent | None = None
    observation_collector: ObservationCollector | None = None

    def __init__(
        self, agent_name: str, namespace: str = "", node: Node | None = None, arena_env_node: str = ""
    ) -> None:
        """
        Initializes the BaseServer.

        Args:
            agent_name (str): The name of the trained agent to load.
            namespace (str, optional): The namespace for the server. Defaults to an empty string.
            node (Node, optional): An existing ROS2 node. If None, a new one is created.
            arena_env_node (str, optional): Name of the env node whose task_reset topic resets the agent. Empty disables the reset subscription.
        """
        self.agent_name = agent_name
        self.arena_env_node = arena_env_node
        self._queued_commands: list[np.ndarray] = []
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
        self._executor = SingleThreadedExecutor()

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
        if self.arena_env_node:
            self._sub_reset_stacked_obs = self.node.create_subscription(
                Int16,
                f"{self.arena_env_node}/task_reset",
                self.__on_scene_reset,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
            )

    def __handle_next_action_srv(
        self, request: GetCommand.Request, response: GetCommand.Response
    ) -> GetCommand.Response:
        """
        Handles the service request to get the next action.

        Args:
            request (GetCommand.Request): The service request.
            response (GetCommand.Response): The service response.

        Returns:
            GetCommand.Response: The service response containing the next action.
        """
        if self.agent is None:
            self.logger.info("Agent not initialized yet.")
            return response

        # Populate action_type from the agent's action space spec.
        action_space_spec = self.agent.space_manager.action_space_manager.spec
        response.action_type = action_space_spec.type

        if self.observation_collector is None:
            self.logger.info("Observation collector not initialized yet.")
            return response

        try:
            action = self.__next_command(self.agent, self.observation_collector.get_observations())
            response.action = [float(v) for v in action]
        except Exception as exc:
            # Gracefully handle transient observation failures (e.g. missing
            # topic data at startup) instead of crashing the whole node.
            self.logger.warn(
                f"[Rosnav-RL] get_action failed — returning empty command. Reason: {type(exc).__name__}: {exc}"
            )
            return response

        return response

    def __next_command(self, agent: RL_Agent, observations: ObservationDict) -> np.ndarray:
        """Pops the next queued command of a chunked policy, else queries the agent."""
        if self._queued_commands:
            observations["is_terminal"] = False
            agent.space_manager.encode_observation(observations)
            return self._queued_commands.pop(0)
        command = agent.get_action(observations, deterministic=agent.spec.deterministic_inference)
        if command.ndim == 2:
            self._queued_commands = list(command[1:])
            return command[0]
        return command

    def __on_scene_reset(self, msg: Int16):
        """
        Resets the last action and stacked observations on scenario reset.

        Args:
            msg (Int16): The reset message.
        """
        if self.agent is None:
            self.logger.info("Agent not initialized yet.")
            return
        self.agent.reset()
        self._queued_commands = []

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
        self.logger.info("[Rosnav-RL | Action Server] Observation collector initialized.")

        self.logger.info("[Rosnav-RL | Action Server] Spinning...")
        self._executor.add_node(self.node)
        self._executor.spin()

    def stop(self) -> None:
        """Stop spinning the node."""
        self._executor.shutdown()
