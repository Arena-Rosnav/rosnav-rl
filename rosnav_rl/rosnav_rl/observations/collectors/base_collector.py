from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, List, Optional, Type, TypeVar

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger

# TODO: Simulator definition is missing
# from rl_utils.utils.constants import Simulator


NameType = TypeVar("NameType", bound=str)
TopicType = TypeVar("TopicType", bound=str)
MessageType = TypeVar("MessageType")
ProcessedObservationType = TypeVar("ProcessedObservationType")


class SimulationNotCompatibleError(Exception):
    """Exception raised when a collector is not compatible with the current simulator."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class CollectionError(Exception):
    """Exception raised when there's an error collecting or processing an observation."""

    def __init__(
        self,
        collector_name: str,
        message: str,
        original_exception: Optional[Exception] = None,
    ) -> None:
        self.collector_name = collector_name
        self.original_exception = original_exception
        super().__init__(f"Error in collector '{collector_name}': {message}")


class BaseUnit(ABC):
    """Base class for all observation collection units."""

    name: ClassVar[str]

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.name}"


class ObservationCollectorUnit(
    BaseUnit, Generic[MessageType, ProcessedObservationType], ABC
):
    """An abstract base class for observation collectors in ROS navigation.

    This class serves as a foundation for units that collect and process observations from ROS topics.
    Each collector is responsible for subscribing to a specific topic, receiving messages, and
    preprocessing them into a standardized format for use in navigation algorithms.

    Class Attributes:
        name (ClassVar[str]): The name identifier for this collector.
        topic (ClassVar[str]): The ROS topic this collector subscribes to.
        msg_data_class (ClassVar[Type[MessageType]]): The expected ROS message type.
        data_class (ClassVar[Type[ProcessedObservationType]]): The type for processed observations.
        applicable_simulators (ClassVar[List[Simulator]]): List of simulators this collector works with.
        is_topic_agent_specific (ClassVar[bool]): Whether the topic is specific to an agent.
        up_to_date_required (ClassVar[bool]): Whether the most recent observation is required.

    Additional Methods:
        validate_message: Validates that a received message meets expected criteria.
        safe_preprocess: Wraps the preprocess method with error handling.

    Args:
        strict (bool, optional): Whether to strictly enforce requirements. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        SimulationNotCompatibleError: If the current simulator is not compatible with this collector.
    """

    name: ClassVar[str]
    topic: ClassVar[str]
    msg_data_class: ClassVar[MessageType]
    data_class: Type[ProcessedObservationType] = ProcessedObservationType
    is_topic_agent_specific: ClassVar[bool] = True
    up_to_date_required: ClassVar[bool] = False

    # Timeout configuration
    timeout: ClassVar[float] = 0.05  # seconds

    # Error tolerance configuration
    max_consecutive_errors: ClassVar[int] = 3
    fallback_value: ClassVar[Optional[Any]] = None

    def __init__(self, node: Optional[Node] = None, *args, **kwargs) -> None:
        """Initialize the observation collector.

        This method sets up the collector, including checking simulator compatibility
        and initializing error counters.

        Args:
            strict (bool, optional): Whether to strictly enforce requirements. Defaults to True.
            node (Node, optional): ROS2 node to use for logging. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            SimulationNotCompatibleError: If the current simulator is not compatible with this collector.
        """
        self._error_count = 0
        self._consecutive_errors = 0
        self._node = node
        self._logger = get_logger(self.__class__.__name__)

    def validate_message(self, msg: MessageType) -> bool:
        """
        Validate that the received message meets expected criteria.

        This method can be overridden by subclasses to perform additional validation
        beyond simple type checking.

        Args:
            msg (MessageType): The message to validate

        Returns:
            bool: True if the message is valid, False otherwise
        """
        return isinstance(msg, self.msg_data_class)

    def safe_preprocess(self, msg: MessageType) -> ProcessedObservationType:
        """
        Safely process a message with error handling.

        Args:
            msg (MessageType): The message to process

        Returns:
            ProcessedObservationType: The processed observation

        Raises:
            CollectionError: If processing fails and error tolerance is exceeded
        """
        try:
            if not self.validate_message(msg):
                self._node.get_logger().warn(
                    f"Message validation failed for {type(msg)}", once=True
                )

            result = self.preprocess(msg)
            # Reset error counters on success
            self._consecutive_errors = 0
            return result

        except Exception as e:
            self._error_count += 1
            self._consecutive_errors += 1

            if self._consecutive_errors > self.max_consecutive_errors:
                raise CollectionError(
                    collector_name=self.name,
                    message=f"Max consecutive errors ({self.max_consecutive_errors}) exceeded",
                    original_exception=e,
                )

            # Return fallback value if available
            if self.fallback_value is not None:
                self._node.get_logger().warn(f"Using fallback value after error: {e}")
                return self.fallback_value

            # Re-raise if no fallback
            raise

    @abstractmethod
    def preprocess(self, msg: MessageType) -> ProcessedObservationType:
        """
        Pre-process the received message into the desired observation format.

        This method transforms a ROS message into the format needed by the agent for processing.
        If the message is not of the expected data class (as specified by `self.msg_data_class`),
        a warning will be logged once.

        Args:
            msg (MessageType): The ROS message to be processed

        Returns:
            ProcessedObservationType: The processed observation data

        Raises:
            None: A warning is logged instead of raising an exception for type mismatches
        """
        if not isinstance(msg, self.msg_data_class):
            self._node.get_logger().warn(
                f"Expected {self.msg_data_class} but got {type(msg)}", once=True
            )
