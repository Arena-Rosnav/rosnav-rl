from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, List, Type, TypeVar

import rospy
from rl_utils.utils.constants import Simulator

NameType = TypeVar("NameType", bound=str)
TopicType = TypeVar("TopicType", bound=str)
MessageType = TypeVar("MessageType")
ProcessedObservationType = TypeVar("ProcessedObservationType")


class SimulationNotCompatibleError(Exception):
    def __init__(self, message):
        super().__init__(message)


class BaseUnit(ABC):
    name: ClassVar[str]

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __repr__(self):
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
        
    Args:
        strict (bool, optional): Whether to strictly enforce requirements. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        
    Raises:
        SimulationNotCompatibleError: If the current simulator is not compatible with this collector.
    """
    name: ClassVar[str]
    topic: ClassVar[str]
    msg_data_class: ClassVar[Type[MessageType]]
    data_class: ClassVar[Type[ProcessedObservationType]] = ProcessedObservationType
    applicable_simulators: ClassVar[List[Simulator]]
    is_topic_agent_specific: ClassVar[bool] = True
    up_to_date_required: ClassVar[bool] = False

    def __init__(self, strict: bool = True, *args, **kwargs) -> None:
        try:
            import task_generator.utils as _task_generator_utils

            if (
                _task_generator_utils.Utils.get_simulator()
                not in self.applicable_simulators
            ):
                raise SimulationNotCompatibleError(
                    f"Collector '{self.name}' is not applicable for simulator {_task_generator_utils.Utils.get_simulator()}"
                )
        except ImportError:
            rospy.logwarn(
                f"[{self.__class__.__name__}] Could not import task_generator.utils. Skipping compatibility check."
            )

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
        if self.data_class and not isinstance(msg, self.msg_data_class):
            rospy.logwarn_once(
                f"[{self.__class__.__name__}] Expected {self.msg_data_class} but got {type(msg)}"
            )
