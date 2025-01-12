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
    """
    ObservationCollectorUnit is an abstract base class for collecting and preprocessing observations in a ROS environment.

    Attributes:
        name (ClassVar[str]): The name of the collector.
        topic (str): The ROS topic to subscribe to for collecting messages.
        msg_data_class (Type[MessageType]): The expected type of the incoming ROS messages.
        data_class (ProcessedObservationType): The type of the processed observation data.
        applicable_simulators (List[Constants.Simulator]): List of simulators where this collector is applicable.
        is_topic_agent_specific (bool): Indicates if the topic is specific to an agent. Defaults to True.
        up_to_date_required (bool): Indicates if the data needs to be up-to-date. Defaults to False.

    Methods:
        __init__(*args, **kwargs): Initializes the collector and checks if it is applicable for the current simulator.
        preprocess(msg: MessageType) -> ProcessedObservationType: Abstract method to preprocess the incoming message.
    """

    name: ClassVar[str]
    topic: ClassVar[str]
    msg_data_class: ClassVar[Type[MessageType]]
    data_class: ClassVar[Type[ProcessedObservationType]] = ProcessedObservationType
    applicable_simulators: ClassVar[List[Simulator]]
    is_topic_agent_specific: ClassVar[bool] = True
    up_to_date_required: ClassVar[bool] = False

    def __init__(self, strict: bool = True, *args, **kwargs) -> None:
        import task_generator.utils as _task_generator_utils

        if (
            _task_generator_utils.Utils.get_simulator()
            not in self.applicable_simulators
        ):
            raise SimulationNotCompatibleError(
                f"Collector '{self.name}' is not applicable for simulator {_task_generator_utils.Utils.get_simulator()}"
            )

    @abstractmethod
    def preprocess(self, msg: MessageType) -> ProcessedObservationType:
        if self.data_class and not isinstance(msg, self.msg_data_class):
            rospy.logwarn_once(
                f"[{self.__class__.__name__}] Expected {self.msg_data_class} but got {type(msg)}"
            )
