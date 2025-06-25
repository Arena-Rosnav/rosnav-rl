from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, Optional

# Update to use ROS2 message types
from rclpy.qos import QoSProfile
from ..utils.type_aliases.ros import (
    _Ros2Message_T,  # Assuming this is updated for ROS2
)
from rclpy.clock import Clock, ClockType
from rclpy.time import Time

ProcessedObservation = TypeVar("ProcessedObservation", bound=Any)


class GenericObservation(Generic[_Ros2Message_T]):
    """A generic observation class that processes and stores ROS2 messages of any type.

    This class provides a way to handle any type of ROS2 message with custom processing functionality.
    It keeps track of whether the observation is stale (outdated) and provides methods to update
    and invalidate the observation.

    Type Parameters:
        _Ros2Message: The type of ROS2 message this observation handles.

    Attributes:
        _value (ProcessedObservation): The processed observation data.
        _stale (bool): Flag indicating whether the observation is stale (True) or fresh (False).
        _msg (_Ros2Message): The latest raw message received.
        _process_fnc (typing.Callable): Function that processes raw messages into usable observation data.
        _timestamp (Time): The rclpy.time.Time when the observation was last updated.
        _clock (rclpy.clock.Clock): Clock instance for getting current time.

    Methods:
        __init__: Initialize the observation with an initial message and processing function.
        update: Update the observation with a new message.
        invalidate: Mark the observation as stale.

    Properties:
        stale: Get or set whether the observation is stale.
        value: Get the processed observation value.
        age: Get the age of the observation in seconds.
    """

    _value: ProcessedObservation
    _stale: bool
    _timestamp: Time

    def __init__(
        self,
        initial_msg: _Ros2Message_T,
        process_fnc: Callable[[_Ros2Message_T], ProcessedObservation] = lambda x: x,
    ) -> None:
        """
        Initializes a new instance of the GenericObservation class.

        Args:
            initial_msg (_Ros2Message): The initial observation message.
            process_fnc (typing.Callable, optional): The function used to process the observation.
                Defaults to lambda x: x.
        """
        self._msg = initial_msg
        self._process_fnc = process_fnc
        self._qos_profile: Optional[QoSProfile] = None
        self._clock = Clock(clock_type=ClockType.ROS_TIME)

        self._value = process_fnc(initial_msg)
        self._stale = True
        self._timestamp = self._clock.now()

    @property
    def stale(self) -> bool:
        """
        Gets whether the observation is stale, either manually set or by age.

        Returns:
            bool: True if the observation is stale, False otherwise.
        """
        # Check if manually marked as stale
        return self._stale

    @stale.setter
    def stale(self, value: bool):
        """
        Sets the stale flag of the observation.

        Args:
            value (bool): The value to set for the stale flag.
        """
        self._stale = value

    @property
    def value(self) -> ProcessedObservation:
        """
        Gets the processed observation value.

        Returns:
            ProcessedObservation: The processed observation value.
        """
        return self._value

    @property
    def age(self) -> float:
        """
        Gets the age of the observation in seconds.

        Returns:
            float: The age of the observation in seconds.
        """
        duration_message = self._clock.now() - self._timestamp
        return duration_message.nanoseconds / 1e9

    @property
    def timestamp(self) -> Time:
        """
        Gets the rclpy.time.Time when the observation was last updated.

        Returns:
            rclpy.time.Time: The timestamp.
        """
        return self._timestamp

    def set_qos_profile(self, profile: QoSProfile) -> None:
        """
        Sets the QoS profile to use for this observation's subscription.

        Args:
            profile (QoSProfile): The ROS2 QoS profile
        """
        self._qos_profile = profile

    @property
    def qos_profile(self) -> Optional[QoSProfile]:
        """
        Gets the QoS profile for this observation's subscription.

        Returns:
            Optional[QoSProfile]: The ROS2 QoS profile, or None if not set
        """
        return self._qos_profile

    def update(self, msg: _Ros2Message_T) -> None:
        """
        Updates the observation with a new message.

        Args:
            msg (_Ros2Message): The new observation message.
        """
        self._msg = msg
        self._value = self._process_fnc(msg)
        self._stale = False
        self._timestamp = self._clock.now()

    def invalidate(self):
        """
        Invalidates the observation by setting the stale flag to True.
        """
        self._stale = True
