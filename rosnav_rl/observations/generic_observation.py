from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from ..utils.type_aliases.ros import (
    _RospyMessage,
)

ProcessedObservation = TypeVar("ProcessedObservation", bound=Any)


class GenericObservation(Generic[_RospyMessage]):
    """A generic observation class that processes and stores ROS messages of any type.

    This class provides a way to handle any type of ROS message with custom processing functionality.
    It keeps track of whether the observation is stale (outdated) and provides methods to update
    and invalidate the observation.

    Type Parameters:
        _RospyMessage: The type of ROS message this observation handles.

    Attributes:
        _value (ProcessedObservation): The processed observation data.
        _stale (bool): Flag indicating whether the observation is stale (True) or fresh (False).
        _msg (_RospyMessage): The latest raw message received.
        _process_fnc (Callable): Function that processes raw messages into usable observation data.

    Methods:
        __init__: Initialize the observation with an initial message and processing function.
        update: Update the observation with a new message.
        invalidate: Mark the observation as stale.
        
    Properties:
        stale: Get or set whether the observation is stale.
        value: Get the processed observation value.
    """

    _value: ProcessedObservation
    _stale: bool

    def __init__(
        self,
        initial_msg: _RospyMessage,
        process_fnc: Callable[[_RospyMessage], ProcessedObservation] = lambda x: x,
    ) -> None:
        """
        Initializes a new instance of the GenericObservation class.

        Args:
            initial_msg (T): The initial observation message.
            process_fnc (ProcessingFnc, optional): The function used to process the observation.
                Defaults to lambda x: x.
        """
        self._msg = initial_msg
        self._process_fnc = process_fnc

        self._value = process_fnc(initial_msg)
        self._stale = True

    @property
    def stale(self) -> bool:
        """
        Gets or sets the stale flag of the observation.

        Returns:
            bool: True if the observation is stale, False otherwise.
        """
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

    def update(self, msg: _RospyMessage) -> None:
        """
        Updates the observation with a new message.

        Args:
            msg (_RospyMessage): The new observation message.
        """
        self._msg = msg
        self._value = self._process_fnc(msg)
        self._stale = False

    def invalidate(self):
        """
        Invalidates the observation by setting the stale flag to True.
        """
        self._stale = True
