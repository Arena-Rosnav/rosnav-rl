"""
This module defines the base classes for the observation management system.

The components are:
- DataSource: An abstract base class for any unit that provides observation data.
- Collector: A data source that collects data from an external source (e.g., ROS topic).
- Generator: A data source that derives new data from other data sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Type, Union, Optional
from rclpy.client import TypeVar
from rclpy.clock import Clock, ClockType
from rclpy.time import Time
from rclpy.qos import QoSProfile

from rosnav_rl.utils.validation import RequiresProtocol
from rosnav_rl.utils.logging import ErrorReportingMixin, ComponentType


class DataSource(ABC):
    """Abstract base class for all observation data sources."""

    def __init__(self, name: str, **kwargs):
        self.name = name

    @abstractmethod
    def get_observation(self, obs_dict: Dict[str, Any], **kwargs) -> Any:
        """
        Returns the observation data.
        For Generators, obs_dict provides access to dependencies.
        For Collectors, it is typically unused.
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# Define a generic type for ROS messages
RosMessageType = TypeVar("RosMessageType")
# Define a generic type for the processed output
ProcessedDataType = TypeVar("ProcessedDataType")


class Collector(
    Generic[RosMessageType, ProcessedDataType], DataSource, ErrorReportingMixin
):
    """
    Collector is a generic data source class for collecting and processing data from external sources, such as ROS topics.

    This class serves as a unified observation unit, combining message subscription, preprocessing, data storage, staleness tracking, timestamping, and health monitoring. It is designed to be subclassed with specific ROS message and processed data types.

    Type Parameters:
        RosMessageType: The type of ROS message this collector handles (e.g., sensor_msgs.LaserScan).
        ProcessedDataType: The type of processed data output after preprocessing.

    Attributes:
        message_type (Type[RosMessageType]): The ROS message type handled by this collector.
        data_class (Type[ProcessedDataType]): The processed data type output.
        timeout (float): Timeout in seconds for data freshness (default: 0.1).
        fallback_value (Union[ProcessedDataType, None]): Value to use if data is unavailable.
        up_to_date_required (bool): Whether fresh data is required for this collector.

    Args:
        name (str): Name of the collector.
        topic (str): ROS topic to subscribe to.
        node: Optional ROS node for time and subscription management.
        up_to_date_required (bool): Whether fresh data is required (default: False).
        **kwargs: Additional keyword arguments.

    Methods:
        _preprocess(msg: RosMessageType) -> ProcessedDataType:
            Abstract method to convert a raw ROS message to the internal processed format.

        update(msg: RosMessageType) -> None:
            Updates the collector with a new message, processes it, and updates state.

        stale (property) -> bool:
            Indicates whether the current observation is stale.

        age (property) -> float:
            Returns the age of the last observation in seconds.

        timestamp (property) -> Optional[Time]:
            Returns the timestamp of the last update.

        set_qos_profile(profile: QoSProfile) -> None:
            Sets the QoS profile for ROS subscriptions.

        qos_profile (property) -> Optional[QoSProfile]:
            Returns the current QoS profile.

        get_observation() -> Optional[ProcessedDataType]:
            Returns the current processed observation value.

    Health Monitoring:
        update_count (int): Number of successful updates.
        error_count (int): Number of errors encountered during updates.

    Error Reporting:
        Inherits error reporting functionality from ErrorReportingMixin.
    """

    # The ROS message type this collector handles (e.g., sensor_msgs.LaserScan)
    message_type: Type[RosMessageType]

    # The data type of the output after preprocessing
    data_class: Type[ProcessedDataType]

    # Configuration
    timeout: float = 0.1
    fallback_value: Union[ProcessedDataType, None] = None
    up_to_date_required: bool = False  # Whether this collector requires fresh data

    def __init_subclass__(cls, **kwargs):
        """Automatically extract generic type parameters and set class variables."""
        super().__init_subclass__(**kwargs)

        # Extract the generic type arguments from the class
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:
                if hasattr(base, "__origin__") and base.__origin__ is Collector:
                    if hasattr(base, "__args__") and len(base.__args__) == 2:
                        cls.message_type = base.__args__[0]
                        cls.data_class = base.__args__[1]
                        break

    def __init__(
        self,
        name: str,
        topic: str,
        node=None,
        up_to_date_required: bool = False,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        # Unified error reporting
        ErrorReportingMixin.__init__(
            self, component_type=ComponentType.COLLECTOR, component_name=name
        )

        self.topic = topic
        self._node = node
        self.up_to_date_required = up_to_date_required

        # Initialize observation state (formerly GenericObservation functionality)
        self._clock = Clock(clock_type=ClockType.ROS_TIME) if node else None
        self._value: Optional[ProcessedDataType] = self._preprocess(
            self.message_type()  # Initialize with a default message type instance
        )
        self._stale: bool = True
        self._timestamp: Optional[Time] = None
        self._qos_profile: Optional[QoSProfile] = None
        self._latest_msg: Optional[RosMessageType] = self.message_type()

        # Health tracking
        self.update_count: int = 0
        self.error_count: int = 0

    @abstractmethod
    def _preprocess(self, msg: RosMessageType) -> ProcessedDataType:
        """Converts raw message to internal format."""
        pass

    def update(self, msg: RosMessageType) -> None:
        """
        Update the collector with a new message.
        This replaces the GenericObservation.update method.
        """
        try:
            self._latest_msg = msg
            self._value = self._preprocess(msg)
            self._stale = False
            self._timestamp = self._clock.now() if self._clock else None
            self.update_count += 1
        except Exception as e:
            self.error_count += 1
            self._report_error(
                f"Failed to update from message: {e}", error_type=type(e).__name__
            )
            raise e

    @property
    def stale(self) -> bool:
        """Check if the observation is stale."""
        return self._stale

    @stale.setter
    def stale(self, value: bool):
        """Set the stale flag."""
        if not isinstance(value, bool):
            raise ValueError("Stale must be a boolean value.")
        self._stale = value

    @property
    def age(self) -> float:
        """Get the age of the observation in seconds."""
        if not self._timestamp or not self._clock:
            return float("inf")
        duration = self._clock.now() - self._timestamp
        return duration.nanoseconds / 1e9

    @property
    def timestamp(self) -> Optional[Time]:
        """Get the timestamp of the last update."""
        return self._timestamp

    def set_qos_profile(self, profile: QoSProfile) -> None:
        """Set the QoS profile for ROS subscriptions."""
        self._qos_profile = profile

    @property
    def qos_profile(self) -> Optional[QoSProfile]:
        """Get the QoS profile."""
        return self._qos_profile

    def get_observation(self) -> Optional[ProcessedDataType]:
        """
        Implementation of DataSource.get_observation for collectors.
        Simply returns the current processed value.
        """
        return self._value


class Generator(
    DataSource, RequiresProtocol, Generic[ProcessedDataType], ErrorReportingMixin
):
    """
    A base class for data sources that generate new data from one or more other data sources.

    This class is intended to be subclassed by concrete generator implementations. It manages
    dependency requirements, error reporting, and enforces the implementation of a core data
    generation method.

    Type Parameters:
        ProcessedDataType: The type of data produced by the generator.

    Class Attributes:
        requires (Dict[str, str]): A mapping of required dependency names to their descriptions or types.
        data_class (Type[ProcessedDataType]): The output data type, automatically set via generics.

    Methods:
        __init_subclass__(cls, **kwargs):
            Automatically extracts the generic type parameter and sets the `data_class` attribute.

        __init__(self, name: str, **kwargs):
            Initializes the generator with a name and sets up error reporting and required keys.

        _generate(self, **kwargs: Any) -> ProcessedDataType:
            Abstract method to be implemented by subclasses. Contains the core logic for generating data.
            For dict inputs, dependencies are passed as keyword arguments.
            For list inputs, dependencies are passed under the 'inputs' keyword.

        get_observation(self, obs_dict: Dict[str, Any], **kwargs) -> ProcessedDataType:
            Retrieves an observation by invoking the `_generate` method with the required dependencies.
            Handles missing or unexpected arguments and reports errors accordingly.

        _format_type_error(self, error: TypeError, obs_dict: Dict[str, Any]) -> str:
            Formats a detailed error message for type or argument errors in the `_generate` method.

    Usage:
        Subclass `Generator` and implement the `_generate` method to define custom data generation logic.
        Specify required dependencies via the `requires` class attribute.
    """

    requires: Dict[str, str] = {}
    # Output data type - automatically set via generics
    data_class: Type[ProcessedDataType]

    def __init_subclass__(cls, **kwargs):
        """Automatically extract generic type parameters and set class variables."""
        super().__init_subclass__(**kwargs)

        # Extract the output data type from generics
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:
                if hasattr(base, "__origin__") and base.__origin__ is Generator:
                    if hasattr(base, "__args__") and len(base.__args__) == 1:
                        cls.data_class = base.__args__[0]
                        break

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        # Set up error reporting
        ErrorReportingMixin.__init__(
            self, component_type=ComponentType.GENERATOR, component_name=name
        )
        self.required_keys = list(self.requires.keys())

    @abstractmethod
    def _generate(self, **kwargs: Any) -> ProcessedDataType:
        """
        The core logic of the generator.
        For dict inputs, receives dependencies as keyword arguments.
        For list inputs, receives a list of dependencies under the 'inputs' keyword.
        """
        pass

    def get_observation(self, obs_dict: Dict[str, Any], **kwargs) -> ProcessedDataType:
        try:
            return self._generate(
                **{k: obs_dict[k] for k in self.required_keys}, **kwargs
            )
        except KeyError as e:
            missing = e.args[0]
            error_msg = (
                f"Missing required dependency: '{missing}'. "
                f"Required: {sorted(self.required_keys)}, Provided: {sorted(obs_dict.keys())}"
            )
            self._report_error(error_msg, error_type="KeyError")
            return None
        except TypeError as e:
            # This can catch missing or unexpected arguments
            self._report_error(
                self._format_type_error(e, obs_dict), error_type="TypeError"
            )
            return None

    def _format_type_error(self, error: TypeError, obs_dict: Dict[str, Any]) -> str:
        """Format a detailed error message for type/argument errors."""
        return (
            f"Type mismatch in '_generate' method: {str(error)}. "
            f"Check method signature matches 'requires' keys and data types are compatible."
        )
