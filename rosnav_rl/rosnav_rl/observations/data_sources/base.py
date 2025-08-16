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


class Collector(Generic[RosMessageType, ProcessedDataType], DataSource):
    """
    A data source that collects data from an external source, like a ROS topic.

    This class combines the functionality of the old Collector and GenericObservation,
    providing a self-contained observation unit that handles:
    - Message subscription and preprocessing
    - Data storage and staleness tracking
    - Timestamping and health monitoring
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
        self.topic = topic
        self._node = node
        self.up_to_date_required = up_to_date_required

        # Initialize observation state (formerly GenericObservation functionality)
        self._clock = Clock(clock_type=ClockType.ROS_TIME) if node else None
        self._value: Optional[ProcessedDataType] = None
        self._stale: bool = True
        self._timestamp: Optional[Time] = None
        self._qos_profile: Optional[QoSProfile] = None
        self._latest_msg: Optional[RosMessageType] = None

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


class Generator(DataSource, RequiresProtocol, Generic[ProcessedDataType]):
    """
    A data source that generates new data from one or more other data sources.
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
            raise ValueError(
                f"Generator '{self.name}' missing required dependency: '{missing}'. "
                f"Required: {sorted(self.required_keys)}, Provided: {sorted(obs_dict.keys())}"
            ) from e
        except TypeError as e:
            # This can catch missing or unexpected arguments
            raise TypeError(self._format_type_error(e, obs_dict)) from e

    def _format_type_error(self, error: TypeError, obs_dict: Dict[str, Any]) -> str:
        """Format a detailed error message for type/argument errors."""
        lines = [
            "\n📋 Type Mismatch in '_generate' Method:",
            "",
            f"🔧 Generator: {self.name}",
            f"🆔 Type: {self.__class__.__name__}",
            f"❌ Error: {str(error)}",
            "",
            "get_observation() will return None.",
        ]

        # lines.extend(
        #     [
        #         "🔧 Solution Steps:",
        #         "   1. Check the '_generate' method signature matches 'requires' keys",
        #         "   2. Verify all required dependencies have compatible data types",
        #         "   3. Ensure '_generate' method accepts the provided argument types",
        #         "   4. Check for any type conversion issues in the data pipeline",
        #     ]
        # )

        return "\n".join(lines)
