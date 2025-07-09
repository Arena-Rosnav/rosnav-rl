from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
)

import message_filters
import rclpy
import threading
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.subscription import Subscription
from rclpy.time import Duration, Time

from ..observations import (
    BaseUnit,
    ObservationCollectorUnit,
    ObservationGeneratorUnit,
)
from ..observations.collectors.base_collector import SimulationNotCompatibleError
from ..states import SimulationStateContainer
from ..utils.rostopic import Namespace
from .dependency_resolution import explore_dependency_hierarchy
from .generic_observation import GenericObservation


@dataclass
class ObservationHealth:
    """Tracks the health status of an observation source with improved metrics."""

    name: str
    clock: rclpy.clock.Clock = rclpy.clock.Clock(
        clock_type=rclpy.clock.ClockType.ROS_TIME
    )
    last_update_time: Time = field(init=False)
    update_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    latest_error: Optional[Exception] = None

    def __post_init__(self):
        """Initialize non-parameter fields."""
        self.last_update_time = Time(clock_type=self.clock.clock_type)

    def record_update(self) -> None:
        """Record a successful observation update with timestamp."""
        self.last_update_time = self.clock.now()
        self.update_count += 1
        self.consecutive_errors = 0

    def record_error(self, error: Exception) -> None:
        """Record observation error with categorization."""
        self.error_count += 1
        self.consecutive_errors += 1
        self.latest_error = error

    @property
    def is_healthy(self) -> bool:
        """Health status based on update history and error patterns."""
        return self.update_count > 0 and self.consecutive_errors < 3

    @property
    def time_since_update(self) -> Duration:
        """Time elapsed since last update, with millisecond precision."""
        return self.clock.now() - self.last_update_time

    @property
    def status_summary(self) -> Dict[str, Any]:
        """Comprehensive health status report."""
        return {
            "is_healthy": self.is_healthy,
            "update_count": self.update_count,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "time_since_update_sec": self.time_since_update.nanoseconds / 1e9,
            "latest_error": str(self.latest_error) if self.latest_error else None,
        }


T = TypeVar("T")


class ObservationManager:
    """Manages observation collection and generation for reinforcement learning agents in ROS2.

    Features:
    - Fully compatible with ROS2 middleware
    - Enhanced error handling and recovery
    - Configurable QoS profiles for robust communication
    - Health monitoring with detailed diagnostics
    - Thread-safe observation collection
    - Dynamic topic remapping
    """

    # =====================================================================
    # Initialization and Setup
    # =====================================================================
    def __init__(
        self,
        node: Node,
        ns: Namespace,
        obs_structure: List[Type[BaseUnit]],
        simulation_state_container: SimulationStateContainer = None,
        topic_mappings: Optional[Dict[str, Callable]] = None,
        obs_unit_kwargs: Optional[dict] = None,
        wait_for_obs: bool = True,
        qos_profile: Optional[QoSProfile] = None,
        enable_synchronization: bool = True,
        sync_tolerance_seconds: float = 0.1,
        buffer_size: int = 50,
    ) -> None:
        """
        Initializes the ObservationManager which manages observations from various ROS topics.

        The ObservationManager handles the subscription to various observation topics,
        processes the incoming data, and maintains the observation state with temporal synchronization.

        Args:
            node (Node): The ROS node used for creating subscribers
            ns (Namespace): Namespace for the ROS topics
            obs_structure (List[BaseUnit]): List of observation units that define the structure of observations
            simulation_state_container (SimulationStateContainer): Container for simulation state data
            topic_mappings (Optional[Dict[str, Callable]]): Dictionary mapping topic patterns to transformation
                                                            functions
            obs_unit_kwargs (Optional[dict]): Additional keyword arguments to pass to observation units. Default: None
            wait_for_obs (bool): Whether to wait for initial observations before proceeding. Default: True
            qos_profile (Optional[QoSProfile]): Quality of Service profile for subscribers. Default: None
                If None, a RELIABLE profile with KEEP_LAST history policy and depth 5 will be used.
            enable_synchronization (bool): Enable temporal synchronization of observations. Default: True
            sync_tolerance_seconds (float): Tolerance for synchronization timestamps. Default: 0.05
            buffer_size (int): Size of temporal buffer for each observation stream. Default: 50

        Attributes:
            _node: The ROS node instance
            _logger: Logger from the ROS node
            _ns: The namespace for topics
            _simulation_state_container: Container for simulation state
            _obs_structure: Hierarchical observation structure with dependencies resolved
            _topic_mappings: Dictionary of topic mapping functions
            _collectable_observations: Dictionary of available observations
            _subscribers: Dictionary of topic subscribers
            _health_monitors: Dictionary of observation health monitors
            _wait_for_obs: Whether to wait for observations
            _obs_timeout: Timeout for observation waiting
            _qos_profile: Quality of Service profile for subscribers
            _synchronizer: Observation synchronizer for temporal coordination
        """
        if simulation_state_container is None:
            simulation_state_container = SimulationStateContainer()
            print(
                "No simulation state container provided. Using default empty container."
            )

        self._node = node
        self._logger = node.get_logger()
        self._ns = Namespace(ns)
        self._simulation_state_container = simulation_state_container

        # Set of observation units with dependencies resolved
        # This will be a list of all units in the dependency chain
        # e.g. when a generator depends on a collector, both will be included
        self._obs_structure = list(explore_dependency_hierarchy(obs_structure).keys())

        # Configure topic mapping with fallback patterns
        self._topic_mappings = topic_mappings or {}
        # self._topic_mappings.update(
        #     {
        #         ".*crowdsim.*": lambda topic, manager: (
        #             Topic(topic).name if manager._is_single_env else str(Topic(topic))
        #         )
        #     }
        # )

        # Initialize containers and configuration
        self._collectable_observations: Dict[str, GenericObservation] = {}
        self._subscribers: Dict[str, Subscription] = {}
        self._health_monitors: Dict[str, ObservationHealth] = {}
        self._wait_for_obs = wait_for_obs

        # Synchronization parameters
        self._enable_synchronization = enable_synchronization
        self._sync_tolerance = sync_tolerance_seconds
        self._buffer_size = buffer_size

        # New attributes for message_filters
        self._lock = threading.Lock()
        self._message_filter_synchronizer = None
        self._sync_collector_names: List[str] = []

        # Set default QoS profile with reliability guarantees if not provided
        self._qos_profile = qos_profile or QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Initialize unit parameters
        obs_unit_kwargs = obs_unit_kwargs or {}
        obs_unit_kwargs.update(
            {
                "ns": self._ns,
                "simulation_state_container": simulation_state_container,
                "node": self._node,
            }
        )

        # Set up observation units and subscriptions
        self._initialize_units(obs_unit_kwargs=obs_unit_kwargs)
        self._setup_collectors()

    def _initialize_units(self, obs_unit_kwargs: dict) -> None:
        """Initialize collector and generator units with improved error handling."""
        # Separate collectors and generators
        collector_classes = [
            unit
            for unit in self._obs_structure
            if issubclass(unit, ObservationCollectorUnit)
        ]
        generator_classes = [
            unit
            for unit in self._obs_structure
            if issubclass(unit, ObservationGeneratorUnit)
        ]

        # Initialize collectors with graceful failure handling
        self._collectors: Dict[str, ObservationCollectorUnit] = {}
        for collector_class in collector_classes:
            try:
                self._collectors[collector_class.name] = collector_class(
                    **obs_unit_kwargs
                )
            except SimulationNotCompatibleError as e:
                self._logger.warn(
                    f"Collector '{collector_class.name}' not compatible: {e}"
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to initialize collector '{collector_class.name}': {e}"
                )

        # Initialize generators
        self._generators: Dict[str, ObservationGeneratorUnit] = {}
        for generator_class in generator_classes:
            try:
                self._generators[generator_class.name] = generator_class(
                    **obs_unit_kwargs
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to initialize generator '{generator_class.name}': {e}"
                )

    def _setup_collectors(self) -> None:
        """Set up observation containers and ROS2 subscribers with robust error handling."""
        sync_collectors = {}
        if self._enable_synchronization:
            sync_collectors = {
                name: collector
                for name, collector in self._collectors.items()
                if getattr(collector, "up_to_date_required", False)
            }

        self._sync_collector_names = list(sync_collectors.keys())
        self._non_sync_collector_names = [
            name for name in self._collectors if name not in self._sync_collector_names
        ]

        # Setup for non-synchronized collectors
        for collector_name in self._non_sync_collector_names:
            collector = self._collectors[collector_name]
            observation_container = GenericObservation(
                initial_msg=collector.msg_data_class(),
                process_fnc=collector.safe_preprocess,
            )
            self._collectable_observations[collector_name] = observation_container
            self._health_monitors[collector_name] = ObservationHealth(
                name=collector_name, clock=self._node.get_clock()
            )

            _topic = self._get_mapped_topic(collector)
            callback = partial(
                self._observation_callback, collector_name=collector_name
            )
            self._subscribers[collector_name] = self._node.create_subscription(
                collector.msg_data_class,
                str(_topic),
                callback,
                qos_profile=self._qos_profile,
            )
            self._logger.debug(
                f"Subscribed to {_topic} for non-synchronized observation '{collector_name}'"
            )

        # Setup for synchronized collectors
        if sync_collectors:
            subscribers = []
            for collector_name, collector in sync_collectors.items():
                observation_container = GenericObservation(
                    initial_msg=collector.msg_data_class(),
                    process_fnc=collector.safe_preprocess,
                )
                self._collectable_observations[collector_name] = observation_container
                self._health_monitors[collector_name] = ObservationHealth(
                    collector_name
                )

                _topic = self._get_mapped_topic(collector)
                subscribers.append(
                    message_filters.Subscriber(
                        self._node,
                        collector.msg_data_class,
                        str(_topic),
                        qos_profile=self._qos_profile,
                    )
                )
                self._logger.debug(
                    f"Preparing {_topic} for synchronized observation '{collector_name}'"
                )

            self._message_filter_synchronizer = (
                message_filters.ApproximateTimeSynchronizer(
                    subscribers,
                    queue_size=self._buffer_size,
                    slop=self._sync_tolerance,
                )
            )
            self._message_filter_synchronizer.registerCallback(
                self._synchronized_callback
            )

            self._logger.info(
                f"Using message_filters for: {self._sync_collector_names}"
            )

    def _get_mapped_topic(self, collector: ObservationCollectorUnit) -> str:
        """Determine the correct topic name with namespace and mappings applied."""
        # Determine base topic with namespace
        _topic = (
            self._ns(collector.topic)
            if collector.is_topic_agent_specific
            else self._ns.simulation_ns(collector.topic)
        )

        # Apply mapping functions that match patterns
        for pattern, mapping in self._topic_mappings.items():
            if re.match(pattern, str(collector.topic)):
                _topic = mapping(_topic, self)

        return _topic

    # =====================================================================
    # Public API
    # =====================================================================
    def get_observations(self, **extra_observations) -> Dict[str, Any]:
        """
        Collects and generates all observations, ensuring temporal synchronization
        for sensor data.

        Args:
            **extra_observations: Additional observations to include.

        Returns:
            Dict[str, Any]: Complete observation dictionary.
        """
        obs_dict = {}

        obs_dict = asyncio.run(
            self._get_collectable_observations(obs_dict, self.collectors)
        )

        obs_dict.update(extra_observations)

        # Generate derived observations
        self._get_generatable_observations(
            obs_dict=obs_dict,
            simulation_state_container=self._simulation_state_container,
        )

        return obs_dict

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive health status for all observation sources."""
        return {
            name: monitor.status_summary
            for name, monitor in self._health_monitors.items()
        }

    def shutdown(self) -> None:
        """Clean up subscriptions and resources."""
        for name, subscription in self._subscribers.items():
            self._node.destroy_subscription(subscription)

        self._logger.info("ObservationManager shutdown complete")

    # =====================================================================
    # Internal Observation Handling
    # =====================================================================
    async def _get_collectable_observations(
        self, obs_dict: Dict[str, Any], collector_names: List[str]
    ) -> Dict[str, Any]:
        """
        Collects observations from ROS topics, waiting for required updates in parallel
        to ensure temporal consistency.
        """
        self._logger.debug(
            f"Starting parallel observation collection for {collector_names}."
        )

        tasks = {}
        for name in collector_names:
            observation_container = self._collectable_observations.get(name)
            if not observation_container:
                self._logger.error(f"No observation container for '{name}', skipping.")
                obs_dict[name] = None
                continue

            collector_unit = self._collectors.get(name)
            if not collector_unit:
                self._logger.error(f"No collector unit found for '{name}', skipping.")
                obs_dict[name] = None
                continue

            if (
                self._wait_for_obs
                and observation_container.stale
                and collector_unit.up_to_date_required
            ):
                self._logger.debug(f"Stale observation '{name}' requires update.")
                tasks[name] = self._wait_for_observation(name)

        if tasks:
            self._logger.debug(f"Awaiting updates for: {list(tasks.keys())}")
            results = await asyncio.gather(*tasks.values())
            for name, success in zip(tasks.keys(), results):
                if not success:
                    self._logger.warn(
                        f"Failed to get non-stale observation for '{name}'. "
                        "Using existing value."
                    )

        with self._lock:
            # Retrieve all observation values after waiting
            for name in collector_names:
                obs_dict[name] = self._collectable_observations[name].value
                if obs_dict[name] is None:
                    self._logger.debug(f"Observation '{name}' has a None value.")

        self._invalidate_observations(collector_names)
        self._logger.debug("Finished parallel observation collection.")
        return obs_dict

    def _get_generatable_observations(
        self,
        obs_dict: Dict[str, Any],
        simulation_state_container: SimulationStateContainer,
    ) -> Dict[str, Any]:
        """Generate derived observations from collected data."""
        # Note: If generator.safe_generate methods are CPU-bound and long-running,
        # they might still block the event loop if not handled carefully (e.g., run in executor).
        # For now, we assume they are reasonably fast or their internal operations are async.
        for generator_name, generator in self._generators.items():
            try:
                # If safe_generate itself becomes async, it should be awaited.
                # For now, assume it's synchronous but called within an async method.
                obs_dict[generator_name] = generator.safe_generate(
                    obs_dict=obs_dict,  # This implies sequential dependency if generators use output of others
                    simulation_state_container=simulation_state_container,
                )
            except KeyError as e:
                self._logger.warn(
                    f"Missing dependency: {e}. "
                    f"Cannot generate observation for '{generator_name}'."
                )
                obs_dict[generator_name] = None  # Or some default error value
            except Exception as e:
                self._logger.error(
                    f"Error generating observation '{generator_name}': {e}"
                )
                obs_dict[generator_name] = None  # Or some default error value
        return obs_dict

    async def _wait_for_observation(self, collector_name: str) -> bool:
        """
        Waits for a specific observation to become non-stale using a polling
        loop with a timeout.
        """
        if collector_name not in self._collectors:
            self._logger.error(f"Cannot wait for unknown collector '{collector_name}'")
            return False

        observation = self._collectable_observations[collector_name]

        try:
            await asyncio.wait_for(
                self._poll_for_update(observation, collector_name),
                timeout=self._collectors[collector_name].timeout,
            )
            self._logger.debug(f"Observation '{collector_name}' updated successfully.")
            return True
        except asyncio.TimeoutError:
            self._logger.warn(
                f"Timeout waiting for '{collector_name}' after {self._collectors[collector_name].timeout}s. "
                "Observation remained stale."
            )
            # Ensure health monitor records the timeout
            if observation.stale:
                self._health_monitors[collector_name].record_error(
                    TimeoutError(f"Timeout waiting for '{collector_name}'")
                )
            return False
        except Exception as e:
            self._logger.error(
                f"Exception while waiting for '{collector_name}': {e}",
                exc_info=True,
            )
            self._health_monitors[collector_name].record_error(e)
            return False

    async def _poll_for_update(
        self, observation: GenericObservation, collector_name: str
    ) -> None:
        """
        Continuously polls for an observation update by spinning the ROS node.
        This loop is intended to be wrapped by `asyncio.wait_for`.
        """
        self._logger.debug(f"Polling for update on '{collector_name}'...")
        while observation.stale:
            # Run the blocking rclpy.spin_once in a separate thread
            await asyncio.to_thread(rclpy.spin_once, self._node, timeout_sec=0.01)
            # Yield control to the event loop to prevent blocking
            await asyncio.sleep(0.001)

    def _invalidate_observations(self, collector_names: List[str]) -> None:
        """Mark all observations as stale to ensure fresh data on next collection."""
        for name in collector_names:
            if name in self._collectable_observations:
                self._collectable_observations[name].invalidate()

    # =====================================================================
    # Callbacks
    # =====================================================================
    def _observation_callback(self, msg: Any, collector_name: str) -> None:
        """Process incoming messages with comprehensive error handling."""
        try:
            with self._lock:
                self._collectable_observations[collector_name].update(msg)
                self._health_monitors[collector_name].record_update()
        except Exception as e:
            self._health_monitors[collector_name].record_error(e)
            self._logger.error(
                f"Error updating observation '{collector_name}': {str(e)}"
            )

    def _synchronized_callback(self, *msgs: Any) -> None:
        """Callback for synchronized messages from message_filters."""
        with self._lock:
            for collector_name, msg in zip(self._sync_collector_names, msgs):
                try:
                    self._collectable_observations[collector_name].update(msg)
                    self._health_monitors[collector_name].record_update()
                except Exception as e:
                    self._health_monitors[collector_name].record_error(e)
                    self._logger.error(
                        f"Error updating synchronized observation '{collector_name}': {str(e)}"
                    )

    @property
    def collectors(self) -> List[str]:
        """Returns the names of all collectors managed by this ObservationManager."""
        return list(self._collectors.keys())

    @property
    def generators(self) -> List[str]:
        """Returns the names of all generators managed by this ObservationManager."""
        return self._generators.keys()
