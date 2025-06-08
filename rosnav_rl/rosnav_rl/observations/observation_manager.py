from __future__ import annotations

import asyncio  # Add asyncio import
import re
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)  # Removed Generic, Union

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    # HistoryPolicy, # Removed
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    # ReliabilityPolicy, # Removed
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
from ..utils.rostopic import Namespace  # Removed Topic
from .dependency_resolution import explore_dependency_hierarchy
from .generic_observation import GenericObservation


@dataclass
class ObservationHealth:
    """Tracks the health status of an observation source with improved metrics."""

    name: str
    last_update_time: Time = field(default_factory=lambda: Time())
    update_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    latest_error: Optional[Exception] = None

    def record_update(self) -> None:
        """Record a successful observation update with timestamp."""
        self.last_update_time = rclpy.clock.Clock().now()
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
        return rclpy.clock.Clock().now() - self.last_update_time

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

    def __init__(
        self,
        node: Node,
        ns: Namespace,
        obs_structure: List[Type[BaseUnit]],
        simulation_state_container: SimulationStateContainer = None,
        topic_mappings: Optional[Dict[str, Callable]] = None,
        is_single_env: bool = False,
        obs_unit_kwargs: Optional[dict] = None,
        wait_for_obs: bool = True,
        obs_timeout: float = 10.0,
        qos_profile: Optional[QoSProfile] = None,
    ) -> None:
        """
        Initializes the ObservationManager which manages observations from various ROS topics.

        The ObservationManager handles the subscription to various observation topics,
        processes the incoming data, and maintains the observation state.

        Args:
            node (Node): The ROS node used for creating subscribers
            ns (Namespace): Namespace for the ROS topics
            obs_structure (List[BaseUnit]): List of observation units that define the structure of observations
            simulation_state_container (SimulationStateContainer): Container for simulation state data
            topic_mappings (Optional[Dict[str, Callable]]): Dictionary mapping topic patterns to transformation
                                                            functions
            is_single_env (bool): Flag indicating if this is a single environment setup. Default: False
            obs_unit_kwargs (Optional[dict]): Additional keyword arguments to pass to observation units. Default: None
            wait_for_obs (bool): Whether to wait for initial observations before proceeding. Default: True
            obs_timeout (float): Timeout in seconds for waiting for observations. Default: 10.0
            qos_profile (Optional[QoSProfile]): Quality of Service profile for subscribers. Default: None
                If None, a RELIABLE profile with KEEP_LAST history policy and depth 5 will be used.

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
            _is_single_env: Flag indicating single environment mode
            _qos_profile: Quality of Service profile for subscribers
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
        self._obs_timeout = obs_timeout
        self._is_single_env = is_single_env or "sim" in ns

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
        for collector_name, collector in self._collectors.items():
            # Create observation container
            observation_container = GenericObservation(
                initial_msg=collector.msg_data_class(),
                process_fnc=collector.safe_preprocess,
            )

            self._collectable_observations[collector_name] = observation_container
            self._health_monitors[collector_name] = ObservationHealth(collector_name)

            # Apply topic mapping functions
            _topic = self._get_mapped_topic(collector)

            # Create subscription with health monitoring
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
                f"Subscribed to {_topic} for observation '{collector_name}'"
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

    def _observation_callback(self, msg: Any, collector_name: str) -> None:
        """Process incoming messages with comprehensive error handling."""
        try:
            self._collectable_observations[collector_name].update(msg)
            self._health_monitors[collector_name].record_update()
        except Exception as e:
            self._health_monitors[collector_name].record_error(e)
            self._logger.error(
                f"Error updating observation '{collector_name}': {str(e)}"
            )

    def _invalidate_observations(self) -> None:
        """Mark all observations as stale to ensure fresh data on next collection."""
        for collector in self._collectable_observations.values():
            collector.invalidate()

    async def _wait_for_observation(self, collector_name: str) -> bool:
        """
        Waits for a specific observation to become non-stale by repeatedly calling
        spin_once in a thread until the observation's callback updates it or timeout.
        """
        if collector_name not in self._collectors:
            self._logger.error(f"Cannot wait for unknown collector '{collector_name}'")
            return False

        observation = self._collectable_observations[collector_name]
        timeout_duration = getattr(
            self._collectors[collector_name], "timeout", self._obs_timeout
        )

        start_time = self._node.get_clock().now()
        self._logger.debug(
            f"Actively waiting for '{collector_name}' to be updated "
            f"(current stale: {observation.stale}, timeout: {timeout_duration}s)."
        )

        while observation.stale:
            current_time = self._node.get_clock().now()
            elapsed_s = (current_time - start_time).nanoseconds / 1e9
            if elapsed_s > timeout_duration:
                self._logger.warn(
                    f"Timeout waiting for '{collector_name}' callback ({elapsed_s:.2f}s > {timeout_duration}s). "
                    f"Observation remained stale."
                )
                if observation.stale:
                    self._health_monitors[collector_name].record_error(
                        TimeoutError(
                            f"Timeout: {collector_name} not updated by callback within {timeout_duration}s"
                        )
                    )
                return False

            try:
                await asyncio.to_thread(rclpy.spin_once, self._node, timeout_sec=0.01)
                self._logger.debug(
                    f"Spin complete for '{collector_name}'. Stale: {observation.stale}. Time: {elapsed_s:.3f}s"
                )
                if not observation.stale:
                    self._logger.debug(
                        f"'{collector_name}' became non-stale after spin."
                    )
                    break  # Exit while loop if updated
            except Exception as e:
                self._logger.error(
                    f"Exception during rclpy.spin_once while waiting for '{collector_name}': {e}"
                )
                self._health_monitors[collector_name].record_error(e)
                return False

            await asyncio.sleep(0.001)  # Yield control

        if not observation.stale:
            self._logger.debug(
                f"'{collector_name}' is now non-stale. Wait loop finished."
            )
            return True
        else:
            self._logger.warn(
                f"'{collector_name}' is still stale after wait loop finished unexpectedly "
                f"(e.g. not by timeout or update)."
            )
            return False

    async def get_observations(self, **extra_observations) -> Dict[str, Any]:
        """Collect and generate all observations with additional custom data.

        Args:
            **extra_observations: Additional observations to include

        Returns:
            Dict[str, Any]: Complete observation dictionary
        """
        obs_dict = {}

        # Collect observations from topics
        await self._get_collectable_observations(obs_dict)

        # Generate derived observations
        await self._get_generatable_observations(
            obs_dict=obs_dict,
            simulation_state_container=self._simulation_state_container,
        )

        # Add extra observations
        obs_dict.update(extra_observations)

        return obs_dict

    async def _get_collectable_observations(
        self, obs_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collects observations from ROS topics, waiting if necessary."""
        self._logger.debug("Starting _get_collectable_observations.")
        for name, observation_container in self._collectable_observations.items():
            collector_unit = self._collectors.get(name)
            if not collector_unit:
                self._logger.error(f"No collector unit found for '{name}', skipping.")
                obs_dict[name] = None  # Or assign a default error value
                continue

            # Check if observation is stale and an update is required by the collector
            if observation_container.stale and collector_unit.up_to_date_required:
                self._logger.debug(
                    f"Observation '{name}' is stale and requires an update, attempting to wait..."
                )
                success = await self._wait_for_observation(name)
                if not success:
                    self._logger.warn(
                        f"Failed to get a non-stale observation for '{name}' after waiting. "
                        f"The current value (which may be stale or initial) will be used."
                    )
                    # The health monitor should have been updated by _wait_for_observation
                    # or the callback if an error occurred.
            # else:
            #    self._logger.debug(
            #        f"Observation '{name}' is not stale (stale={observation_container.stale}) "
            #        f"or does not require an up-to-date value (up_to_date_required={collector_unit.up_to_date_required})."
            #    )

            # Assign the value from the observation container.
            # This value should be the latest if _wait_for_observation was successful,
            # or the existing one if no wait was performed or if waiting failed.
            obs_dict[name] = observation_container.value

            if observation_container.value is None:
                # This might be normal if the initial message is None and no update has occurred yet,
                # or if the preprocessing can result in None.
                self._logger.debug(
                    f"Observation '{name}' has a None value after processing."
                )

        self._invalidate_observations()  # Mark observations stale for the next collection cycle.
        self._logger.debug("Finished _get_collectable_observations.")
        return obs_dict

    async def _get_generatable_observations(
        self,
        obs_dict: Dict[str, Any],
        simulation_state_container: SimulationStateContainer,
    ) -> Dict[str, Any]:
        """Generate derived observations from collected data asynchronously."""
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
