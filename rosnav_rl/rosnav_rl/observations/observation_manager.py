from __future__ import annotations

import asyncio
import threading
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
)

import message_filters
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription

from ..states import SimulationStateContainer
from ..utils.rostopic import Namespace
from ..spaces.observation_space.utils import validate_generators
from .base import Collector, DataSource, Generator
from .dependency_resolver import DependencyResolver

T = TypeVar("T")


class ObservationManager:
    """
    Manages observation collection and generation for RL agents in ROS2 using the new Collector/Generator system.

    - Instantiates collectors and generators from a YAML or dict config (see observations.yaml)
    - Handles alias resolution for semantic observation names
    - Provides unified get_observations() for downstream consumers
    """

    def __init__(
        self,
        node: Node,
        ns: Namespace,
        data_sources: Dict[str, DataSource],
        simulation_state_container: SimulationStateContainer = None,
        wait_for_obs: bool = True,
        qos_profile: Optional[QoSProfile] = 10,
        enable_synchronization: bool = True,
        sync_tolerance_seconds: float = 0.1,
        buffer_size: int = 50,
        validate_generators: bool = True,
    ) -> None:
        """
        Initialize the ObservationManager with a dictionary of data sources (collectors/generators).

        Args:
            node (Node): The ROS node used for creating subscribers
            ns (Namespace): Namespace for the ROS topics
            data_sources (Dict[str, DataSource]): Dictionary mapping names to DataSource instances
            simulation_state_container (SimulationStateContainer): Container for simulation state data
            topic_mappings (Optional[Dict[str, Callable]]): Dictionary mapping topic patterns
                to transformation functions
            wait_for_obs (bool): Whether to wait for initial observations before proceeding
            qos_profile (Optional[QoSProfile]): Quality of Service profile for subscribers
            enable_synchronization (bool): Enable temporal synchronization of observations
            sync_tolerance_seconds (float): Tolerance for synchronization timestamps
            buffer_size (int): Size of temporal buffer for each observation stream
        """
        if simulation_state_container is None:
            simulation_state_container = SimulationStateContainer()
            print(
                "No simulation state container provided. Using default empty container. Not recommended for production use."
            )

        self._node = node
        self._logger = node.get_logger()
        self._ns = Namespace(ns)
        self._simulation_state_container = simulation_state_container
        self._data_sources = data_sources
        self._wait_for_obs = wait_for_obs
        self._qos_profile = qos_profile
        self._validate_generators = validate_generators

        # Separate collectors and generators
        self._collectors = {
            name: ds for name, ds in data_sources.items() if isinstance(ds, Collector)
        }
        self._generators = {
            name: ds for name, ds in data_sources.items() if isinstance(ds, Generator)
        }

        # Dependency resolver for generators
        self._dependency_resolver = DependencyResolver(
            self._generators, self._collectors
        )

        # Synchronization parameters
        self._enable_synchronization = enable_synchronization
        self._sync_tolerance = sync_tolerance_seconds
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._message_filter_synchronizer = None
        self._sync_collector_names: List[str] = []

        # Set up ROS subscriptions for collectors
        self._subscribers: Dict[str, Subscription] = {}
        self._setup_collectors()

    def _setup_collectors(self) -> None:
        """Set up ROS2 subscribers for collector data sources."""
        # Determine which collectors need synchronization
        sync_collectors = {}
        if self._enable_synchronization:
            sync_collectors = {
                name: collector
                for name, collector in self._collectors.items()
                if collector.up_to_date_required
            }

        self._sync_collector_names = list(sync_collectors.keys())
        non_sync_collector_names = [
            name for name in self._collectors if name not in self._sync_collector_names
        ]

        # Setup for non-synchronized collectors
        for collector_name in non_sync_collector_names:
            collector = self._collectors[collector_name]
            self._setup_individual_collector(collector_name, collector)

        # Setup for synchronized collectors using message_filters
        if sync_collectors:
            self._setup_synchronized_collectors(sync_collectors)

    def _setup_individual_collector(self, name: str, collector: Collector) -> None:
        """Set up a single collector with its own subscription."""
        # Set QoS profile if the collector doesn't have one
        if collector.qos_profile is None:
            collector.set_qos_profile(self._qos_profile)

        callback = partial(self._observation_callback, collector=collector)

        self._subscribers[name] = self._node.create_subscription(
            collector.message_type,
            collector.topic,
            callback,
            qos_profile=collector.qos_profile,
        )
        self._logger.debug(f"Subscribed to {collector.topic} for collector '{name}'")

    def _setup_synchronized_collectors(
        self, sync_collectors: Dict[str, Collector]
    ) -> None:
        """Set up synchronized collectors using message_filters."""
        subscribers = []

        for name, collector in sync_collectors.items():
            if collector.qos_profile is None:
                collector.set_qos_profile(self._qos_profile)

            subscriber = message_filters.Subscriber(
                self._node,
                collector.message_type,
                collector.topic,
                qos_profile=collector.qos_profile,
            )
            subscribers.append(subscriber)
            self._logger.debug(
                f"Preparing {subscriber.topic} for synchronized collector '{name}'"
            )

        self._message_filter_synchronizer = message_filters.ApproximateTimeSynchronizer(
            subscribers,
            queue_size=self._buffer_size,
            slop=self._sync_tolerance,
        )
        self._message_filter_synchronizer.registerCallback(self._synchronized_callback)
        self._logger.info(f"Using message_filters for: {self._sync_collector_names}")

    def get_observations(self, **extra_observations) -> Dict[str, Any]:
        """
        Collect all observations from collectors and generators.

        Args:
            **extra_observations: Additional observations to include.

        Returns:
            Dict[str, Any]: Complete observation dictionary.
        """
        obs_dict = {}

        # Collect observations from collectors
        obs_dict = asyncio.run(self._get_collectable_observations(obs_dict))

        # Add any extra observations
        obs_dict.update(extra_observations)

        # Generate derived observations
        self._get_generatable_observations(obs_dict)

        return obs_dict

    async def _get_collectable_observations(
        self, obs_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect observations from all collectors, waiting for required updates."""
        self._logger.debug(
            f"Collecting observations from {len(self._collectors)} collectors"
        )

        # Check which collectors need fresh data
        stale_collectors = []
        for name, collector in self._collectors.items():
            if collector.stale and collector.up_to_date_required:
                if self._wait_for_obs:
                    stale_collectors.append(name)
            # Always get the current value (stale or not)
            obs_dict[name] = collector.get_observation()

        # Wait for stale collectors if needed
        if stale_collectors:
            self._logger.debug(f"Waiting for updates from: {stale_collectors}")
            tasks = [self._wait_for_observation(name) for name in stale_collectors]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Update observation values after waiting
            for name in stale_collectors:
                obs_dict[name] = self._collectors[name].get_observation()

        # Mark all collectors as stale for next collection cycle
        self._invalidate_observations()

        return obs_dict

    def _get_generatable_observations(self, obs_dict: Dict[str, Any]) -> None:
        """Generate derived observations from collected data with optional validation."""
        # Optional validation of generator requirements (can be disabled for performance)
        if self._validate_generators:
            validate_generators(obs_dict, self._generators)

        for name in self._dependency_resolver.execution_order:
            try:
                obs_dict[name] = self._generators[name].get_observation(
                    obs_dict,
                    simulation_state_container=self._simulation_state_container,
                )
            except Exception as e:
                self._logger.error(f"Error generating observation '{name}': {e}")
                obs_dict[name] = None

    async def _wait_for_observation(self, collector_name: str) -> bool:
        """Wait for a specific collector to receive new data."""
        if collector_name not in self._collectors:
            self._logger.error(f"Cannot wait for unknown collector '{collector_name}'")
            return False

        collector = self._collectors[collector_name]
        timeout = getattr(collector, "timeout", 0.1)

        try:
            await asyncio.wait_for(self._poll_for_update(collector), timeout=timeout)
            self._logger.debug(f"Collector '{collector_name}' updated successfully")
            return True
        except asyncio.TimeoutError:
            self._logger.warn(
                f"Timeout waiting for '{collector_name}' after {timeout}s"
            )
            return False
        except Exception as e:
            self._logger.error(f"Exception while waiting for '{collector_name}': {e}")
            return False

    async def _poll_for_update(self, collector: Collector) -> None:
        """Continuously poll for a collector update."""
        while collector.stale:
            await asyncio.sleep(0.001)

    def _invalidate_observations(self) -> None:
        """Mark all collectors as stale to ensure fresh data on next collection."""
        for collector in self._collectors.values():
            collector.stale = True

    def _observation_callback(self, msg: Any, collector: Collector) -> None:
        """Process incoming messages for individual collectors."""
        try:
            with self._lock:
                collector.update(msg)
        except Exception as e:
            self._logger.error(f"Error updating collector '{collector.name}': {e}")

    def _synchronized_callback(self, *msgs: Any) -> None:
        """Callback for synchronized messages from message_filters."""
        with self._lock:
            for collector_name, msg in zip(self._sync_collector_names, msgs):
                try:
                    collector = self._collectors[collector_name]
                    collector.update(msg)
                except Exception as e:
                    self._logger.error(
                        f"Error updating synchronized collector '{collector_name}': {e}"
                    )

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all collectors."""
        health_status = {}
        for name, collector in self._collectors.items():
            health_status[name] = {
                "update_count": collector.update_count,
                "error_count": collector.error_count,
                "is_stale": collector.stale,
                "age_seconds": collector.age,
                "has_data": collector.get_observation() is not None,
            }
        return health_status

    def shutdown(self) -> None:
        """Clean up subscriptions and resources."""
        for name, subscription in self._subscribers.items():
            self._node.destroy_subscription(subscription)
        self._logger.info("ObservationManager shutdown complete")

    @property
    def collectors(self) -> List[str]:
        """Returns the names of all collectors."""
        return list(self._collectors.keys())

    @property
    def generators(self) -> List[str]:
        """Returns the names of all generators."""
        return list(self._generators.keys())
