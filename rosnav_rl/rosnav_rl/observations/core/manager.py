from __future__ import annotations

from typing import Any, Dict, List, Optional, TypeVar, Type

from rclpy.node import Node
from rclpy.qos import QoSProfile

from rosnav_rl.spaces.observation_space.utils import GeneratorSchemaValidator
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.rostopic import Namespace
from ..data_sources.base import Collector, DataSource, Generator
from ..factory.resolver import DependencyResolver
from ..strategies.collector import CollectorManager
from ..strategies.generator import GeneratorManager
from ..strategies.subscription import SubscriptionManager
from .pipeline import ObservationPipeline

T = TypeVar("T")


class ObservationManager:
    """
    Manages observation collection and generation for RL agents in ROS2 using the new Collector/Generator system.

    - Instantiates collectors and generators from a YAML or dict config (see observations.yaml)
    - Handles alias resolution for semantic observation names
    - Provides unified get_observations() for downstream consumers
    """

    observation_pipeline: Type[ObservationPipeline] = ObservationPipeline

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
            wait_for_obs (bool): Whether to wait for initial observations before proceeding
            qos_profile (Optional[QoSProfile]): Quality of Service profile for subscribers
            enable_synchronization (bool): Enable temporal synchronization of observations
            sync_tolerance_seconds (float): Tolerance for synchronization timestamps
            buffer_size (int): Size of temporal buffer for each observation stream
            validate_generators (bool): Enable dependency-aware generator validation.
                - At init: Validates all generator dependencies exist in configuration
                - At runtime: Validates root generators upfront, other generators just-in-time
        """
        if simulation_state_container is None:
            simulation_state_container = SimulationStateContainer()
            print(
                "No simulation state container provided. Using default empty container. "
                "Not recommended for production use."
            )

        self._node = node
        self._logger = node.get_logger()
        self._ns = Namespace(ns)
        self._simulation_state_container = simulation_state_container
        self._data_sources = data_sources
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

        # Validate generator dependencies during initialization
        if self._validate_generators:
            GeneratorSchemaValidator.validate_configuration_dependencies(
                self._generators, self._collectors
            )

        # Initialize strategy components
        self._subscription_manager = SubscriptionManager(
            node=node,
            ns=self._ns,
            enable_synchronization=enable_synchronization,
            sync_tolerance_seconds=sync_tolerance_seconds,
            buffer_size=buffer_size,
            qos_profile=qos_profile,
        )

        collection_strategy = CollectorManager(
            node=node,
            wait_for_obs=wait_for_obs,
        )

        generator_strategy = GeneratorManager(
            node=node,
            dependency_resolver=self._dependency_resolver,
            simulation_state_container=simulation_state_container,
            validate_generators=validate_generators,
        )

        # Create observation pipeline
        self._pipeline = self.observation_pipeline(
            collection_strategy=collection_strategy,
            generator_strategy=generator_strategy,
        )

        # Set up ROS subscriptions for collectors
        self._setup_collectors()

    def _setup_collectors(self) -> None:
        """Set up ROS2 subscribers for collector data sources."""

        def _observation_callback(msg: Any, collector: Collector) -> None:
            """Process incoming messages for individual collectors."""
            try:
                with self._subscription_manager.lock:
                    collector.update(msg)
            except Exception as e:
                self._logger.error(f"Error updating collector '{collector.name}': {e}")

        def _synchronized_callback(*msgs: Any) -> None:
            """Callback for synchronized messages from message_filters."""
            with self._subscription_manager.lock:
                for collector_name, msg in zip(
                    self._subscription_manager.sync_collector_names, msgs
                ):
                    try:
                        collector = self._collectors[collector_name]
                        collector.update(msg)
                    except Exception as e:
                        self._logger.error(
                            f"Error updating synchronized collector '{collector_name}': {e}"
                        )

        self._subscription_manager.setup_collectors(
            self._collectors,
            _observation_callback,
            _synchronized_callback,
        )

    def get_observations(self, **extra_observations) -> Dict[str, Any]:
        """
        Collect all observations from collectors and generators using the elegant pipeline.

        Args:
            **extra_observations: Additional observations to include.

        Returns:
            Dict[str, Any]: Complete observation dictionary.
        """
        return self._pipeline.forward(
            collectors=self._collectors,
            generators=self._generators,
            extra_observations=extra_observations,
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
        self._subscription_manager.shutdown()
        self._logger.info("ObservationManager shutdown complete")

    @property
    def collectors(self) -> List[str]:
        """Returns the names of all collectors."""
        return list(self._collectors.keys())

    @property
    def generators(self) -> List[str]:
        """Returns the names of all generators."""
        return list(self._generators.keys())

    def get_dependency_info(self) -> Dict[str, Any]:
        """Get debugging information about generator dependencies."""
        return self._dependency_resolver.get_dependency_info()
