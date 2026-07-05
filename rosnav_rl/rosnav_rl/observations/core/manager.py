from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Type, Union

from rclpy.node import Node
from rclpy.qos import QoSProfile

from rosnav_rl.utils.validation import GeneratorSchemaValidator
from rosnav_rl.utils.rostopic import Namespace
from ..data_sources.base import Collector, DataSource, Generator
from ..factory.resolver import DependencyResolver
from ..strategies.collector import CollectorManager
from ..strategies.generator import GeneratorManager
from ..strategies.subscription import SubscriptionManager
from .pipeline import ObservationPipeline

if TYPE_CHECKING:
    from rosnav_rl.cfg.parameters import AgentParameters

T = TypeVar("T")


def _import_observation_factory():
    """Lazy import to avoid circular dependencies."""
    from ..factory.factory import ObservationFactory

    return ObservationFactory


class ObservationManager:
    """
    Central manager for observation collection and generation in ROS2-based RL agents.

    The ObservationManager orchestrates the complete observation pipeline:
    1. **Data Collection**: Subscribes to ROS topics via Collectors
    2. **Feature Generation**: Computes derived features via Generators
    3. **Temporal Synchronization**: Aligns messages across topics (optional)
    4. **Dependency Resolution**: Automatically resolves generator dependencies
    5. **Health Monitoring**: Tracks data freshness and error rates

    Architecture:
        ```
        ROS Topics → Collectors → [Sync] → ObservationManager
                         ↓
                    Generators (computed on-demand)
                         ↓
                    get_observations() → RL Agent
        ```

    Key Features:
        - **Factory Pattern**: Create from YAML config via `from_config()`
        - **Alias System**: Use semantic names (e.g., 'robot_pose') instead of specific sources
        - **Lazy Evaluation**: Generators compute only requested observations
        - **Buffer Management**: Efficient memory reuse in collectors and generators
        - **Message Filters**: Optional temporal synchronization for multiple topics
        - **Validation**: Optional dependency checking and schema validation

    Attributes:
        observation_pipeline (Type[ObservationPipeline]): Pipeline strategy class

    Configuration:
        See `from_config()` for detailed configuration documentation and examples.

    Thread Safety:
        - Collector updates use locks for thread-safe message handling
        - Generator computations are stateless (safe for parallel access)
        - Call `shutdown()` before destroying to clean up ROS resources

    Performance:
        - Collectors cache latest message (O(1) access)
        - Generators use pre-allocated buffers where possible
        - Dependency graph is resolved once at initialization
        - Vectorized operations in numeric generators

    Example:
        ```python
        import yaml
        from rosnav_rl.observations.core.manager import ObservationManager

        # Load configuration
        with open('observations.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Create manager from config
        obs_manager = ObservationManager.from_config(
            config=config,
            node=ros_node,
            ns='jackal',
            simulation_state_container=sim_state
        )

        # Main loop
        while running:
            # Get all configured observations
            obs = obs_manager.get_observations()

            # Access by semantic name (from aliases)
            robot_pose = obs['robot_pose']
            laser_scan = obs['front_laser']

            # Or by original data source name
            ped_locations = obs['arena_pedestrian_relative_locations']

            # Feed to RL agent
            action = agent.act(obs)

        # Clean up
        obs_manager.shutdown()
        ```

    See Also:
        - ObservationFactory: Creates data sources from configuration
        - DependencyResolver: Resolves generator dependency graphs
        - SubscriptionManager: Manages ROS subscriptions and synchronization
    """

    observation_pipeline: Type[ObservationPipeline] = ObservationPipeline

    def __init__(
        self,
        node: Node,
        ns: Union[str, Namespace],
        data_sources: Dict[str, DataSource],
        simulation_state_container: AgentParameters = None,
        wait_for_obs: bool = True,
        qos_profile: Optional[QoSProfile] = 10,
        enable_synchronization: bool = True,
        sync_tolerance_seconds: float = 0.1,
        buffer_size: int = 50,
        validate_generators: bool = True,
        allow_default_params: bool = False,
    ) -> None:
        """
        Initialize the ObservationManager with a dictionary of data sources (collectors/generators).

        Args:
            node (Node): The ROS node used for creating subscribers
            ns (Namespace): Namespace for the ROS topics
            data_sources (Dict[str, DataSource]): Dictionary mapping names to DataSource instances
            simulation_state_container (AgentParameters): Container for simulation state data
            wait_for_obs (bool): Whether to wait for initial observations before proceeding
            qos_profile (Optional[QoSProfile]): Quality of Service profile for subscribers
            enable_synchronization (bool): Enable temporal synchronization of observations
            sync_tolerance_seconds (float): Tolerance for synchronization timestamps
            buffer_size (int): Size of temporal buffer for each observation stream
            validate_generators (bool): Enable dependency-aware generator validation.
                - At init: Validates all generator dependencies exist in configuration
                - At runtime: Validates root generators upfront, other generators just-in-time
            allow_default_params (bool): If True, permits falling back to a default
                AgentParameters() when simulation_state_container is None (with a
                warning). Off by default: real robot params (e.g. safe distance)
                silently substituted with defaults is a safety hazard, not a
                convenience — callers that genuinely want a placeholder must opt in.

        Raises:
            ValueError: If simulation_state_container is None and allow_default_params
                is False.
        """
        if simulation_state_container is None:
            if not allow_default_params:
                raise ValueError(
                    "ObservationManager requires a simulation_state_container "
                    "(AgentParameters) — pass one explicitly, or pass "
                    "allow_default_params=True to fall back to defaults "
                    "(not recommended: real robot params such as safe distance "
                    "would silently be replaced with placeholder values)."
                )
            from rosnav_rl.cfg.parameters import AgentParameters  # noqa: PLC0415
            simulation_state_container = AgentParameters()
            logging.getLogger(__name__).warning(
                "No AgentParameters provided to ObservationManager. "
                "Using default values — not recommended for production use."
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

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        node: Node,
        ns: Union[str, Namespace],
        simulation_state_container: AgentParameters = None,
        **manager_kwargs,
    ) -> "ObservationManager":
        """
        Create an ObservationManager from a configuration dictionary.

        This factory method provides a convenient way to instantiate an ObservationManager
        from a YAML configuration file or dictionary. It automatically:
        - Creates all collectors and generators defined in the config
        - Resolves aliases to data source names
        - Sets up ROS subscriptions with appropriate QoS profiles
        - Validates generator dependencies
        - Configures synchronization if enabled

        Args:
            config (Dict[str, Any]): Configuration dictionary with the following structure:
                - 'aliases': Optional dict mapping semantic names to data source names
                - 'datasources': Dict of collector and generator configurations
                  Each datasource should have:
                    - 'type': Class name (e.g., 'LaserScanCollector', 'RobotPoseTFGenerator')
                    - 'params': Dict of parameters including 'topic' for collectors
            node (Node): ROS2 node instance for creating subscriptions and logging
            ns (Namespace): Namespace prefix for ROS topics (e.g., 'jackal', 'robot_1')
            simulation_state_container (SimulationStateContainer, optional): Container holding
                simulation state (robot config, environment params). If None, uses empty default.
            **manager_kwargs: Additional keyword arguments passed to ObservationManager:
                - wait_for_obs (bool): Wait for initial data before proceeding (default: True)
                - qos_profile (QoSProfile): ROS2 QoS settings (default: 10)
                - enable_synchronization (bool): Enable temporal sync (default: True)
                - sync_tolerance_seconds (float): Timestamp tolerance in seconds (default: 0.1)
                - buffer_size (int): Message buffer size for sync (default: 50)
                - validate_generators (bool): Validate dependencies (default: True)

        Returns:
            ObservationManager: Fully configured and ready-to-use observation manager instance
                with active ROS subscriptions for all collectors

        Raises:
            ValueError: If configuration is malformed or data source types are unknown
            KeyError: If required configuration keys are missing

        Example:
            Basic usage with YAML config file:
            ```python
            import yaml
            from rosnav_rl.observations.core.manager import ObservationManager

            # Load configuration
            with open('observations.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Create manager
            obs_manager = ObservationManager.from_config(
                config=config,
                node=my_ros_node,
                ns='jackal',
                simulation_state_container=sim_state
            )

            # Get observations
            observations = obs_manager.get_observations()
            ```

            Advanced usage with custom settings:
            ```python
            obs_manager = ObservationManager.from_config(
                config=config,
                node=my_ros_node,
                ns='robot_1',
                simulation_state_container=sim_state,
                wait_for_obs=True,              # Wait for first messages
                enable_synchronization=True,     # Enable message_filters sync
                sync_tolerance_seconds=0.05,     # 50ms tolerance
                buffer_size=100,                 # Larger buffer
                validate_generators=True         # Check dependencies at init
            )
            ```

            Minimal configuration structure:
            ```yaml
            aliases:
              robot_pose: robot_pose_from_odom
              people_data: arena_pedestrian_detections

            datasources:
              robot_pose_from_odom:
                type: OdometryCollector
                params:
                  topic: "odom"
                  up_to_date_required: true

              arena_pedestrian_detections:
                type: ArenaPedestrianCollector
                params:
                  topic: "/task_generator_node/arena_peds"
                  up_to_date_required: true

              arena_pedestrian_relative_locations:
                type: ArenaPedestrianRelativeLocationGenerator
            ```

        Note:
            - Collectors subscribe to ROS topics and cache the latest messages
            - Generators compute derived features on-demand from collector data
            - All topic names are automatically prefixed with the namespace
            - Synchronization uses message_filters.ApproximateTimeSynchronizer
            - For synchronized collectors, set 'up_to_date_required: true' in config
        """
        ObservationFactory = _import_observation_factory()
        factory = ObservationFactory()

        # Create data sources with common kwargs
        common_kwargs = {
            "node": node,
            "ns": ns,
            "simulation_state_container": simulation_state_container,
        }

        data_sources = factory.create_data_sources(config, **common_kwargs)

        return cls(
            node=node,
            ns=ns,
            data_sources=data_sources,
            simulation_state_container=simulation_state_container,
            **manager_kwargs,
        )

    def _setup_collectors(self) -> None:
        """Set up ROS2 subscribers for collector data sources."""

        def _observation_callback(msg: Any, collector: Collector) -> None:
            """Process incoming messages for individual collectors."""
            try:
                with self._subscription_manager.lock:
                    collector.update(msg)
                    self._logger.debug(
                        f"✓ Collector '{collector.name}' updated successfully "
                        f"(count={collector.update_count})"
                    )
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
                        self._logger.debug(
                            f"✓ Synchronized collector '{collector.name}' updated successfully "
                            f"(count={collector.update_count})"
                        )
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
