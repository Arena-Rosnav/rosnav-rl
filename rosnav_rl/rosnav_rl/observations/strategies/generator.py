"""
Generator Strategy for ROS2 observation generators.

Handles the execution and validation of generators with proper dependency ordering.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict

from rclpy.node import Node

from rosnav_rl.utils.logging import ComponentType, ErrorSeverity, collect_error
from rosnav_rl.utils.validation import GeneratorSchemaValidator

if TYPE_CHECKING:
    from rosnav_rl.cfg.parameters import AgentParameters

from ..data_sources.base import Generator
from ..factory.resolver import DependencyResolver


class GeneratorManager:
    """Handles generator execution with dependency resolution and validation."""

    def __init__(
        self,
        node: Node,
        dependency_resolver: DependencyResolver,
        simulation_state_container: AgentParameters,
        validate_generators: bool = True,
    ):
        """
        Initialize the generator strategy.

        Args:
            node: ROS2 node for logging
            dependency_resolver: Resolver for generator execution order
            simulation_state_container: Container for simulation state data
            validate_generators: Whether to enable generator validation
        """
        self._node = node
        self._logger = node.get_logger()
        self._dependency_resolver = dependency_resolver
        self._simulation_state_container = simulation_state_container
        self._validate_generators = validate_generators
        self._generator_failure_counts: Dict[str, int] = defaultdict(int)

    @property
    def generator_failure_counts(self) -> Dict[str, int]:
        """Per-generator count of get_observation() failures (obs_dict[name] set to None)."""
        return dict(self._generator_failure_counts)

    def generate_observations(
        self, generators: Dict[str, Generator], obs_dict: Dict[str, Any]
    ) -> None:
        """
        Generate derived observations from collected data with optional validation.

        Args:
            generators: Dictionary of generators to execute
            obs_dict: Observation dictionary to update with generated data
        """
        if not generators:
            return

        # Validate root generators only once, then disable (keys are static)
        if self._validate_generators:
            self._validate_root_generators(obs_dict, generators)
            self._validate_generators = False

        # Execute generators in dependency order
        sim_state = self._simulation_state_container
        for name in self._dependency_resolver.execution_order:
            try:
                generator = generators[name]
                obs_dict[name] = generator.get_observation(
                    obs_dict, simulation_state_container=sim_state
                )
            except Exception as e:
                self._generator_failure_counts[name] += 1
                collect_error(
                    component_type=ComponentType.GENERATOR,
                    component_name=name,
                    severity=ErrorSeverity.ERROR,
                    message=(
                        f"Error generating observation '{name}': {e}. "
                        f"obs_dict['{name}'] set to None."
                    ),
                    error_type=type(e).__name__,
                )
                obs_dict[name] = None

    def _validate_root_generators(
        self, obs_dict: Dict[str, Any], generators: Dict[str, Generator]
    ) -> None:
        """Validate root generators that only depend on collectors."""
        try:
            GeneratorSchemaValidator.validate_root_generators(obs_dict, generators)
        except Exception as e:
            self._logger.error(f"Root generator validation failed: {e}")
            # Continue execution - validation is advisory

    def _execute_generator(
        self, name: str, generator: Generator, obs_dict: Dict[str, Any]
    ) -> None:
        """
        Execute a single generator with proper error handling.
        Note: Primarily used for one-off calls. The hot path in generate_observations()
        is inlined above for performance.

        Args:
            name: Name of the generator
            generator: Generator instance to execute
            obs_dict: Observation dictionary to update
        """
        try:
            obs_dict[name] = generator.get_observation(
                obs_dict,
                simulation_state_container=self._simulation_state_container,
            )
        except Exception as e:
            self._generator_failure_counts[name] += 1
            collect_error(
                component_type=ComponentType.GENERATOR,
                component_name=name,
                severity=ErrorSeverity.ERROR,
                message=(
                    f"Error generating observation '{name}': {e}. "
                    f"obs_dict['{name}'] set to None."
                ),
                error_type=type(e).__name__,
            )
            obs_dict[name] = None  # Graceful degradation
