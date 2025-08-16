"""
Generator Strategy for ROS2 observation generators.

Handles the execution and validation of generators with proper dependency ordering.
"""

from __future__ import annotations

from typing import Any, Dict

from rclpy.node import Node

from rosnav_rl.utils.validation import GeneratorSchemaValidator
from rosnav_rl.states import SimulationStateContainer

from ..data_sources.base import Generator
from ..factory.resolver import DependencyResolver


class GeneratorManager:
    """Handles generator execution with dependency resolution and validation."""

    def __init__(
        self,
        node: Node,
        dependency_resolver: DependencyResolver,
        simulation_state_container: SimulationStateContainer,
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

        # Optional validation of root generators (those that only depend on collectors)
        if self._validate_generators:
            self._validate_root_generators(obs_dict, generators)

        # Execute generators in dependency order
        for name in self._dependency_resolver.execution_order:
            self._execute_generator(name, generators[name], obs_dict)

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

        Args:
            name: Name of the generator
            generator: Generator instance to execute
            obs_dict: Observation dictionary to update
        """
        try:
            # Optional per-generator validation (dependency-aware)
            if self._validate_generators:
                GeneratorSchemaValidator.validate_single_generator(
                    obs_dict, name, generator
                )

            # Generate the observation
            obs_dict[name] = generator.get_observation(
                obs_dict,
                simulation_state_container=self._simulation_state_container,
            )

            self._logger.debug(f"Successfully generated observation '{name}'")
        except Exception as e:
            self._logger.error(f"Error generating observation '{name}': {e}")
            obs_dict[name] = None  # Graceful degradation
