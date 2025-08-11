"""
Observation Pipeline for coordinating the complete observation collection and generation process.

Provides a clean, composable interface for the observation workflow.
"""

from __future__ import annotations

from typing import Any, Dict

from ..data_sources.base import Collector, Generator
from ..strategies.collector import CollectorManager
from ..strategies.generator import GeneratorManager


class ObservationPipeline:
    """Elegant pipeline for coordinating observation collection and generation."""

    def __init__(
        self,
        collection_strategy: CollectorManager,
        generator_strategy: GeneratorManager,
    ):
        """
        Initialize the observation pipeline.

        Args:
            collection_strategy: Strategy for collecting observations from collectors
            generator_strategy: Strategy for generating derived observations
        """
        self._collection_strategy = collection_strategy
        self._generator_strategy = generator_strategy

    def forward(
        self,
        collectors: Dict[str, Collector],
        generators: Dict[str, Generator],
        extra_observations: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete observation pipeline.

        Args:
            collectors: Dictionary of collectors to collect from
            generators: Dictionary of generators to execute
            extra_observations: Additional observations to include

        Returns:
            Complete observation dictionary
        """
        # Initialize observation dictionary
        obs_dict = {}

        # Stage 1: Collect observations from collectors
        obs_dict = self._collection_strategy.collect_observations(collectors, obs_dict)

        # Stage 2: Add any extra observations
        if extra_observations:
            obs_dict.update(extra_observations)

        # Stage 3: Generate derived observations
        self._generator_strategy.generate_observations(generators, obs_dict)

        return obs_dict
