"""
Factory functions for creating observation data sources from configuration.

This module provides utilities to create collectors and generators from
YAML configuration files or dictionaries, bridging the gap between
configuration and instantiated data sources.
"""

from typing import Dict, Any, Type
from .base import DataSource, Collector, Generator


class ObservationFactory:
    """Factory for creating observation data sources from configuration."""

    def __init__(self):
        # Registry of available collector and generator classes
        self._collectors: Dict[str, Type[Collector]] = {}
        self._generators: Dict[str, Type[Generator]] = {}

        # Auto-register all available classes
        self._register_collectors()
        self._register_generators()

    def _register_collectors(self):
        """Register all available collector classes."""
        # Get all collector classes from the collectors module
        import inspect
        from . import collectors

        for name, obj in inspect.getmembers(collectors):
            if inspect.isclass(obj) and issubclass(obj, Collector) and obj != Collector:
                self._collectors[name] = obj

    def _register_generators(self):
        """Register all available generator classes."""
        # Get all generator classes from the generators module
        import inspect
        from . import generators

        for name, obj in inspect.getmembers(generators):
            if inspect.isclass(obj) and issubclass(obj, Generator) and obj != Generator:
                self._generators[name] = obj

    def create_data_sources(
        self, config: Dict[str, Any], **kwargs
    ) -> Dict[str, DataSource]:
        """
        Create data sources from configuration.

        Args:
            config: Configuration dictionary with 'datasources' section
            **kwargs: Additional arguments to pass to data source constructors

        Returns:
            Dictionary mapping names to DataSource instances
        """
        data_sources = {}

        # Handle aliases - create mappings from alias to actual data source
        aliases = config.get("aliases", {})
        datasources_config = config.get("datasources", {})

        # Create all data sources first
        for name, ds_config in datasources_config.items():
            data_source = self._create_data_source(name, ds_config, **kwargs)
            if data_source:
                data_sources[name] = data_source

        # Apply aliases by copying references
        for alias, target in aliases.items():
            if target in data_sources:
                data_sources[alias] = data_sources[target]

        return data_sources

    def _create_data_source(
        self, name: str, config: Dict[str, Any], **kwargs
    ) -> DataSource:
        """Create a single data source from configuration."""
        ds_type = config.get("type")
        if not ds_type:
            raise ValueError(f"Data source '{name}' missing 'type' field")

        params = config.get("params", {})

        # Merge kwargs with params, with params taking precedence
        merged_kwargs = {**kwargs, **params}

        # Try to create collector first
        if ds_type in self._collectors:
            collector_class = self._collectors[ds_type]

            # Extract topic from config or use name as default
            topic = params.get("topic", name)

            return collector_class(name=name, topic=topic, **merged_kwargs)

        # Try to create generator
        elif ds_type in self._generators:
            generator_class = self._generators[ds_type]

            # Generators need inputs mapping
            inputs = params.get("inputs", {})

            return generator_class(name=name, inputs=inputs, **merged_kwargs)

        else:
            raise ValueError(f"Unknown data source type: {ds_type}")

    def list_available_types(self) -> Dict[str, Dict[str, Type]]:
        """List all available collector and generator types."""
        return {
            "collectors": self._collectors.copy(),
            "generators": self._generators.copy(),
        }


def create_observation_manager_from_config(
    config: Dict[str, Any], node, ns, simulation_state_container=None, **manager_kwargs
):
    """
    Convenience function to create an ObservationManager from configuration.

    Args:
        config: Configuration dictionary with observation pipeline definition
        node: ROS node
        ns: Namespace
        simulation_state_container: Simulation state container
        **manager_kwargs: Additional arguments for ObservationManager

    Returns:
        Configured ObservationManager instance
    """
    from .observation_manager import ObservationManager

    factory = ObservationFactory()

    # Create data sources with common kwargs
    common_kwargs = {
        "node": node,
        "ns": ns,
        "simulation_state_container": simulation_state_container,
    }

    data_sources = factory.create_data_sources(config, **common_kwargs)

    return ObservationManager(
        node=node,
        ns=ns,
        data_sources=data_sources,
        simulation_state_container=simulation_state_container,
        **manager_kwargs,
    )
