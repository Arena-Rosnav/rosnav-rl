"""
Factory functions for creating observation data sources from configuration.

This module provides utilities to create collectors and generators from
YAML configuration files or dictionaries, bridging the gap between
configuration and instantiated data sources.
"""

from typing import Dict, Any, Type, List
from ..data_sources.base import DataSource, Collector, Generator


def _ros_msg_type_string(msg_cls) -> str:
    """Derive 'package/MsgClass' string from a ROS message class.

    Examples
    --------
    sensor_msgs.msg.LaserScan  →  'sensor_msgs/LaserScan'
    geometry_msgs.msg.Twist    →  'geometry_msgs/Twist'
    """
    package = msg_cls.__module__.split(".")[0]
    return f"{package}/{msg_cls.__name__}"


class ObservationFactory:
    """Factory for creating observation data sources from configuration.

    Collector type resolution supports two formats:

    1. **Class name** (legacy, still fully supported)::

           type: LaserScanCollector

    2. **ROS message type** (preferred, more readable)::

           type: sensor_msgs/LaserScan

       The factory automatically maps the message type to the unique
       collector that handles it.  If multiple collectors handle the same
       message type, the mapping is ambiguous and a ``ValueError`` that
       lists the conflicting class names is raised; use the class name to
       resolve the ambiguity.
    """

    def __init__(self):
        # Registry keyed by class name  →  Type[Collector / Generator]
        self._collectors: Dict[str, Type[Collector]] = {}
        self._generators: Dict[str, Type[Generator]] = {}
        # Registry keyed by 'pkg/Msg'  →  Type[Collector]
        self._msg_type_to_collector: Dict[str, Type[Collector]] = {}
        # Track conflicts: message types claimed by more than one collector
        self._msg_type_conflicts: Dict[str, List[str]] = {}

        # Auto-register all available classes
        self._register_collectors()
        self._register_generators()

    def _register_collectors(self):
        """Register all available collector classes."""
        import inspect
        from ..data_sources import collectors

        for name, obj in inspect.getmembers(collectors):
            if not (inspect.isclass(obj) and issubclass(obj, Collector) and obj is not Collector):
                continue
            self._collectors[name] = obj

            # Build the ROS-message-type index if message_type is available
            msg_cls = getattr(obj, "message_type", None)
            if msg_cls is None:
                continue
            key = _ros_msg_type_string(msg_cls)
            if key in self._msg_type_to_collector:
                # Record the conflict but don't clobber the existing entry
                self._msg_type_conflicts.setdefault(key, [self._msg_type_to_collector[key].__name__])
                self._msg_type_conflicts[key].append(name)
            else:
                self._msg_type_to_collector[key] = obj

    def _register_generators(self):
        """Register all available generator classes."""
        import inspect
        from ..data_sources import generators

        for name, obj in inspect.getmembers(generators):
            if inspect.isclass(obj) and issubclass(obj, Generator) and obj is not Generator:
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

        # Get the set of datasource names that have aliases
        aliased_datasources = set(aliases.values())

        # Create all data sources first
        for name, ds_config in datasources_config.items():
            data_source = self._create_data_source(name, ds_config, **kwargs)
            if data_source:
                # Only store with original name if it's not aliased
                if name not in aliased_datasources:
                    data_sources[name] = data_source

                # Store with alias name if it has one
                for alias, target in aliases.items():
                    if target == name:
                        data_sources[alias] = data_source

        return data_sources

    def _create_data_source(
        self, name: str, config: Dict[str, Any], **kwargs
    ) -> DataSource:
        """Create a single data source from configuration.

        The ``type`` field accepts either:

        * A **collector class name** (``LaserScanCollector``) or
          **generator class name** (``GoalLocationInRobotFrameGenerator``).
        * A **ROS message type** string (``sensor_msgs/LaserScan``) — the
          factory resolves the unique collector that handles that message type.
        """
        ds_type = config.get("type")
        if not ds_type:
            raise ValueError(f"Data source '{name}' missing 'type' field")

        params = config.get("params", {})
        merged_kwargs = {**kwargs, **params}

        # --- Resolve collector class -----------------------------------------
        collector_class = None

        if "/" in ds_type:
            # ROS message type format, e.g. 'sensor_msgs/LaserScan'
            if ds_type in self._msg_type_conflicts:
                names = self._msg_type_conflicts[ds_type]
                raise ValueError(
                    f"Data source '{name}': message type '{ds_type}' is handled by "
                    f"multiple collectors: {names}. "
                    f"Use the class name directly to resolve the ambiguity."
                )
            if ds_type not in self._msg_type_to_collector:
                available = sorted(self._msg_type_to_collector.keys())
                raise ValueError(
                    f"Data source '{name}': no collector found for message type "
                    f"'{ds_type}'. Available message types: {available}"
                )
            collector_class = self._msg_type_to_collector[ds_type]
        elif ds_type in self._collectors:
            collector_class = self._collectors[ds_type]

        if collector_class is not None:
            topic = merged_kwargs.pop("topic", name)
            return collector_class(name=name, topic=topic, **merged_kwargs)

        # --- Resolve generator class -----------------------------------------
        if ds_type in self._generators:
            generator_class = self._generators[ds_type]
            return generator_class(name=name, **merged_kwargs)

        raise ValueError(
            f"Unknown data source type: '{ds_type}'. "
            f"Use a collector class name, a generator class name, "
            f"or a ROS message type (e.g. 'sensor_msgs/LaserScan')."
        )

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

    .. deprecated::
        Use `ObservationManager.from_config()` instead. This function is kept
        for backward compatibility but will be removed in a future version.

    Args:
        config: Configuration dictionary with observation pipeline definition
        node: ROS node
        ns: Namespace
        simulation_state_container: Simulation state container
        **manager_kwargs: Additional arguments for ObservationManager

    Returns:
        Configured ObservationManager instance
    """
    from ..core.manager import ObservationManager

    return ObservationManager.from_config(
        config=config,
        node=node,
        ns=ns,
        simulation_state_container=simulation_state_container,
        **manager_kwargs,
    )
