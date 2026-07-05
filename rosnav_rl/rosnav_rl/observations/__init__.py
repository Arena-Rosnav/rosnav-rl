from .utils.static import *

__all__ = [
    "ObservationManager",
    "ObservationPipeline",
    "DependencyResolver",
    "DependencyMissingError",
    "CollectorManager",
    "GeneratorManager",
    "SubscriptionManager",
    "WaitingStrategy",
]

# .core / .factory / .strategies are rclpy-coupled (subscribers, collectors).
# Deferred so importing e.g. rosnav_rl.observations.utils.types (a compat
# shim for rosnav_rl.utils.observation_types, used by ROS-free spaces/
# reward code) doesn't force this package to require ROS.
_LAZY = {
    "ObservationManager": ".core",
    "ObservationPipeline": ".core",
    "DependencyResolver": ".factory",
    "DependencyMissingError": ".factory",
    "CollectorManager": ".strategies",
    "GeneratorManager": ".strategies",
    "SubscriptionManager": ".strategies",
    "WaitingStrategy": ".strategies",
}


def __getattr__(name: str):
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(submodule, __name__), name)
