# Main public API
from .core import ObservationManager, ObservationPipeline

# Legacy imports for backward compatibility
from .factory import DependencyResolver, DependencyMissingError
from .strategies import (
    CollectorManager,
    GeneratorManager,
    SubscriptionManager,
    WaitingStrategy,
)
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
