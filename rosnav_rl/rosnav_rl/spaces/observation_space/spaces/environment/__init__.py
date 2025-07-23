"""Environment # Basic spaces
from .basic_environment_spaces import (
    PedestrianVelXSpace,
    PedestrianVelYSpace,
    StackedLaserMapSpace
)

# Advanced spaces
from .advanced_environment_spaces import (
    EnvironmentContextSpace,
    SpatialAwarenessSpace,
    ObstacleProximitySpace
)uction Ready & Legacy

This module contains both production-ready and legacy environment observation spaces.

Production Spaces:
    - EnvironmentContextSpace: Environment context from laser data
    - SpatialAwarenessSpace: Spatial awareness with corridors
    - ObstacleProximitySpace: Obstacle proximity zones

Legacy Spaces:
    - PedestrianVelXSpace: Pedestrian X velocity feature map
    - PedestrianVelYSpace: Pedestrian Y velocity feature map
    - StackedLaserMapSpace: Stacked laser feature map
"""

# Legacy spaces
from .legacy_feature_maps import (
    PedestrianVelXSpace,
    PedestrianVelYSpace,
    StackedLaserMapSpace,
)

# Production ready spaces
from .robust_environment_space import (
    EnvironmentContextSpace,
    SpatialAwarenessSpace,
    ObstacleProximitySpace,
)

__all__ = [
    "PedestrianVelXSpace",
    "PedestrianVelYSpace",
    "StackedLaserMapSpace",
    "EnvironmentContextSpace",
    "SpatialAwarenessSpace",
    "ObstacleProximitySpace",
]
