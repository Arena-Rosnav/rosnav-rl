"""Environment Spaces - Production Ready & Legacy

This module contains both production-ready and legacy environment observation spaces.

Production Spaces:
    - EnvironmentContextSpace: Environment context from laser data
    - SpatialAwarenessSpace: Spatial awareness with corridors
    - ObstacleProximitySpace: Obstacle proximity zones

Feature Map Spaces:
    - PedestrianVelXSpace: Pedestrian X velocity feature map
    - PedestrianVelYSpace: Pedestrian Y velocity feature map
    - StackedLaserMapSpace: Stacked laser feature map
    - PedestrianLocationSpace: Pedestrian location feature map
    - PedestrianSocialStateSpace: Pedestrian social state feature map
    - PedestrianTypeSpace: Pedestrian type feature map
"""

# Feature map spaces
from .feature_map_spaces import (
    PedestrianVelXSpace,
    PedestrianVelYSpace,
    StackedLaserMapSpace,
    LaserCartesianMapSpace,
    PedestrianLocationSpace,
    PedestrianSocialStateSpace,
    PedestrianTypeSpace,
)

# Advanced environment spaces
from .advanced_environment_spaces import (
    EnvironmentContextSpace,
    SpatialAwarenessSpace,
    ObstacleProximitySpace,
)

# Graph-based pedestrian spaces (Social-Dreamer)
from .graph_spaces import (
    PedestrianNodeSetSpace,
    PedestrianMaskSpace,
)

# World-frame robot pose for SE(2) frame canonicalization (Social-Dreamer M5.2+)
from .robot_pose_space import RobotPoseSpace

__all__ = [
    "PedestrianVelXSpace",
    "PedestrianVelYSpace",
    "StackedLaserMapSpace",
    "LaserCartesianMapSpace",
    "PedestrianLocationSpace",
    "PedestrianSocialStateSpace",
    "PedestrianTypeSpace",
    "EnvironmentContextSpace",
    "SpatialAwarenessSpace",
    "ObstacleProximitySpace",
    "PedestrianNodeSetSpace",
    "PedestrianMaskSpace",
    "RobotPoseSpace",
]
