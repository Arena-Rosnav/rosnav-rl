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

Crowd Scalar Spaces:
    - CrowdTimeToContactSpace: Crowd time-to-contact summary
    - CrowdReachabilitySpace: Share of the robot's velocities blocked by the crowd
"""

from .advanced_environment_spaces import (
    EnvironmentContextSpace,
    ObstacleProximitySpace,
    SpatialAwarenessSpace,
)
from .crowd_spaces import CrowdReachabilitySpace, CrowdTimeToContactSpace
from .feature_map_spaces import (
    LaserCartesianMapSpace,
    PedestrianLocationSpace,
    PedestrianSocialStateSpace,
    PedestrianTypeSpace,
    PedestrianVelXSpace,
    PedestrianVelYSpace,
    StackedLaserMapSpace,
)

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
    "CrowdTimeToContactSpace",
    "CrowdReachabilitySpace",
]
