"""Laser Perception Spaces Package

Advanced and basic laser processing spaces for robotic navigation.
"""

from .basic_laser_spaces import LaserScanSpace, ReducedLaserScanSpace
from .laser_spaces import (
    MultiLaserFusionSpace,
    MultiRangeLaserSpace,
    ReliableLaserSpace,
)

__all__ = [
    # Advanced laser processing
    "ReliableLaserSpace",
    "MultiRangeLaserSpace",
    "MultiLaserFusionSpace",
    # Basic laser processing
    "LaserScanSpace",
    "ReducedLaserScanSpace",
]
