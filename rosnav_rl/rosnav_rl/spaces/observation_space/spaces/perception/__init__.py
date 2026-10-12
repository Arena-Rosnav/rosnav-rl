"""Perception Spaces - Advanced & Basic

This module contains both advanced and basic perception observation spaces.

Advanced Spaces:
    - ReliableLaserSpace: Enhanced single-sensor laser processing
    - MultiRangeLaserSpace: Multi-scale laser representation
    - MultiLaserFusionSpace: Multi-sensor laser fusion

Basic Spaces:
    - LaserScanSpace: Standard laser scan space
    - ReducedLaserScanSpace: Reduced laser scan space
    - RGBDSpace: RGB-D vision space
"""

# Import from sub-packages
from .laser import (
    LaserScanSpace,
    MultiLaserFusionSpace,
    MultiRangeLaserSpace,
    ReducedLaserScanSpace,
    ReliableLaserSpace,
)
<<<<<<< HEAD

from .vision import (
    RGBDSpace,
    RGBSpace,
)
=======
from .vision import RGBDSpace
>>>>>>> 12ae3c9

__all__ = [
    # Basic spaces
    "LaserScanSpace",
    "ReducedLaserScanSpace",
    "RGBDSpace",
    "RGBSpace",
    # Advanced spaces
    "ReliableLaserSpace",
    "MultiRangeLaserSpace",
    "MultiLaserFusionSpace",
]
