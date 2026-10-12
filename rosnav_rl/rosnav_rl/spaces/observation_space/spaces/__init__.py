"""Hierarchical Observation Spaces

This module provides a hierarchical organization of observation spaces for RosNav-RL
using the SpaceFactory pattern.

Categories:
    - localization: Robot pose and odometry spaces
    - perception: Sensor data processing (laser, vision)
    - navigation: Goal and path planning related spaces
    - dynamics: Motion and action related spaces
    - environment: Environmental context and obstacle spaces
    - meta: Mission and performance context spaces

All spaces are automatically registered with SpaceFactory when imported.
"""

# Import all categories to register spaces with SpaceFactory
from . import dynamics, environment, localization, meta, navigation, perception

__all__ = [
    "localization",
    "perception",
    "navigation",
    "dynamics",
    "environment",
    "meta",
]
