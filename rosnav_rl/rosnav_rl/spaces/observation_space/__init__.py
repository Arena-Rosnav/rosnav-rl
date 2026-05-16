"""Observation Space Module

Provides SpaceFactory for registering and creating observation spaces,
and SpaceCategory enum for type-safe categorization.
"""

from .space_categories import SpaceCategory
from .observation_space_factory import SpaceFactory
from .observation_space_manager import ObservationSpaceManager

__all__ = ["SpaceCategory", "SpaceFactory", "ObservationSpaceManager"]
