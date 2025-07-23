"""Simple Usage Examples for SpaceFactory-based Observation Manager

Clean, simple examples showing how to use the SpaceFactory pattern
without all the bloated monitoring and validation.
"""

from ..observation_space_factory import SpaceFactory
from .simple_observation_manager import SimpleObservationSpaceManager


def get_basic_navigation_config():
    """Get a basic navigation configuration using common spaces."""
    return {
        "laser": {"num_beams": 512},
        "dist_angle_to_goal": {},
        "last_action": {},
    }


def get_advanced_navigation_config():
    """Get an advanced navigation configuration."""
    return {
        "reliable_laser": {
            "num_beams": 512,
            "max_range": 10.0,
            "min_range": 0.1,
            "noise_reduction": True,
        },
        "robust_goal": {
            "use_global_plan": True,
            "local_goal_distance": 2.0,
        },
        "motion_state": {
            "include_acceleration": True,
            "history_length": 5,
        },
        "is_first": {},
    }


def create_simple_manager_example():
    """Example of creating and using the simple observation manager."""
    # Create manager
    manager = SimpleObservationSpaceManager()

    # Load configuration - SpaceFactory handles instantiation
    config = get_basic_navigation_config()
    manager.load_configuration(config)

    # Show what's available and loaded
    print("Available spaces:", manager.get_available_spaces())
    print("Loaded spaces:", manager.get_loaded_spaces())

    # Get combined gym space
    gym_space = manager.get_gym_space()
    print("Combined gym space:", gym_space)

    return manager


def demonstrate_spacefactory_registry():
    """Show all spaces registered with SpaceFactory."""
    print("SpaceFactory Registry:")
    for space_name in sorted(SpaceFactory.registry.keys()):
        space_class = SpaceFactory.registry[space_name]
        print(f"  {space_name}: {space_class.__name__}")


if __name__ == "__main__":
    print("=== Simple Observation Space Manager Example ===")

    # Import all categories to trigger SpaceFactory registration
    from . import (
        localization,
        perception,
        navigation,
        dynamics,
        environment,
        meta,
    )  # noqa: F401

    demonstrate_spacefactory_registry()
    print()

    manager = create_simple_manager_example()
