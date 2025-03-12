import numpy as np
import random
import string
from typing import List, Dict, Tuple, Union, Optional


def generate_discrete_action_dict(
    linear_range: Tuple[float, float],
    angular_range: Tuple[float, float],
    num_linear_actions: int,
    num_angular_actions: int,
    translational_range: Optional[Tuple[float, float]] = None,
    num_translational_actions: int = 0,
) -> List[Dict[str, Union[str, float]]]:
    """
    Generate a discrete action dictionary for robot control with linear and angular velocity pairs,
    and optional translational velocity.
    
    This function creates a list of action dictionaries, where each dictionary represents a 
    unique combination of linear and angular velocities, and optionally translational velocity.
    The function ensures that a zero action (0, 0) is included in the action space.
    
    Args:
        linear_range: A tuple (min, max) specifying the range of linear velocities.
        angular_range: A tuple (min, max) specifying the range of angular velocities.
        num_linear_actions: The number of discrete linear velocity values to generate.
        num_angular_actions: The number of discrete angular velocity values to generate.
        translational_range: Optional tuple (min, max) for translational velocities.
        num_translational_actions: Number of discrete translational velocity values to generate,
            defaults to 0 (no translational actions).
            
    Returns:
        A list of dictionaries, where each dictionary has the following keys:
        - 'name': A random string of lowercase letters (length 12) to identify the action.
        - 'linear': The linear velocity value.
        - 'angular': The angular velocity value.
        - 'translational': The translational velocity value (None if not specified).
    """
    NAME_LEN = 12  # Length for random action name

    # Generate linear and angular actions
    linear_actions = np.linspace(
        linear_range[0], linear_range[1], num_linear_actions, dtype=np.float16
    )
    angular_actions = np.linspace(
        angular_range[0], angular_range[1], num_angular_actions, dtype=np.float16
    )

    # Initialize discrete action space
    discrete_action_space = [
        (float(linear_action), float(angular_action))
        for linear_action in linear_actions
        for angular_action in angular_actions
    ]

    # Include zero action if not present
    if (0, 0) not in discrete_action_space:
        discrete_action_space.append((0, 0))

    # Generate translational actions if specified
    translational_actions = []
    if translational_range is not None and num_translational_actions > 0:
        translational_actions = np.linspace(
            translational_range[0],
            translational_range[1],
            num_translational_actions,
            dtype=np.float16,
        )

    # Create action dictionary list
    action_dicts = []

    for linear, angular in discrete_action_space:
        for trans in translational_actions if translational_actions else [None]:
            action_dicts.append(
                {
                    "name": "".join(random.sample(string.ascii_lowercase, NAME_LEN)),
                    "linear": linear,
                    "angular": angular,
                    "translational": trans,  # Optional field
                }
            )

    return action_dicts
