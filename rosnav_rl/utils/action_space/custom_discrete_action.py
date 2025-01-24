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
    Generates a dictionary of discrete actions for a robot, combining linear and angular velocities.

    Args:
        linear_range (Tuple[float, float]): The range (min, max) of linear velocities.
        angular_range (Tuple[float, float]): The range (min, max) of angular velocities.
        num_linear_actions (int): The number of discrete linear actions to generate.
        num_angular_actions (int): The number of discrete angular actions to generate.
        translational_range (Optional[Tuple[float, float]]): The range (min, max) of translational velocities.
        num_translational_actions (int): The number of discrete translational actions to generate.

    Returns:
        List[Dict[str, Union[str, float]]]: A list of dictionaries, each containing:
            - "name" (str): A randomly generated name for the action.
            - "linear" (float): The linear velocity component of the action.
            - "angular" (float): The angular velocity component of the action.
            - "translational" (Optional[float]): The translational velocity component of the action if specified.
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
