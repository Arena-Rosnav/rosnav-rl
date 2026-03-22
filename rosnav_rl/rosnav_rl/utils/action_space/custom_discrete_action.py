import numpy as np
import random
import string
from typing import List, Dict, Tuple, Union, Optional


def _random_name(length: int = 12) -> str:
    return "".join(random.sample(string.ascii_lowercase, length))


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
                    "name": _random_name(),
                    "linear": linear,
                    "angular": angular,
                    "translational": trans,  # Optional field
                }
            )

    return action_dicts


def generate_navigational_actions(
    linear_range: Tuple[float, float],
    angular_range: Tuple[float, float],
) -> List[Dict[str, Union[str, float]]]:
    """Generate a navigation-optimised discrete action set for differential-drive robots.

    Hand-crafted to cover the most useful motions for indoor navigation:
    forward progress at multiple speeds, in-place turns, and combined
    forward-with-lean manoeuvres.  Produces ~11 well-separated actions that
    tend to train faster than large uniform grids.

    Strategy:
        - stop
        - slow forward / full-speed forward
        - slight lean left / right at high forward speed
        - moderate turn left / right at medium speed
        - sharp in-place turn left / right
        - backward at low speed (useful for recovery)
    """
    v_max = linear_range[1]
    v_slow = max(linear_range[0], 0.0) + (v_max - max(linear_range[0], 0.0)) * 0.35
    v_med = max(linear_range[0], 0.0) + (v_max - max(linear_range[0], 0.0)) * 0.65
    v_back = linear_range[0] * 0.5  # gentle reverse

    w_max = angular_range[1]
    w_lean = w_max * 0.25   # gentle curve
    w_turn = w_max * 0.65   # pronounced turn
    # full w_max for in-place spins

    primitives = [
        (0.0, 0.0),           # stop
        (v_slow, 0.0),        # slow forward
        (v_max, 0.0),         # fast forward
        (v_max, w_lean),      # lean left (fast)
        (v_max, -w_lean),     # lean right (fast)
        (v_med, w_turn),      # curve left
        (v_med, -w_turn),     # curve right
        (0.0, w_max),         # spin left
        (0.0, -w_max),        # spin right
        (v_slow, w_max),      # creep + spin left
        (v_slow, -w_max),     # creep + spin right
        (v_back, 0.0),        # back up
    ]

    return [
        {"name": _random_name(), "linear": float(lin), "angular": float(ang), "translational": None}
        for lin, ang in primitives
    ]


def generate_exponential_actions(
    linear_range: Tuple[float, float],
    angular_range: Tuple[float, float],
    n_forward: int = 5,
    n_angular: int = 7,
) -> List[Dict[str, Union[str, float]]]:
    """Generate a log-spaced discrete action grid for differential-drive robots.

    Logarithmic spacing gives finer resolution near zero velocity — important
    for precise obstacle-avoidance manoeuvres — while still reaching the
    velocity extremes.

    Args:
        linear_range: ``(v_min, v_max)`` — only forward speeds are log-spaced
            (backward at half-speed is appended separately).
        angular_range: ``(−w_max, +w_max)`` — symmetric around zero.
        n_forward: Number of forward linear speed levels (default 5).
        n_angular: Number of angular speed levels per side (default 7);
            total angular = ``2*n_angular + 1`` (including zero).
    """
    v_max = linear_range[1]
    v_min_pos = max(linear_range[0], 0.0)

    # Log-spaced forward speeds: from a small epsilon up to v_max
    log_min = np.log1p(v_min_pos)
    log_max = np.log1p(v_max)
    forward_speeds = np.expm1(np.linspace(log_min, log_max, n_forward)).tolist()

    # Log-spaced angular speeds: symmetric, from small epsilon to w_max
    w_max = angular_range[1]
    log_w_max = np.log1p(w_max)
    pos_angulars = np.expm1(np.linspace(0, log_w_max, n_angular + 1)).tolist()
    angular_speeds = ([-a for a in reversed(pos_angulars[1:])] + pos_angulars)  # zero only once

    primitives: List[Tuple[float, float]] = []
    for lin in forward_speeds:
        for ang in angular_speeds:
            primitives.append((lin, ang))

    # Always include stop and gentle reverse
    primitives.append((0.0, 0.0))
    if linear_range[0] < 0:
        primitives.append((linear_range[0] * 0.5, 0.0))

    # Deduplicate
    seen = set()
    unique = []
    for p in primitives:
        key = (round(p[0], 4), round(p[1], 4))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return [
        {"name": _random_name(), "linear": float(lin), "angular": float(ang), "translational": None}
        for lin, ang in unique
    ]
