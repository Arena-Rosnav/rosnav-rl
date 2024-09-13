from typing import Any, Dict, List, Tuple

import numpy as np
import rospy
from rl_utils.utils.observation_collector import ObservationDict
from rl_utils.utils.observation_collector.traversal import get_required_observations
from std_msgs.msg import Float32
from task_generator.shared import Namespace
from tools.dynamic_parameter import DynamicParameter

from .constants import REWARD_CONSTANTS
from .utils import load_rew_fnc

from .reward_units.reward_units import RewardSafeDistance


class RewardFunction:
    """Represents a reward function for a reinforcement learning environment.

    Attributes:
        _rew_func_name (str): Name of the yaml file that contains the reward function specifications.
        _robot_radius (float): Radius of the robot.
        _safe_dist (float): Safe distance of the agent.
        _goal_radius (float): Radius of the goal.

        _distinguished_safe_dist: If true, the unity-based collider approach is used for safe dist.
        _max_steps (int): Maximum number of steps allowed in the environment.

        _internal_state_info (Dict[str, Any]): Centralized internal state info for the reward units.
            E.g. to avoid computing the same parameter multiple times in a single step.

        _curr_reward (float): Current reward value.
        _info (Dict[str, Any]): Dictionary containing reward function information.

        _rew_fnc_dict (Dict[str, Dict[str, Any]]): Dictionary containing reward function specifications.
        _reward_units (List[RewardUnit]): List of reward units for calculating the reward.

    Methods:
        __init__: Initializes the RewardFunction object.
        _setup_reward_function: Sets up the reward function.
        add_reward: Adds the specified value to the current reward.
        add_info: Adds the specified information to the reward function's info dictionary.
        add_internal_state_info: Adds internal state information to the reward function.
        get_internal_state_info: Retrieves internal state information based on the specified key.
        update_internal_state_info: Updates the internal state info after each time step.
        reset_internal_state_info: Resets all global state information.
        _reset: Reset on every environment step.
        reset: Reset before each episode.
        calculate_reward: Calculates the reward based on several observations.
        get_reward: Retrieves the current reward and info dictionary.

    Properties:
        robot_radius: Getter for the robot radius.
        goal_radius: Getter and setter for the goal radius.
        safe_dist: Getter for the safe distance.
        safe_dist_breached: Getter for the safe distance breached flag.
    """

    _rew_func_name: str
    _robot_radius: float
    _safe_dist: float
    _goal_radius: float

    _distinguished_safe_dist: bool

    _max_steps: int

    _curr_reward: float
    _info: Dict[str, Any]

    _rew_fnc_dict: Dict[str, Dict[str, Any]]
    _reward_units: List["RewardUnit"]

    def __init__(
        self,
        rew_func_name: str,
        robot_radius: float,
        goal_radius: float,
        max_steps: int,
        safe_dist: float,
        distinguished_safe_dist: bool,
        ns: Namespace,
        reward_unit_kwargs: dict = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize a reward function for a reinforcement learning environment.

        Args:
            rew_func_name (str): Name of the yaml file that contains the reward function specifications.
            robot_radius (float): Radius of the robot.
            goal_radius (float): Radius of the goal.
            max_steps (int): Maximum number of steps in the environment.
            safe_dist (float): Safe distance of the agent.
            internal_state_updates (List[InternalStateInfoUpdate], optional): List of internal state updates. Defaults to None.
            reward_unit_kwargs (dict, optional): Keyword arguments for reward units. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._rew_func_name = rew_func_name
        self._robot_radius = robot_radius
        self._safe_dist = safe_dist
        self._goal_radius = goal_radius
        self._max_steps = max_steps

        self._distinguished_safe_dist = distinguished_safe_dist
        self._ns = ns

        self._curr_reward = 0
        self._info = {}

        self._rew_fnc_dict = load_rew_fnc(self._rew_func_name)

        reward_unit_kwargs = reward_unit_kwargs or {}
        self._reward_units: List["RewardUnit"] = self._setup_reward_function(
            **reward_unit_kwargs
        )

        self._goal_radius_updater = DynamicParameter(
            cls=self, key="goal_radius", message_type=Float32
        )

        self._verbose = verbose
        self._reward_overview = {}

    def _setup_reward_function(self, **kwargs) -> List["RewardUnit"]:
        """Sets up the reward function.

        Returns:
            List[RewardUnit]: List of reward units for calculating the reward.
        """
        import rl_utils.utils.rewards as rew_pkg

        return [
            rew_pkg.RewardUnitFactory.instantiate(unit_name)(
                reward_function=self, **kwargs, **params
            )
            for unit_name, params in self._rew_fnc_dict.items()
        ]

    def add_reward(self, value: float, *args, **kwargs):
        """Adds the specified value to the current reward.

        Args:
            value (float): Reward to be added. Typically called by the RewardUnit.
        """
        self._curr_reward += value

        if "called_by" in kwargs:
            self._reward_overview[kwargs["called_by"]] = value

    def add_info(self, info: Dict[str, Any]):
        """Adds the specified information to the reward function's info dictionary.

        Args:
            info (Dict[str, Any]): RewardUnits information to be added.
        """
        self._info.update(info)

    def _reset(self):
        """Reset on every environment step."""
        self._curr_reward = 0
        self._info = {}
        self._reward_overview = {}

    def reset(self):
        """Reset before each episode."""
        for reward_unit in self._reward_units:
            reward_unit.reset()

    def calculate_reward(self, obs_dict: ObservationDict, *args, **kwargs) -> None:
        """Calculates the reward based on several observations.

        Args:
            laser_scan (np.ndarray): Array containing the laser data.
        """
        for reward_unit in self._reward_units:
            if self._info.get("safe_dist_violation", False):
                continue
            reward_unit(obs_dict, **kwargs)

    def get_reward(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """Retrieves the current reward and info dictionary.

        Args:
            laser_scan (np.ndarray): Array containing the laser data.
            point_cloud (np.ndarray): Array containing the point cloud data.
            from_aggregate_obs (bool): Iff the observation from the aggreation (GetDump.srv) should be considered.

        Returns:
            Tuple[float, Dict[str, Any]]: Tuple of the current timesteps reward and info.
        """
        self._reset()
        self.calculate_reward(obs_dict=obs_dict, **kwargs)
        if self._verbose:
            self.print_reward_overview()
        return self._curr_reward, self._info

    def print_reward_overview(self):
        rospy.loginfo("_____________________________")
        rospy.loginfo("Reward Overview:")
        for key, value in self._reward_overview.items():
            rospy.loginfo(f"{key}: {value}")
        rospy.loginfo("-----------------------------")
        rospy.loginfo(f"Total Reward: {self._curr_reward}")
        rospy.loginfo("_____________________________")

    @property
    def robot_radius(self) -> float:
        return self._robot_radius

    @property
    def goal_radius(self) -> float:
        return self._goal_radius

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @goal_radius.setter
    def goal_radius(self, value) -> None:
        if value < REWARD_CONSTANTS.MIN_GOAL_RADIUS:
            raise ValueError(
                f"Given goal radius ({value}) smaller than {REWARD_CONSTANTS.MIN_GOAL_RADIUS}"
            )

        self._goal_radius = value

    @property
    def safe_dist(self) -> float:
        return self._safe_dist

    @property
    def distinguished_safe_dist(self) -> bool:
        return self._distinguished_safe_dist

    @property
    def ns(self) -> Namespace:
        return self._ns

    @property
    def units(self) -> List["RewardUnit"]:
        return self._reward_units

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for name, params in self._rew_fnc_dict.items():
            format_string += "\n"
            format_string += f"{name}: {params}"
        format_string += "\n)"
        return format_string

    @property
    def required_observations(self):
        return get_required_observations(self._reward_units)
