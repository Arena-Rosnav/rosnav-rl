from typing import Any, Dict, List, Tuple

import rospy
from rl_utils.state_container import SimulationStateContainer
from rl_utils.utils.observation_collector import ObservationDict
from rl_utils.utils.observation_collector.traversal import get_required_observations

from .utils import load_rew_fnc


class RewardFunction:
    _reward_file_name: str

    _curr_reward: float
    _info: Dict[str, Any]

    _rew_fnc_dict: Dict[str, Dict[str, Any]]
    _reward_units: List["RewardUnit"]

    __simulation_state_container: SimulationStateContainer

    def __init__(
        self,
        reward_file_name: str,
        simulation_state_container: SimulationStateContainer = None,
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
        self.__simulation_state_container = simulation_state_container
        self._reward_file_name = reward_file_name

        self._curr_reward = 0
        self._info = {}

        self._rew_fnc_dict = load_rew_fnc(self._reward_file_name)

        reward_unit_kwargs = reward_unit_kwargs or {}
        self._reward_units: List["RewardUnit"] = self._setup_reward_function(
            **reward_unit_kwargs
        )

        # TODO: Add dynamic parameter for goal radius
        # self._goal_radius_updater = DynamicParameter(
        #     cls=self, key="goal_radius", message_type=Float32
        # )

        self._verbose = verbose
        self._reward_overview = {}

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for name, params in self._rew_fnc_dict.items():
            format_string += "\n"
            format_string += f"{name}: {params}"
        format_string += "\n)"
        return format_string

    @property
    def units(self) -> List["RewardUnit"]:
        return self._reward_units

    @property
    def required_observations(self):
        return get_required_observations(self._reward_units)

    @property
    def config(self) -> List[Dict[str, Any]]:
        return self._rew_fnc_dict

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
            if (
                self._info.get("safe_dist_violation", False)
                and not reward_unit._on_safe_dist_violation
            ):
                continue
            reward_unit(
                obs_dict, state_container=self.__simulation_state_container, **kwargs
            )

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

    def _setup_reward_function(self, **kwargs) -> List["RewardUnit"]:
        """Sets up the reward function.

        Returns:
            List[RewardUnit]: List of reward units for calculating the reward.
        """
        import rosnav_rl.reward as rew_pkg

        return [
            rew_pkg.RewardUnitFactory.instantiate(unit_name)(
                reward_function=self, **kwargs, **params
            )
            for unit_name, params in self._rew_fnc_dict.items()
        ]

    def _reset(self):
        """Reset on every environment step."""
        self._curr_reward = 0
        self._info = {}
        self._reward_overview = {}
