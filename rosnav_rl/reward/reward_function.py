from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import rospy
from rl_utils.state_container import SimulationStateContainer
from rl_utils.utils.type_alias.observation import ObservationDict

from .utils import load_rew_fnc

if TYPE_CHECKING:
    from .reward_units.base_reward_units import RewardUnit


class RewardFunction:
    """
    RewardFunction class is responsible for managing and calculating rewards in a reinforcement learning environment.

    Attributes:
        _reward_file_name (str): The name of the file containing reward function configurations.
        _curr_reward (float): The current reward value.
        _info (Dict[str, Any]): Dictionary containing additional information about the reward.
        _rew_fnc_dict (Dict[str, Dict[str, Any]]): Dictionary containing reward function configurations.
        _reward_units (List["RewardUnit"]): List of reward units used to calculate the reward.
        _verbose (bool): Flag to enable verbose logging.
        _reward_overview (Dict[str, float]): Overview of rewards added by different units.

    Methods:
        __init__(reward_file_name: str, simulation_state_container: SimulationStateContainer = None, reward_unit_kwargs: dict = None, verbose: bool = False, *args, **kwargs):
            Initializes the RewardFunction with the specified parameters.

        __repr__() -> str:
            Returns a string representation of the RewardFunction.

        reward_units() -> List["RewardUnit"]:
            Returns the list of reward units.

        config() -> List[Dict[str, Any]]:
            Returns the reward function configuration dictionary.

        add_reward(value: float, *args, **kwargs):
            Adds the specified value to the current reward.

        add_info(info: Dict[str, Any]):
            Adds the specified information to the reward function's info dictionary.

        reset():
            Resets the reward function before each episode.

        calculate_reward(obs_dict: ObservationDict, *args, **kwargs) -> None:
            Calculates the reward based on several observations.

        get_reward(obs_dict: ObservationDict, *args, **kwargs) -> Tuple[float, Dict[str, Any]]:
            Retrieves the current reward and info dictionary.

        print_reward_overview():
            Prints an overview of the rewards added by different units.

        _setup_reward_function(**kwargs) -> List["RewardUnit"]:
            Sets up the reward function and returns a list of reward units.

        _reset():
            Resets the reward function on every environment step.
    """

    _reward_file_name: str

    _curr_reward: float
    _info: Dict[str, Any]

    _rew_fnc_dict: Dict[str, Dict[str, Any]]
    _reward_units: List["RewardUnit"]

    def __init__(
        self,
        reward_file_name: str,
        reward_unit_kwargs: dict = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes the reward function.

        Args:
            reward_file_name (str): The name of the file containing the reward function.
            simulation_state_container (SimulationStateContainer, optional): Container for the simulation state. Defaults to None.
            reward_unit_kwargs (dict, optional): Additional keyword arguments for reward units. Defaults to None.
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
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
    def reward_units(self) -> List["RewardUnit"]:
        return self._reward_units

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
        simulation_state_container = obs_dict.get("simulation_state_container")
        for reward_unit in self._reward_units:
            if (
                self._info.get("safe_dist_violation", False)
                and not reward_unit._on_safe_dist_violation
            ):
                continue

            # if "simulation_state_container" not in obs_dict:
            #     obs_dict["simulation_state_container"] = (
            #         self.__simulation_state_container
            #     )
            reward_unit(
                obs_dict=obs_dict,
                simulation_state_container=simulation_state_container,
                **kwargs,
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
