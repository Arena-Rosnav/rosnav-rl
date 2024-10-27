from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import rospy
from rl_utils.state_container import SimulationStateContainer
from rl_utils.utils.type_alias.observation import ObservationDict
from .utils import load_rew_fnc

if TYPE_CHECKING:
    from .reward_units.base_reward_units import RewardUnit


class RewardFunction:
    """
    RewardFunction class for managing and calculating rewards in a reinforcement learning environment.

    Attributes:
        _reward_file_name (str): The name of the file containing reward function configurations.
        _verbose (bool): Flag to enable verbose logging.
        _reward_unit_kwargs (dict): Additional keyword arguments for reward units.
        _curr_reward (float): Current accumulated reward.
        _info (dict): Dictionary containing additional information about the reward calculation.
        _reward_overview (dict): Overview of the reward breakdown.
        _rew_fnc_dict (dict): Dictionary containing reward function configurations.
        _reward_units (list): List of instantiated reward units.

    Methods:
        __init__(reward_file_name: str, reward_unit_kwargs: dict = None, verbose: bool = False, *args, **kwargs):
            Initializes the RewardFunction with the given parameters.

        _initialize_state():
            Initializes internal state variables.

        _initialize_reward_units():
            Sets up reward units from the configuration file.

        reward_units() -> List["RewardUnit"]:
            Returns the list of reward units.

        config() -> Dict[str, Dict[str, Any]]:
            Returns the reward function configuration dictionary.

        calculate_reward(obs_dict: ObservationDict, simulation_state_container: SimulationStateContainer, *args, **kwargs) -> None:
            Calculates the reward based on the observation dictionary and simulation state.

        get_reward(obs_dict: ObservationDict, *args, **kwargs) -> Tuple[float, Dict[str, Any]]:
            Resets the state, calculates the reward, and returns the current reward and additional information.

        add_reward(value: float, **kwargs):
            Adds a reward value and tracks its source.

        add_info(info: Dict[str, Any]):
            Updates the info dictionary with new information.

        reset():
            Resets the state before each episode.

        _reset():
            Resets the state before each step.

        _print_reward_overview():
            Prints a detailed reward breakdown if verbose mode is enabled.

        __repr__() -> str:
            Returns a string representation of the RewardFunction instance.
    """

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
            reward_file_name (str): The name of the reward file.
            reward_unit_kwargs (dict, optional): Additional keyword arguments for reward units. Defaults to None.
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._reward_file_name = reward_file_name
        self._verbose = verbose
        self._reward_unit_kwargs = reward_unit_kwargs or {}

        self._initialize_state()
        self._initialize_reward_units()

    def _initialize_state(self):
        """Initialize internal state variables"""
        self._curr_reward = 0
        self._info = {}
        self._reward_overview = {}
        self._rew_fnc_dict = load_rew_fnc(self._reward_file_name)

    def _initialize_reward_units(self):
        """Set up reward units from configuration"""
        import rosnav_rl.reward as rew_pkg

        self._reward_units = [
            rew_pkg.RewardUnitFactory.instantiate(unit_name)(
                reward_function=self, **self._reward_unit_kwargs, **params
            )
            for unit_name, params in self._rew_fnc_dict.items()
        ]

    @property
    def reward_units(self) -> List["RewardUnit"]:
        return self._reward_units

    @property
    def config(self) -> Dict[str, Dict[str, Any]]:
        return self._rew_fnc_dict

    def calculate_reward(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        *args,
        **kwargs,
    ) -> None:
        for reward_unit in self._reward_units:
            if (
                self._info.get("safe_dist_violation", False)
                and not reward_unit._on_safe_dist_violation
            ):
                continue

            reward_unit(
                obs_dict=obs_dict,
                simulation_state_container=simulation_state_container,
                **kwargs,
            )

    def get_reward(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        *args,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        self._reset()
        self.calculate_reward(
            obs_dict=obs_dict,
            simulation_state_container=simulation_state_container,
            **kwargs,
        )

        if self._verbose:
            self._print_reward_overview()

        return self._curr_reward, self._info

    def add_reward(self, value: float, **kwargs):
        """Add a reward value and track its source"""
        self._curr_reward += value

        if "called_by" in kwargs:
            self._reward_overview[kwargs["called_by"]] = value

    def add_info(self, info: Dict[str, Any]):
        """Update the info dictionary with new information"""
        self._info.update(info)

    def reset(self):
        """Reset state before each episode"""
        for reward_unit in self._reward_units:
            reward_unit.reset()

    def _reset(self):
        """Reset state before each step"""
        self._curr_reward = 0
        self._info = {}
        self._reward_overview = {}

    def _print_reward_overview(self):
        """Print detailed reward breakdown if verbose mode is enabled"""
        rospy.loginfo("____________________________________")
        rospy.loginfo("Reward Overview:")

        for key, value in self._reward_overview.items():
            rospy.loginfo(f"{key}: {value:.4f}")

        rospy.loginfo("------------------------------------")
        rospy.loginfo(f"Total Reward: {self._curr_reward:.4f}")
        rospy.loginfo("____________________________________")

    def __repr__(self) -> str:
        parts = [self.__class__.__name__ + "("]
        for name, params in self._rew_fnc_dict.items():
            parts.append(f"{name}: {params}")
        parts.append(")")
        return "\n".join(parts)
