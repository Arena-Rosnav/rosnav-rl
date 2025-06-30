from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, cast

from rosnav_rl.observations import BaseUnit
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.type_aliases import ObservationDict

from ..reward_function import RewardFunction


class RewardUnit(ABC):
    """
    RewardUnit is an abstract base class for defining reward units in a reinforcement learning context.
    It provides methods for initializing the unit, adding rewards and information to the episode,
    checking parameters, and resetting the unit state. Derived classes must implement the __call__ method.

    Attributes:
        required_observation_units (List[BaseUnit]): List of required observations for the reward unit.
        _reward_function (RewardFunction): The reward function holding this unit.
        _on_safe_dist_violation (bool): Whether the unit is applied on safe distance violation.

    Methods:
        on_safe_dist_violation: Returns whether the unit is applied on safe distance violation.
        add_reward(value: float): Adds the given value to the episode's reward.
        add_info(info: dict): Adds the given information to the episode's info dict.
        check_parameters(*args, **kwargs): Checks the parsed unit parameters and sends a warning if parameters were chosen inappropriately.
        reset(): Resets the unit state after each episode.
        __call__(*args: Any, **kwargs: Any) -> Any: Abstract method to alter the reward and possibly the info dict. Must be overridden in derived classes.
    """

    required_observation_units: List[BaseUnit] = []

    def __init__(
        self,
        reward_function: RewardFunction,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initializes the RewardUnit.

        Args:
            reward_function (RewardFunction): The RewardFunction instance holding this unit.
            _on_safe_dist_violation (bool, optional): Whether the unit is applied on safe distance violation. Defaults to True.
        """
        self._reward_function = reward_function
        self._on_safe_dist_violation = _on_safe_dist_violation

        # Validate required observations
        self._validate_required_observations()

    def _validate_required_observations(self) -> None:
        """Validates that all required observation units are properly defined."""
        if not hasattr(self, "required_observation_units"):
            raise AttributeError(
                f"Class {self.__class__.__name__} must define 'required_observation_units'"
            )

    @property
    def on_safe_dist_violation(self) -> bool:
        """Returns whether the unit is applied on safe distance violation."""
        return self._on_safe_dist_violation

    def add_reward(self, value: float) -> None:
        """Adds the given value to the episode's reward.

        Args:
            value (float): The reward value to add.
        """
        self._reward_function.add_reward(value=value, called_by=self.__class__.__name__)

    def add_info(self, info: Dict[str, Any]) -> None:
        """Adds the given information to the episode's info dict.

        Args:
            info (Dict[str, Any]): Information to add to the episode's info dict.
        """
        self._reward_function.add_info(info=info)

    def check_parameters(self, *args: Any, **kwargs: Any) -> None:
        """Method to check the parsed unit parameters. Send warning if params were chosen inappropriately."""
        pass

    def reset(self) -> None:
        """Method to reset the unit state after each episode."""
        pass

    @abstractmethod
    def __call__(
        self,
        obs_dict: ObservationDict,
        state_container: SimulationStateContainer,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Process observations and update rewards.

        Args:
            obs_dict: Dictionary of observations.
            state_container: Container for simulation state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Implementation-specific return value.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement __call__")


# The GlobalplanRewardUnit is commented out in the original code, so I'm leaving it as is
