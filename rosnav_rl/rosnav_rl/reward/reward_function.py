from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import rospy
from pydantic.dataclasses import Field, dataclass

from rosnav_rl.cfg.reward import RewardFunctionDict
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.type_aliases import ObservationDict

if TYPE_CHECKING:
    from .reward_units.base_reward_units import RewardUnit


@dataclass
class RewardState:
    """Container for reward calculation state."""

    current_reward: float = 0.0
    info: Dict[str, Any] = Field(default_factory=dict)
    reward_overview: Dict[str, float] = Field(default_factory=dict)


@dataclass
class RewardConfig:
    """Container for reward configuration."""

    reward_function_dict: Dict[str, Any]
    unit_kwargs: Dict[str, Any] = Field(default_factory=dict)
    verbose: bool = False


class RewardFunction:
    """
    RewardFunction class to manage and calculate rewards based on given configurations.

        Attributes:
            config (RewardConfig): Configuration for the reward function.
            state (RewardState): Current state of the reward function.
            _reward_units (List[RewardUnit]): List of reward unit instances.

        Methods:
            __init__(function_dict, unit_kwargs=None, verbose=True):
                Initialize the reward function with given configurations.
            _create_reward_units() -> List["RewardUnit"]:
                Create reward unit instances from configuration.
            _create_reward_unit(factory, unit_name, params) -> "RewardUnit":
                Create a single reward unit instance.
            calculate_reward(obs_dict, simulation_state_container, **kwargs) -> None:
                Calculate rewards using all reward units.
            _skip_on_safe_dist_violation(reward_unit) -> bool:
                Determine if a reward unit should be skipped.
            get_reward(obs_dict, simulation_state_container, **kwargs) -> Tuple[float, Dict[str, Any]]:
            add_reward(value, **kwargs) -> None:
            add_info(info) -> None:
                Update the info dictionary.
            reset() -> None:
                Reset all reward units between episodes.
            _reset_state() -> None:
                Reset the reward state between steps.
            _log_reward_overview() -> None:
                Log detailed reward breakdown.
            reward_units() -> List["RewardUnit"]:
                Get the list of reward units.
            __repr__() -> str:
                String representation of the reward function.
    """

    def __init__(
        self,
        function_dict: RewardFunctionDict,
        unit_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the reward function.

        Args:
            function_dict (Dict[str, Union[str, float, int]]): Dictionary containing reward function parameters.
            unit_kwargs (Optional[Dict[str, Any]]): Additional arguments for reward units. Defaults to None.
            verbose (bool): Enable detailed logging. Defaults to True.
        """
        self.config = RewardConfig(
            reward_function_dict=function_dict,
            unit_kwargs=unit_kwargs or {},
            verbose=verbose,
        )
        self.state = RewardState()
        self._reward_units = self._create_reward_units()

    def _create_reward_units(self) -> List["RewardUnit"]:
        """Create reward unit instances from configuration."""
        import rosnav_rl.reward as rew_pkg

        return [
            self._create_reward_unit(rew_pkg.RewardUnitFactory, unit_name, params)
            for unit_name, params in self.config.reward_function_dict.items()
        ]

    def _create_reward_unit(
        self, factory: Any, unit_name: str, params: Dict[str, Any]
    ) -> "RewardUnit":
        """Create a single reward unit instance."""
        unit_class = factory.instantiate(unit_name)
        return unit_class(reward_function=self, **self.config.unit_kwargs, **params)

    def calculate_reward(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> None:
        """Calculate rewards using all reward units."""
        for reward_unit in self._reward_units:
            if self._skip_on_safe_dist_violation(reward_unit):
                continue

            reward_unit(
                obs_dict=obs_dict,
                simulation_state_container=simulation_state_container,
                **kwargs,
            )

    def _skip_on_safe_dist_violation(self, reward_unit: "RewardUnit") -> bool:
        """Determine if a reward unit should be skipped."""
        return (
            self.state.info.get("safe_dist_violation", False)
            and not reward_unit._on_safe_dist_violation
        )

    def get_reward(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate and return the current reward and information.

        Returns:
            Tuple of (reward value, info dictionary)
        """
        self._reset_state()
        self.calculate_reward(
            obs_dict=obs_dict,
            simulation_state_container=simulation_state_container,
            **kwargs,
        )

        if self.config.verbose:
            self._log_reward_overview()

        return self.state.current_reward, self.state.info

    def add_reward(self, value: float, **kwargs) -> None:
        """
        Add a reward value and track its source.

        Args:
            value: Reward value to add
            **kwargs: Additional metadata about the reward
        """
        self.state.current_reward += value

        if called_by := kwargs.get("called_by"):
            self.state.reward_overview[called_by] = value

    def add_info(self, info: Dict[str, Any]) -> None:
        """Update the info dictionary."""
        self.state.info.update(info)

    def reset(self) -> None:
        """Reset all reward units between episodes."""
        for reward_unit in self._reward_units:
            reward_unit.reset()

    def _reset_state(self) -> None:
        """Reset the reward state between steps."""
        self.state = RewardState()

    def _log_reward_overview(self) -> None:
        """Log detailed reward breakdown."""
        log_messages = [
            "____________________________________",
            "Reward Overview:",
            *[
                f"{key}: {value:.4f}"
                for key, value in self.state.reward_overview.items()
            ],
            "------------------------------------",
            f"Total Reward: {self.state.current_reward:.4f}",
            "____________________________________",
        ]

        for message in log_messages:
            rospy.loginfo(message)

    @property
    def reward_units(self) -> List["RewardUnit"]:
        """Get the list of reward units."""
        return self._reward_units

    def __repr__(self) -> str:
        """String representation of the reward function."""
        return "\n".join(
            [
                f"{self.__class__.__name__}(",
                *[
                    f"  {name}: {params}"
                    for name, params in self.config.reward_function_dict.items()
                ],
                ")",
            ]
        )
