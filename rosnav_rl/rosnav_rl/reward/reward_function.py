from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

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
    """The RewardFunction class manages reward calculation for reinforcement learning.

    This class calculates rewards based on configured reward units, which can be combined
    to create complex reward functions. It handles the instantiation of reward units,
    aggregation of rewards, and management of reward-related state information.

    Attributes:
        config (RewardConfig): Configuration for the reward function.
        state (RewardState): Current state of rewards and information.
        _reward_units (List[RewardUnit]): List of instantiated reward unit objects.

    Example:
        ```python
        reward_dict = {
            "goal_reached": {"reward": 15.0},
            "safe_dist": {"reward": -1.0, "safe_dist": 0.3}
        }
        reward_function = RewardFunction(reward_dict)
        reward, info = reward_function.get_reward(observation, sim_state)
        ```
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
        self._reward_units: List["RewardUnit"] = []
        self._create_reward_units()

    def _create_reward_units(self) -> None:
        """Create reward unit instances from configuration."""
        import rosnav_rl.reward as rew_pkg

        self._reward_units = [
            self._create_reward_unit(rew_pkg.RewardUnitFactory, unit_name, params)
            for unit_name, params in self.config.reward_function_dict.items()
        ]

    def _create_reward_unit(
        self, factory: Any, unit_name: str, params: Dict[str, Any]
    ) -> "RewardUnit":
        """Create a single reward unit instance.

        Args:
            factory: The factory to create the reward unit.
            unit_name: Name of the reward unit.
            params: Parameters for the reward unit.

        Returns:
            An instance of the specified reward unit.

        Raises:
            ValueError: If the unit cannot be instantiated.
        """
        try:
            unit_class = factory.instantiate(unit_name)
            return unit_class(reward_function=self, **self.config.unit_kwargs, **params)
        except Exception as e:
            raise ValueError(
                f"Failed to create reward unit '{unit_name}': {str(e)}"
            ) from e

    def calculate_reward(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> None:
        """Calculate rewards using all reward units.

        Args:
            obs_dict: Dictionary of observations.
            simulation_state_container: Container for simulation state.
            **kwargs: Additional arguments passed to reward units.
        """
        for reward_unit in self._reward_units:
            if self._skip_on_safe_dist_violation(reward_unit):
                continue
            try:
                reward_unit(
                    obs_dict=obs_dict,
                    simulation_state_container=simulation_state_container,
                    **kwargs,
                )
            except KeyError as e:
                raise KeyError(
                    f"KeyError in reward unit '{reward_unit.name}': {str(e)}. Check if the "
                    "observation dictionary contains the required observations."
                ) from e

    def _skip_on_safe_dist_violation(self, reward_unit: "RewardUnit") -> bool:
        """Determine if a reward unit should be skipped.

        Args:
            reward_unit: The reward unit to check.

        Returns:
            True if the unit should be skipped, False otherwise.
        """
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

        Args:
            obs_dict: Dictionary of observations.
            simulation_state_container: Container for simulation state.
            **kwargs: Additional arguments for reward calculation.

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
        """Update the info dictionary.

        Args:
            info: Dictionary of information to add.
        """
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
            print(message)

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

    def copy(self) -> "RewardFunction":
        """Create a deep copy of the reward function.

        Returns:
            A new RewardFunction with the same configuration.
        """
        return RewardFunction(
            function_dict=self.config.reward_function_dict,
            unit_kwargs=self.config.unit_kwargs,
            verbose=self.config.verbose,
        )
