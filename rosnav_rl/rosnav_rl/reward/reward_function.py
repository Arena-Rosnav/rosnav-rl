import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from rosnav_rl.cfg.reward import RewardFunctionDict
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.utils.logging import (
    ComponentType,
    ErrorReportingMixin,
)
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.utils.validation import validate_reward_units

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .reward_units.base_reward_units import RewardUnit

# Configuration constants
ERROR_MESSAGE_SUFFIX = " ---> thus not applied (+/- 0.0)."


@dataclass
class RewardState:
    """
    A class to represent the state of the reward during reinforcement learning.

    Attributes:
        current_reward (float): The current reward value.
        info (Dict[str, Any]): Additional information related to the reward calculation.
        reward_overview (Dict[str, float]): Overview of different reward components and their values.
    """

    current_reward: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)
    reward_overview: Dict[str, float] = field(default_factory=dict)


class RewardFunction(ErrorReportingMixin):
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
        enable_validation: bool = True,
        verbose: int = 0,
        **kwargs,
    ):
        """Initialize the reward function.

        Args:
            function_dict: Dictionary containing reward function parameters.
            unit_kwargs: Additional arguments for reward units.
            enable_validation: Whether to validate reward units.
            verbose: Enable detailed logging.
        """
        super().__init__(
            component_type=ComponentType.REWARD_FUNCTION,
            component_name="RewardFunction",
        )

        # Configuration
        self.reward_function_dict = function_dict
        self.unit_kwargs = unit_kwargs or {}
        self.verbose = verbose

        # State
        self.state = RewardState()
        self._validate_units = enable_validation

        # Reusable kwargs dict to avoid per-step allocation
        self._execution_kwargs: Dict[str, Any] = {}

        # Reward units
        self._reward_units: List["RewardUnit"] = []
        self._safe_dist_sensitive_units: List["RewardUnit"] = []
        self._safe_dist_insensitive_units: List["RewardUnit"] = []

        self._initialize_reward_units()

    def _initialize_reward_units(self) -> None:
        """Initialize and categorize reward units for optimal performance."""
        import rosnav_rl.reward as rew_pkg

        self._reward_units = [
            self._create_reward_unit(rew_pkg.RewardUnitFactory, unit_name, params)
            for unit_name, params in self.reward_function_dict.items()
        ]

        self._categorize_units_by_safety_sensitivity()

    def _categorize_units_by_safety_sensitivity(self) -> None:
        """Categorize units based on safe distance violation sensitivity."""
        for unit in self._reward_units:
            if (
                hasattr(unit, "_on_safe_dist_violation")
                and unit._on_safe_dist_violation
            ):
                self._safe_dist_sensitive_units.append(unit)
            else:
                self._safe_dist_insensitive_units.append(unit)

    def _create_reward_unit(
        self, factory: Any, unit_name: str, params: Dict[str, Any]
    ) -> "RewardUnit":
        """Create a single reward unit instance with error handling.

        Args:
            factory: The factory to create the reward unit.
            unit_name: Name of the reward unit.
            params: Parameters for the reward unit.

        Returns:
            An instance of the specified reward unit.
        """
        with self._error_context(f"creating reward unit '{unit_name}'"):
            unit_class = factory.instantiate(unit_name)
            return unit_class(reward_function=self, **self.unit_kwargs, **params)

    def calculate_reward(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> None:
        """Calculate rewards by executing all reward units sequentially.

        Args:
            obs_dict: Dictionary of observations.
            simulation_state_container: Container for simulation state.
            **kwargs: Additional arguments passed to reward units.
        """
        self._validate_if_enabled(obs_dict)
        eligible_units = self._get_eligible_units(obs_dict)

        if not eligible_units:
            return

        execution_kwargs = self._prepare_execution_kwargs(
            obs_dict, simulation_state_container, **kwargs
        )
        self._execute_reward_units(eligible_units, execution_kwargs)

    def _validate_if_enabled(self, obs_dict: ObservationDict) -> None:
        """Validate reward unit requirements if validation is enabled.

        After the first successful validation, automatically disables further
        checks for performance since the key set is static at runtime.
        """
        if self._validate_units:
            with self._error_context("validating reward unit requirements"):
                validate_reward_units(
                    obs_dict,
                    {unit.__class__.__name__: unit for unit in self._reward_units},
                )
                # Disable after first successful validation — keys don't change at runtime
                self._validate_units = False

    def _get_eligible_units(self, obs_dict: ObservationDict) -> List["RewardUnit"]:
        """Get eligible reward units based on current safety state from observations.

        Reads ``laser_safety_violation`` directly from the live observation
        dictionary so the check reflects the *current* step rather than the
        (already-cleared) reward state, which was always False before this fix.
        """
        safe_dist_violation = bool(obs_dict.get("laser_safety_violation", False))
        return (
            self._safe_dist_sensitive_units
            if safe_dist_violation
            else self._reward_units
        )

    def _prepare_execution_kwargs(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare arguments for reward unit execution.

        Reuses a single mutable dict to avoid per-step allocation.
        The dict is cleared and repopulated each call.
        """
        ek = self._execution_kwargs
        ek.clear()
        ek.update(obs_dict)
        ek["simulation_state_container"] = simulation_state_container
        if kwargs:
            ek.update(kwargs)
        return ek

    def _execute_reward_units(
        self, units: List["RewardUnit"], kwargs: Dict[str, Any]
    ) -> None:
        """Execute reward units sequentially."""
        if len(units) == 1:
            self._execute_single_unit(units[0], kwargs)
        else:
            self._calculate_reward_sequential(units, kwargs)

    def _execute_single_unit(self, unit: "RewardUnit", kwargs: Dict[str, Any]) -> None:
        """Execute a single reward unit with optimized error handling."""
        try:
            unit(**kwargs)
        except Exception as e:
            self._handle_unit_error(unit.__class__.__name__, e)

    def _handle_unit_error(self, unit_name: str, error: Exception) -> None:
        """Handle errors from reward unit execution."""
        error_message = (
            f"Error during reward calculation: {str(error)}{ERROR_MESSAGE_SUFFIX}"
        )
        if self.verbose:
            self._report_error(message=error_message, component_name=unit_name)

    def _calculate_reward_sequential(
        self, reward_units: List["RewardUnit"], all_kwargs: Dict[str, Any]
    ) -> None:
        """Calculate rewards sequentially with optimized error handling."""
        for reward_unit in reward_units:
            try:
                reward_unit(**all_kwargs)
            except Exception as e:
                self._handle_unit_error(reward_unit.__class__.__name__, e)

    def get_reward(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate and return the current reward and information.

        Args:
            obs_dict: Dictionary of observations.
            simulation_state_container: Container for simulation state.
            **kwargs: Additional arguments for reward calculation.

        Returns:
            Tuple of (reward value, info dictionary)
        """
        self._reset_state()
        self.calculate_reward(obs_dict, simulation_state_container, **kwargs)

        if self.verbose >= 2:
            self._log_reward_overview()

        return self.state.current_reward, self.state.info

    def add_reward(self, value: float, **kwargs) -> None:
        """Add a reward value and track its source.

        Args:
            value: Reward value to add
            **kwargs: Additional metadata about the reward
        """
        called_by = kwargs.get("called_by")

        self.state.current_reward += value
        if called_by:
            self.state.reward_overview[called_by] = value

    def _update_reward_and_overview(
        self, value: float, called_by: Optional[str]
    ) -> None:
        """Update current reward and overview tracking."""
        self.state.current_reward += value
        if called_by:
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
        """Reset the reward state between steps (optimized for performance)."""
        self.state.current_reward = 0.0
        self.state.info.clear()
        self.state.reward_overview.clear()

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
            _logger.debug(message)

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
                    for name, params in self.reward_function_dict.items()
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
            function_dict=self.reward_function_dict,
            unit_kwargs=self.unit_kwargs,
            enable_validation=self._validate_units,
            verbose=self.verbose,
        )
