import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from rosnav_rl.cfg.reward import RewardFunctionDict
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.logging import (
    ComponentType,
    ErrorReportingMixin,
    ErrorSeverity,
)
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.utils.validation import validate_reward_units

if TYPE_CHECKING:
    from .reward_units.base_reward_units import RewardUnit

# Configuration constants
DEFAULT_MAX_WORKERS = 8
DEFAULT_TIMEOUT_SECONDS = 0.1
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
        parallel: bool = False,
        max_workers: Optional[int] = DEFAULT_MAX_WORKERS,
        timeout: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    ):
        """Initialize the reward function.

        Args:
            function_dict: Dictionary containing reward function parameters.
            unit_kwargs: Additional arguments for reward units.
            enable_validation: Whether to validate reward units.
            verbose: Enable detailed logging.
            parallel: Enable parallel calculation of reward units.
            max_workers: Maximum number of worker threads for parallel execution.
            timeout: Timeout in seconds for parallel execution.
        """
        super().__init__(
            component_type=ComponentType.REWARD_FUNCTION,
            component_name="RewardFunction",
        )

        # Configuration
        self.reward_function_dict = function_dict
        self.unit_kwargs = unit_kwargs or {}
        self.verbose = verbose
        self.parallel = parallel
        self.max_workers = max_workers
        self.timeout = timeout

        # State
        self.state = RewardState()
        self._validate_units = enable_validation
        self._lock = threading.Lock() if parallel else None

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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> None:
        """Calculate rewards using all reward units with optional parallel processing.

        Args:
            obs_dict: Dictionary of observations.
            simulation_state_container: Container for simulation state.
            **kwargs: Additional arguments passed to reward units.
        """
        self._validate_if_enabled(obs_dict)
        eligible_units = self._get_eligible_units()

        if not eligible_units:
            return

        execution_kwargs = self._prepare_execution_kwargs(
            obs_dict, simulation_state_container, **kwargs
        )
        self._execute_reward_units(eligible_units, execution_kwargs)

    def _validate_if_enabled(self, obs_dict: ObservationDict) -> None:
        """Validate reward unit requirements if validation is enabled."""
        if self._validate_units:
            with self._error_context("validating reward unit requirements"):
                validate_reward_units(
                    obs_dict,
                    {unit.__class__.__name__: unit for unit in self._reward_units},
                )

    def _get_eligible_units(self) -> List["RewardUnit"]:
        """Get eligible reward units based on safety state."""
        safe_dist_violation = self.state.info.get("safe_dist_violation", False)
        return (
            self._safe_dist_sensitive_units
            if safe_dist_violation
            else self._reward_units
        )

    def _prepare_execution_kwargs(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare arguments for reward unit execution."""
        obs_dict["simulation_state_container"] = simulation_state_container
        return obs_dict

    def _execute_reward_units(
        self, units: List["RewardUnit"], kwargs: Dict[str, Any]
    ) -> None:
        """Execute reward units using the appropriate strategy."""
        if len(units) == 1:
            self._execute_single_unit(units[0], kwargs)
        elif self.parallel and len(units) > 1:
            self._calculate_reward_parallel(units, kwargs)
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

    def _calculate_reward_parallel(
        self, reward_units: List["RewardUnit"], all_kwargs: Dict[str, Any]
    ) -> None:
        """Calculate rewards in parallel using ThreadPoolExecutor."""
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                results = self._execute_parallel_tasks(
                    executor, reward_units, all_kwargs
                )
                self._process_parallel_results(results, len(reward_units))
        except concurrent.futures.TimeoutError as e:
            self._handle_timeout_error(e)
        except Exception as e:
            self._handle_parallel_execution_error(e, reward_units, all_kwargs)

    def _execute_parallel_tasks(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        reward_units: List["RewardUnit"],
        all_kwargs: Dict[str, Any],
    ) -> List[Tuple[bool, str, Optional[Exception]]]:
        """Execute reward units in parallel and collect results."""
        future_to_unit = {
            executor.submit(self._execute_unit_safely, unit, all_kwargs): unit
            for unit in reward_units
        }

        completed_futures = concurrent.futures.as_completed(
            future_to_unit, timeout=self.timeout
        )

        return [future.result() for future in completed_futures]

    def _execute_unit_safely(
        self, unit: "RewardUnit", kwargs: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[Exception]]:
        """Execute a single reward unit and return success status."""
        try:
            unit(**kwargs)
            return True, unit.__class__.__name__, None
        except Exception as e:
            return False, unit.__class__.__name__, e

    def _process_parallel_results(
        self, results: List[Tuple[bool, str, Optional[Exception]]], total_units: int
    ) -> None:
        """Process results from parallel execution."""
        failed_units = [(name, exc) for success, name, exc in results if not success]

        for unit_name, exception in failed_units:
            if self.verbose and exception:
                self._report_error(
                    component_type=ComponentType.REWARD_UNIT,
                    component_name=unit_name,
                    severity=ErrorSeverity.WARNING,
                    message=str(exception),
                    error_type=type(exception).__name__,
                )

        if self.verbose and failed_units:
            self._report_warning(
                f"Parallel reward calculation: {len(failed_units)} units failed out of {total_units}"
            )

    def _handle_timeout_error(self, error: concurrent.futures.TimeoutError) -> None:
        """Handle timeout errors in parallel execution."""
        error_msg = f"Parallel reward calculation exceeded timeout of {self.timeout}s"
        self._report_error(error_msg, error_type="TimeoutError")
        raise TimeoutError(error_msg) from error

    def _handle_parallel_execution_error(
        self,
        error: Exception,
        reward_units: List["RewardUnit"],
        all_kwargs: Dict[str, Any],
    ) -> None:
        """Handle general errors in parallel execution with fallback."""
        error_msg = f"Parallel execution failed: {str(error)}"

        if self.verbose:
            self._report_warning(f"{error_msg}, falling back to sequential")
        else:
            self._report_error(error_msg, error_type=type(error).__name__)

        self._calculate_reward_sequential(reward_units, all_kwargs)

    def get_reward(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
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

        if self.parallel and self._lock:
            with self._lock:
                self._update_reward_and_overview(value, called_by)
        else:
            self._update_reward_and_overview(value, called_by)

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
            parallel=self.parallel,
            max_workers=self.max_workers,
            timeout=self.timeout,
        )
