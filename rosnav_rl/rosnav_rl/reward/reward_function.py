from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import concurrent.futures
import threading


from rosnav_rl.cfg.reward import RewardFunctionDict
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.utils.validation import validate_reward_units

if TYPE_CHECKING:
    from .reward_units.base_reward_units import RewardUnit


class RewardState:
    """Container for reward calculation state."""

    def __init__(self):
        self.current_reward: float = 0.0
        self.info: Dict[str, Any] = {}
        self.reward_overview: Dict[str, float] = {}


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
        validate_units: bool = True,
        verbose: bool = False,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = 5.0,
    ):
        """
        Initialize the reward function.

        Args:
            function_dict (Dict[str, Union[str, float, int]]): Dictionary containing reward function parameters.
            unit_kwargs (Optional[Dict[str, Any]]): Additional arguments for reward units. Defaults to None.
            validate_units (bool): Whether to validate reward units. Defaults to True.
            verbose (bool): Enable detailed logging. Defaults to False.
            parallel (bool): Enable parallel calculation of reward units. Defaults to False.
            max_workers (Optional[int]): Maximum number of worker threads for parallel execution.
                                       Defaults to None (uses system default).
            timeout (Optional[float]): Timeout in seconds for parallel execution. Defaults to 5.0.
        """
        self.reward_function_dict = function_dict
        self.unit_kwargs = unit_kwargs or {}
        self.verbose = verbose
        self.parallel = parallel
        self.max_workers = max_workers
        self.timeout = timeout
        self.state = RewardState()
        self._validate_units = validate_units
        self._reward_units: List["RewardUnit"] = []
        self._lock = threading.Lock() if parallel else None
        self._create_reward_units()

    def _create_reward_units(self) -> None:
        """Create reward unit instances from configuration."""
        import rosnav_rl.reward as rew_pkg

        self._reward_units = [
            self._create_reward_unit(rew_pkg.RewardUnitFactory, unit_name, params)
            for unit_name, params in self.reward_function_dict.items()
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
            return unit_class(reward_function=self, **self.unit_kwargs, **params)
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
        """Calculate rewards using all reward units with optional parallel processing.

        Args:
            obs_dict: Dictionary of observations.
            simulation_state_container: Container for simulation state.
            **kwargs: Additional arguments passed to reward units.

        Raises:
            KeyError: If required observations are missing (with detailed error context)
            RuntimeError: If reward unit execution fails (with unit context)
            ValueError: If reward values are invalid (NaN, infinite, etc.)
            TimeoutError: If parallel execution exceeds timeout
        """
        # Optional validation of reward unit requirements
        # Only validate in verbose mode to avoid performance overhead in production
        if self._validate_units:
            try:
                validate_reward_units(
                    obs_dict,
                    {unit.__class__.__name__: unit for unit in self._reward_units},
                )
            except Exception as e:
                raise RuntimeError(
                    f"Reward unit validation failed: {str(e)}. "
                    "Check that all required observations are available in obs_dict."
                ) from e

        # Prepare all available arguments for reward units
        # Include commonly needed arguments in a structured way
        all_kwargs = {
            "obs_dict": obs_dict,
            "simulation_state_container": simulation_state_container,
            **obs_dict,  # Flatten observations for direct access
            **kwargs,
        }

        # Filter reward units that should be processed
        eligible_units = [
            unit
            for unit in self._reward_units
            if not self._skip_on_safe_dist_violation(unit)
        ]

        # Choose execution strategy based on configuration
        if self.parallel and len(eligible_units) > 1:
            self._calculate_reward_parallel(eligible_units, all_kwargs)
        else:
            self._calculate_reward_sequential(eligible_units, all_kwargs)

    def _calculate_reward_sequential(
        self, reward_units: List["RewardUnit"], all_kwargs: Dict[str, Any]
    ) -> None:
        """Calculate rewards sequentially (original behavior).

        Args:
            reward_units: List of reward units to process.
            all_kwargs: Arguments to pass to reward units.
        """
        for reward_unit in reward_units:
            try:
                reward_unit(**all_kwargs)
            except Exception as e:
                if self.verbose:
                    print(
                        f"Warning: Reward unit {reward_unit.__class__.__name__} failed: {e}"
                    )
                # Continue with other units even if one fails

    def _calculate_reward_parallel(
        self, reward_units: List["RewardUnit"], all_kwargs: Dict[str, Any]
    ) -> None:
        """Calculate rewards in parallel using ThreadPoolExecutor.

        Args:
            reward_units: List of reward units to process.
            all_kwargs: Arguments to pass to reward units.

        Raises:
            TimeoutError: If execution exceeds configured timeout.
            RuntimeError: If parallel execution fails critically.
        """

        def execute_reward_unit(
            unit: "RewardUnit",
        ) -> Tuple[bool, str, Optional[Exception]]:
            """Execute a single reward unit and return success status.

            Returns:
                Tuple of (success, unit_name, exception_if_any)
            """
            try:
                unit(**all_kwargs)
                return True, unit.__class__.__name__, None
            except Exception as e:
                return False, unit.__class__.__name__, e

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all reward unit tasks
                future_to_unit = {
                    executor.submit(execute_reward_unit, unit): unit
                    for unit in reward_units
                }

                # Wait for completion with timeout
                completed_futures = concurrent.futures.as_completed(
                    future_to_unit, timeout=self.timeout
                )

                # Process results
                failed_units = []
                for future in completed_futures:
                    success, unit_name, exception = future.result()
                    if not success:
                        failed_units.append((unit_name, exception))
                        if self.verbose:
                            print(
                                f"Warning: Reward unit {unit_name} failed: {exception}"
                            )

                # Log summary if verbose
                if self.verbose and failed_units:
                    print(
                        f"Parallel reward calculation: {len(failed_units)} units failed out of {len(reward_units)}"
                    )

        except concurrent.futures.TimeoutError as e:
            raise TimeoutError(
                f"Parallel reward calculation exceeded timeout of {self.timeout}s"
            ) from e
        except Exception as e:
            if self.verbose:
                print(f"Parallel execution failed, falling back to sequential: {e}")
            # Fallback to sequential execution
            self._calculate_reward_sequential(reward_units, all_kwargs)

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

        if self.verbose:
            self._log_reward_overview()

        return self.state.current_reward, self.state.info

    def add_reward(self, value: float, **kwargs) -> None:
        """
        Add a reward value and track its source.

        Args:
            value: Reward value to add
            **kwargs: Additional metadata about the reward
        """
        if self.parallel:
            with self._lock:
                self.state.current_reward += value
                if called_by := kwargs.get("called_by"):
                    self.state.reward_overview[called_by] = value
        else:
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
            verbose=self.verbose,
            parallel=self.parallel,
            max_workers=self.max_workers,
            timeout=self.timeout,
        )
