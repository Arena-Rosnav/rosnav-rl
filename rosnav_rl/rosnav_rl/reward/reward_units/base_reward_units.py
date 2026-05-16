from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import numpy as np

from rosnav_rl.utils.validation import RequiresProtocol
from rosnav_rl.utils.logging import ErrorReportingMixin, ComponentType

from ..reward_function import RewardFunction

logger = logging.getLogger(__name__)


class RewardUnit(ErrorReportingMixin, ABC, RequiresProtocol):
    """
    Enhanced RewardUnit base class with schema-based validation and improved robustness.

    This class implements the unified RequiresProtocol for consistent validation across
    observation spaces, generators, and reward units. It provides efficient schema-based
    validation, robust error handling, and performance optimizations.

    Schema-Based Requirements:
        'requires' is a mapping from logical input names (str) to schema-based type annotations
        (see rosnav_rl.observations.utils.types). This enables schema-driven configuration,
        validation, and documentation of reward unit dependencies.

        Example:
            requires = {
                "laser_safety": SafetyStatus,
                "dist_angle_to_goal": DistanceAngleMetrics,
                "robot_action": RobotVelocity,
            }

    Attributes:
        requires (Dict[str, Any]): Schema-based requirement mapping (class attribute)
        required_observation_units (List[BaseUnit]): Legacy observation units (deprecated)
        _reward_function (RewardFunction): The reward function holding this unit
        _on_safe_dist_violation (bool): Whether unit applies on safe distance violation
        _cached_requirements (Optional[Set[str]]): Cached requirement keys for performance
    """

    # Schema-based requirements - subclasses must override
    requires: Dict[str, Any] = {}

    def __init__(
        self,
        reward_function: RewardFunction,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the RewardUnit with enhanced validation and caching.

        Args:
            reward_function (RewardFunction): The RewardFunction instance holding this unit
            _on_safe_dist_violation (bool, optional): Whether unit applies on safe distance violation. Defaults to True.
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Raises:
            AttributeError: If required schema is not properly defined
            ValueError: If initialization parameters are invalid
        """
        super().__init__(
            component_type=ComponentType.REWARD_UNIT,
            component_name=self.__class__.__name__,
            *args,
            **kwargs,
        )

        self._reward_function = reward_function
        self._on_safe_dist_violation = _on_safe_dist_violation
        self._cached_requirements: Optional[set] = None

        # Enhanced validation with better error messages
        self._validate_schema_definition()

    def _validate_schema_definition(self) -> None:
        """Validate that the schema-based requirements are properly defined.

        Uses unified error reporting instead of raising exceptions directly.
        """
        if not hasattr(self, "requires"):
            self._report_critical(
                "Must define 'requires' attribute. See RequiresProtocol documentation for examples."
            )
            return

        if not isinstance(self.requires, dict):
            self._report_critical(
                f"'requires' must be a dictionary, got {type(self.requires)}"
            )
            return

        # Cache requirement keys for performance
        self._cached_requirements = set(self.requires.keys())

        # Validate requirement entries
        for key, value in self.requires.items():
            if not isinstance(key, str):
                self._report_error(
                    f"Requirement key '{key}' must be a string, got {type(key)}"
                )

    @property
    def on_safe_dist_violation(self) -> bool:
        """Returns whether the unit is applied on safe distance violation."""
        return self._on_safe_dist_violation

    def add_reward(self, value: float) -> None:
        """Add reward value with enhanced error handling.

        Args:
            value (float): The reward value to add
        """
        if not isinstance(value, (int, float, np.floating, np.integer)):
            self._report_error(
                f"Reward value must be numeric, got {type(value)}: {value}",
                error_type="TypeError",
            )
            return

        if value != value:  # NaN check
            self._report_error("Reward value cannot be NaN", error_type="ValueError")
            return

        if abs(value) == float("inf"):
            self._report_error(
                "Reward value cannot be infinite", error_type="ValueError"
            )
            return

        self._reward_function.add_reward(value=value, called_by=self.__class__.__name__)

    def add_info(self, info: Dict[str, Any]) -> None:
        """Add information to episode info dict with validation.

        Args:
            info (Dict[str, Any]): Information to add to the episode's info dict
        """
        if not isinstance(info, dict):
            self._report_error(
                f"Info must be a dictionary, got {type(info)}", error_type="TypeError"
            )
            return

        self._reward_function.add_info(info=info)

    def check_parameters(self, *args: Any, **kwargs: Any) -> None:
        """Enhanced parameter validation with specific checks.

        Override this method in subclasses to implement unit-specific validation logic.
        Base implementation provides common validation patterns.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # Common parameter checks
        if hasattr(self, "_reward_value"):
            reward = getattr(self, "_reward_value", None)
            if reward is not None and abs(reward) > 100:
                self._report_warning(
                    f"Has large reward magnitude: {reward}. "
                    "Consider scaling rewards for better training stability."
                )

    def reset(self) -> None:
        """Reset unit state with enhanced cleanup.

        Override this method in subclasses to implement unit-specific reset logic.
        Base implementation handles common cleanup patterns.
        """
        # Clear any cached state that should reset between episodes
        # Subclasses should call super().reset() and add their own cleanup
        pass

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """Process observations and update rewards with schema-based arguments.

        Subclasses must implement this method with explicit, schema-typed keyword arguments
        matching the 'requires' specification. This ensures type safety and clear interfaces.

        Args:
            **kwargs: Schema-validated keyword arguments matching 'requires'

        Example Implementation:
            ```python
            class MyRewardUnit(RewardUnit):
                requires = {
                    "laser_safety": SafetyStatus,
                    "dist_angle_to_goal": DistanceAngleMetrics,
                }

                def __call__(self, laser_safety: SafetyStatus, dist_angle_to_goal: DistanceAngleMetrics):
                    if laser_safety.violation:
                        self.add_reward(-10.0)
                    # ... rest of implementation
            ```

        Raises:
            NotImplementedError: If not implemented in subclass
        """
        raise NotImplementedError(
            f"RewardUnit '{self.__class__.__name__}' must implement __call__ method "
            "with schema-based keyword arguments matching 'requires'."
        )
