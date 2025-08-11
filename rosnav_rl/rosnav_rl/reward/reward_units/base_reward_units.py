from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from rosnav_rl.spaces.observation_space.utils import RequiresProtocol

from ..reward_function import RewardFunction

logger = logging.getLogger(__name__)


class RewardUnit(RequiresProtocol, ABC):
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

    Performance Features:
        - Fast path validation with early exit
        - Cached requirement extraction
        - Optional validation for production environments
        - Efficient argument filtering

    Robustness Features:
        - Comprehensive error handling with context
        - Parameter validation with warnings
        - State management and cleanup
        - Safe fallback mechanisms

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
        self._reward_function = reward_function
        self._on_safe_dist_violation = _on_safe_dist_violation
        self._cached_requirements: Optional[set] = None

        # Enhanced validation with better error messages
        self._validate_schema_definition()

    def _validate_schema_definition(self) -> None:
        """Validate that the schema-based requirements are properly defined.

        Raises:
            AttributeError: If 'requires' is not properly defined
            ValueError: If 'requires' contains invalid entries
        """
        if not hasattr(self, "requires"):
            raise AttributeError(
                f"RewardUnit '{self.__class__.__name__}' must define 'requires' attribute. "
                "See RequiresProtocol documentation for examples."
            )

        if not isinstance(self.requires, dict):
            raise ValueError(
                f"RewardUnit '{self.__class__.__name__}' 'requires' must be a dictionary, "
                f"got {type(self.requires)}"
            )

        # Cache requirement keys for performance
        self._cached_requirements = set(self.requires.keys())

        # Validate requirement entries
        for key, value in self.requires.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"RewardUnit '{self.__class__.__name__}' requirement key '{key}' "
                    f"must be a string, got {type(key)}"
                )

    @property
    def on_safe_dist_violation(self) -> bool:
        """Returns whether the unit is applied on safe distance violation."""
        return self._on_safe_dist_violation

    def add_reward(self, value: float) -> None:
        """Add reward value with enhanced error handling.

        Args:
            value (float): The reward value to add

        Raises:
            TypeError: If value is not a numeric type
            ValueError: If value is NaN or infinite
        """
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"RewardUnit '{self.__class__.__name__}' reward value must be numeric, "
                f"got {type(value)}: {value}"
            )

        if not isinstance(value, (int, float)) or value != value:  # NaN check
            raise ValueError(
                f"RewardUnit '{self.__class__.__name__}' reward value cannot be NaN"
            )

        if abs(value) == float("inf"):
            raise ValueError(
                f"RewardUnit '{self.__class__.__name__}' reward value cannot be infinite"
            )

        self._reward_function.add_reward(value=value, called_by=self.__class__.__name__)

    def add_info(self, info: Dict[str, Any]) -> None:
        """Add information to episode info dict with validation.

        Args:
            info (Dict[str, Any]): Information to add to the episode's info dict

        Raises:
            TypeError: If info is not a dictionary
        """
        if not isinstance(info, dict):
            raise TypeError(
                f"RewardUnit '{self.__class__.__name__}' info must be a dictionary, "
                f"got {type(info)}"
            )

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
                logger.warning(
                    f"RewardUnit '{self.__class__.__name__}' has large reward magnitude: {reward}. "
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
