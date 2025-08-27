from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Any

import numpy as np
from gymnasium import spaces

from ..normalization import get_normalizer
from rosnav_rl.utils.validation import RequiresProtocol
from rosnav_rl.utils.logging import ErrorReportingMixin, ComponentType

ERROR_MESSAGE_SUFFIX = " ---> Using null observation."


class BaseObservationSpace(ErrorReportingMixin, ABC, RequiresProtocol):
    """Base class for observation spaces in reinforcement learning environments.

    This class defines the interface for observation spaces and provides common
    functionality for normalization, validation, and error handling of observations.

    Class Attributes:
        name (str): The name of the observation space.
        requires (Dict[str, Any]): Schema-based requirements defining the data sources
            needed by this observation space. Maps data source names to type annotations
            with rich metadata including descriptions, shapes, units, and constraints.

    Example:
        class LaserObservationSpace(BaseObservationSpace):
            name = "laser_scan"
            requires = {
                "front_laser": LidarRanges,
                "robot_pose": Pose2D,
            }
    """

    name: ClassVar[str]
    requires: ClassVar[Dict[str, Any]] = {}

    def __init__(
        self,
        normalize: bool = False,
        normalizer: str = "max_abs",
        **kwargs,
    ) -> None:
        """Initialize the observation space.

        Args:
            normalize: Whether to normalize observations.
            normalizer: Name of normalizer ("max_abs", "min_max", "standard", "identity").
            **kwargs: Additional arguments passed to the normalizer.
        """
        # Initialize parent mixins
        super().__init__(
            component_type=ComponentType.OBSERVATION_SPACE,
            component_name=self.name,
        )

        self._space = self.get_gym_space()
        self._normalize = normalize
        self._normalizer = self._setup_normalizer(normalize, normalizer, **kwargs)

        # Store configuration for debugging and serialization
        self._config = {
            "normalize": normalize,
            "normalizer": normalizer,
            **kwargs,
        }

    def __repr__(self) -> str:
        """Return string representation of the observation space."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    # ==========================================
    # Properties
    # ==========================================

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration parameters."""
        return self._config.copy()

    @property
    def space(self) -> spaces.Space:
        """Get the gymnasium Space object."""
        return self._space

    @property
    def shape(self) -> tuple:
        """Get the shape of the observation space."""
        return self._space.shape

    # ==========================================
    # Abstract Methods
    # ==========================================

    @abstractmethod
    def get_gym_space(self) -> spaces.Space:
        """Define and return the gymnasium Space object.

        Returns:
            spaces.Space: The gymnasium space representing this observation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_gym_space()"
        )

    @abstractmethod
    def encode_observation(self, *args, **kwargs) -> np.ndarray:
        """Encode the observation into a numpy array.

        Args:
            *args: Positional arguments specific to the observation space.
            **kwargs: Keyword arguments specific to the observation space.

        Returns:
            np.ndarray: The encoded observation array.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement encode_observation()"
        )

    # ==========================================
    # Public API Methods
    # ==========================================

    def safe_encode_observation(self, *args, **kwargs) -> np.ndarray:
        """Safely encode observations with error handling and null fallback.

        This method wraps encode_observation() to catch errors and return a properly
        shaped null array when encoding fails, ensuring the system continues to function.

        Returns:
            np.ndarray: Encoded observation or null array if encoding fails.
        """
        try:
            result = self.encode_observation(*args, **kwargs)
            if result is not None:
                return result
            self._report_warning(
                f"encode_observation() returned None. {ERROR_MESSAGE_SUFFIX}"
            )
        except KeyError as e:
            self._report_error(
                f"Missing observation data key {e}. {ERROR_MESSAGE_SUFFIX}",
                error_type="KeyError",
            )
        except (ValueError, TypeError, AttributeError) as e:
            self._report_error(
                f"Error during encoding: {e}. {ERROR_MESSAGE_SUFFIX}",
                error_type=type(e).__name__,
            )
        except Exception as e:
            self._report_error(
                f"Unexpected error during encoding: {e}. {ERROR_MESSAGE_SUFFIX}",
                error_type=type(e).__name__,
            )

        return self._create_null_observation()

    # ==========================================
    # Private Helper Methods
    # ==========================================

    def _create_null_observation(self) -> np.ndarray:
        """Create a null observation array with the correct shape.

        Returns:
            np.ndarray: Zero-filled array matching the gym space specification.
        """
        if isinstance(self._space, spaces.Box):
            return np.zeros(self._space.shape, dtype=np.float32)
        elif isinstance(self._space, spaces.Discrete):
            return np.array([0], dtype=np.int32)
        elif isinstance(self._space, spaces.Dict):
            return self._create_null_dict_observation()
        else:
            self._report_warning(
                f"Unsupported observation space type: {type(self._space)}. "
                "Returning zero-filled array."
            )
            # Fallback for unknown space types
            return np.array([0.0], dtype=np.float32)

    def _create_null_dict_observation(self) -> Dict[str, np.ndarray]:
        """Create a null observation for Dict spaces."""
        null_dict = {}
        for key, subspace in self._space.spaces.items():
            if isinstance(subspace, spaces.Box):
                null_dict[key] = np.zeros(subspace.shape, dtype=np.float32)
            elif isinstance(subspace, spaces.Discrete):
                null_dict[key] = np.array([0], dtype=np.int32)
            else:
                null_dict[key] = np.array([0.0], dtype=np.float32)
        return null_dict

    def _setup_normalizer(self, normalize: bool, normalizer_name: str, **kwargs):
        """Set up the normalizer instance."""
        if not normalize:
            return get_normalizer("identity")

        try:
            return get_normalizer(normalizer_name, **kwargs)
        except ValueError as e:
            self._report_warning(
                f"Error setting up normalizer '{normalizer_name}': {e}. Using identity normalizer."
            )
            return get_normalizer("identity")

    def _apply_normalization(self, observation_arr: np.ndarray) -> np.ndarray:
        """Apply normalization to observation array if enabled.

        Args:
            observation_arr: The observation array to normalize.

        Returns:
            np.ndarray: Normalized array if normalization is enabled, otherwise unchanged.
        """
        if (
            self._normalize
            and hasattr(self._space, "low")
            and hasattr(self._space, "high")
        ):
            return self._normalizer.normalize(
                observation_arr, self._space.low, self._space.high
            )
        return observation_arr

    def _validate_observation(self, observation_arr: np.ndarray) -> np.ndarray:
        """Validate and fix observation array issues.

        Args:
            observation_arr: The observation array to validate.

        Returns:
            np.ndarray: Validated and potentially corrected observation array.
        """
        if (
            not np.isfinite(observation_arr).all()
            or not np.isreal(observation_arr).all()
        ):
            self._report_warning(
                "Invalid observation array detected, replacing with zeros."
            )
            return np.zeros_like(observation_arr, dtype=np.float32)
        return observation_arr

    # ==========================================
    # Decorators
    # ==========================================

    @staticmethod
    def apply_normalization(func):
        """Decorator to apply normalization after observation encoding.

        Usage:
            @BaseObservationSpace.apply_normalization
            def encode_observation(self, ...):
                return observation_array
        """

        def wrapper(self: "BaseObservationSpace", *args, **kwargs) -> np.ndarray:
            observation_arr = func(self, *args, **kwargs)
            return self._apply_normalization(observation_arr)

        return wrapper

    @staticmethod
    def check_dtype(func):
        """Decorator to validate observation array data types.

        Usage:
            @BaseObservationSpace.check_dtype
            def encode_observation(self, ...):
                return observation_array
        """

        def wrapper(self: "BaseObservationSpace", *args, **kwargs) -> np.ndarray:
            observation_arr = func(self, *args, **kwargs)
            return self._validate_observation(observation_arr)

        return wrapper
