"""Observation normalization functions and utilities."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Union
import numpy as np


class Normalizer(ABC):
    """Abstract base class for observation normalizers."""

    @abstractmethod
    def normalize(
        self, observation: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Normalize observation array using bounds."""
        pass

    @abstractmethod
    def denormalize(
        self, normalized_obs: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Reverse normalization (if applicable)."""
        pass


class MaxAbsScaler(Normalizer):
    """Max absolute scaling normalizer: scales to [-1, 1] range."""

    def normalize(
        self, observation: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Scale observation to [-1, 1] range."""
        denominator = high - low
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return (2 * (observation - low)) / denominator - 1

    def denormalize(
        self, normalized_obs: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Reverse max-abs scaling."""
        return ((normalized_obs + 1) * (high - low)) / 2 + low


class MinMaxScaler(Normalizer):
    """Min-max scaling normalizer: scales to [0, 1] range."""

    def normalize(
        self, observation: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Scale observation to [0, 1] range."""
        denominator = high - low
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return (observation - low) / denominator

    def denormalize(
        self, normalized_obs: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Reverse min-max scaling."""
        return normalized_obs * (high - low) + low


class StandardScaler(Normalizer):
    """Standard scaling normalizer: zero mean, unit variance."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def normalize(
        self, observation: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Standardize using bounds as rough mean/std estimates."""
        # Use bounds to estimate mean and std
        mean = (low + high) / 2
        std = (high - low) / 4  # Rough estimate assuming ~95% of data in bounds
        std = np.where(std == 0, self.epsilon, std)
        return (observation - mean) / std

    def denormalize(
        self, normalized_obs: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Reverse standardization."""
        mean = (low + high) / 2
        std = (high - low) / 4
        std = np.where(std == 0, self.epsilon, std)
        return normalized_obs * std + mean


class IdentityNormalizer(Normalizer):
    """No-op normalizer that returns input unchanged."""

    def normalize(
        self, observation: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Return observation unchanged."""
        return observation

    def denormalize(
        self, normalized_obs: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Return observation unchanged."""
        return normalized_obs


# Registry of available normalizers
NORMALIZERS: Dict[str, Union[Normalizer, Callable[[], Normalizer]]] = {
    "max_abs": MaxAbsScaler(),
    "min_max": MinMaxScaler(),
    "standard": StandardScaler,  # Callable to allow epsilon configuration
    "identity": IdentityNormalizer(),
    "none": IdentityNormalizer(),
}


def get_normalizer(name: str, **kwargs) -> Normalizer:
    """Get normalizer instance by name.

    Args:
        name: Name of normalizer ('max_abs', 'min_max', 'standard', 'identity', 'none')
        **kwargs: Additional arguments for normalizer construction

    Returns:
        Normalizer instance

    Raises:
        ValueError: If normalizer name is not recognized
    """
    if name not in NORMALIZERS:
        available = list(NORMALIZERS.keys())
        raise ValueError(f"Unknown normalizer '{name}'. Available: {available}")

    normalizer = NORMALIZERS[name]

    # If it's a class/callable, instantiate it
    if callable(normalizer) and not isinstance(normalizer, Normalizer):
        return normalizer(**kwargs)

    return normalizer
