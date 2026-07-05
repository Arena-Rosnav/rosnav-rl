"""Simple Observation Space Manager

Clean, lightweight manager for hierarchical observation spaces using SpaceFactory.
No bloat, just essential functionality.
"""

from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.logging import flush_and_log_errors
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.utils.validation import validate_observation_spaces

from .observation_space_factory import SpaceFactory
from .spaces.base_observation_space import BaseObservationSpace


class ObservationSpaceManager:
    """Simple hierarchical observation space manager using SpaceFactory."""

    def __init__(
        self,
        validate_observations=True,
        **kwargs,
    ):
        """Initialize simple observation space manager.

        Args:
            validate_observations: If True, validates observation keys before encoding.
        """
        self.spaces: OrderedDict[str, BaseObservationSpace] = (
            OrderedDict()
        )  # Preserve space order
        self.config = {}
        self.validate_observations = validate_observations
        self.space_factory = SpaceFactory

        # Performance optimization: cache required keys per space
        self._space_required_keys: Dict[str, List[str]] = {}
        # Reusable per-space kwargs dicts and encoded-output dict — cleared
        # and repopulated each call instead of rebuilt every tick (same
        # trick as RewardFunction._prepare_execution_kwargs).
        self._space_args_buffer: Dict[str, Dict[str, Any]] = {}
        self._encoded_buffer: "OrderedDict[str, Any]" = OrderedDict()

    def _auto_load_spacefactory(self, auto_load_spaces=True):
        """Automatically load SpaceFactory and register all spaces."""
        try:

            if auto_load_spaces:
                # Import all space modules to trigger registration
                try:
                    from .spaces import (  # noqa: F401
                        dynamics,
                        environment,
                        localization,
                        meta,
                        navigation,
                        perception,
                    )

                    print(
                        f"Auto-loaded {len(SpaceFactory.registry)} observation spaces"
                    )
                except ImportError as e:
                    print(f"Warning: Could not auto-load all spaces: {e}")

            return SpaceFactory
        except ImportError:
            print(
                "Warning: Could not import SpaceFactory. You'll need to provide it manually."
            )
            return None

    def load_configuration(self, config: Dict[str, Dict[str, Any]]):
        """Load configuration and instantiate spaces using SpaceFactory.

        Args:
            config: Configuration dict {space_name: {params}}
        """
        if self.space_factory is None:
            raise RuntimeError("SpaceFactory not available. Cannot load configuration.")

        self.config = config
        self.spaces = OrderedDict()
        self._space_args_buffer = {}
        self._encoded_buffer = OrderedDict()

        # Simply load spaces in config order (or alphabetically for consistency)
        space_names = sorted(config.keys())  # Alphabetical for consistency

        for space_name in space_names:
            space_config = config[space_name]
            space_instance = self.space_factory.instantiate(space_name, **space_config)
            self.spaces[space_name] = space_instance

            # Cache required keys for performance optimization
            self._space_required_keys[space_name] = list(space_instance.requires.keys())
            self._space_args_buffer[space_name] = {}

        print(f"Loaded {len(self.spaces)} spaces: {list(self.spaces.keys())}")

    def get_available_spaces(self) -> List[str]:
        """Get list of all available spaces from SpaceFactory registry."""
        if self.space_factory is None:
            return []
        return list(self.space_factory.registry.keys())

    def get_available_spaces_by_category(self) -> Dict[str, List[str]]:
        """Get available spaces organized by category from SpaceFactory."""
        if self.space_factory is None:
            return {}
        return self.space_factory.get_spaces_by_category()

    def encode_observation(self, observations: ObservationDict) -> Dict[str, Any]:
        """Encode observations for all loaded spaces with optional validation.

        Args:
            observations: Raw observation dictionary

        Returns:
            Encoded observations by space

        Raises:
            MissingObservationError: If validation enabled and required keys are missing
            Various space-specific exceptions: Propagated from individual spaces
        """
        if not self.spaces:
            return {}

        # Validate once then disable for performance — keys are static at runtime
        if self.validate_observations:
            validate_observation_spaces(observations, self.spaces)
            self.validate_observations = False

        return self._encode_sequential(observations)

    def _extract_space_args(
        self, space_name: str, observations: ObservationDict
    ) -> Dict[str, Any]:
        """Extract required arguments for a specific space from observations.

        Args:
            space_name: Name of the space
            observations: Full observation dictionary

        Returns:
            Dictionary containing only the arguments required by this space

        Note:
            Reuses a single mutable dict per space (cleared and repopulated
            each call) instead of allocating a fresh dict every tick.
        """
        args = self._space_args_buffer[space_name]
        args.clear()
        for key in self._space_required_keys[space_name]:
            args[key] = observations[key]
        return args

    def _encode_sequential(self, observations: ObservationDict) -> Dict[str, Any]:
        """Encode observations sequentially using new space interface.

        Args:
            observations: Raw observation dictionary

        Returns:
            Encoded observations by space

        Note:
            Space-specific errors are propagated to caller for proper error handling.
            Reuses a single mutable OrderedDict (cleared and repopulated each
            call) instead of allocating a fresh one every tick.
        """
        encoded = self._encoded_buffer
        encoded.clear()

        for space_name, space in self.spaces.items():

            # Call space with safe encoding that handles errors gracefully
            encoded[space_name] = space.safe_encode_observation(
                **self._extract_space_args(space_name, observations)
            )

        # If single space, return value directly
        if len(encoded) == 1:
            return list(encoded.values())[0]

        return dict(encoded)

    @property
    def space_list(self) -> List[BaseObservationSpace]:
        """Return the list of observation spaces."""
        return list(self.spaces.values())

    def reset_spaces(self) -> None:
        """Reset internal state of all observation spaces for a new episode.

        Also flushes the global error/warning collector (see
        rosnav_rl.utils.logging.error_logging) — this is the one call site
        every episode-reset path (training, inference, action server) already
        goes through, so it doubles as the system's log-flush cadence.
        """
        for space in self.spaces.values():
            space.reset()
        flush_and_log_errors()

    @property
    def observation_space(self) -> spaces.Space:
        """Return the combined observation space."""
        if not self.spaces:
            return spaces.Box(low=0, high=0, shape=(0,), dtype=np.float32)

        if len(self.spaces) == 1:
            # Single space - return directly
            return list(self.spaces.values())[0].get_gym_space()
        else:
            # Multiple spaces - combine as flat Dict
            space_dict = OrderedDict()
            for name, space in self.spaces.items():
                space_dict[name] = space.get_gym_space()
            return spaces.Dict(space_dict)

    @property
    def required_observations(self) -> Dict[str, List[str]]:
        """Return required keys for each space."""
        return self._space_required_keys
