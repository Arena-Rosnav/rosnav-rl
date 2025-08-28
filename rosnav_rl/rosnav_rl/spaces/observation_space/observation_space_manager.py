"""Simple Observation Space Manager

Clean, lightweight manager for hierarchical observation spaces using SpaceFactory.
No bloat, just essential functionality.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.utils.validation import validate_observation_spaces

from .observation_space_factory import SpaceFactory
from .spaces.base_observation_space import BaseObservationSpace


class ObservationSpaceManager:
    """Simple hierarchical observation space manager using SpaceFactory."""

    def __init__(
        self,
        parallel_encoding=True,
        validate_observations=True,
    ):
        """Initialize simple observation space manager.

        Args:
            space_factory: SpaceFactory instance to use. If None, will auto-import.
            auto_load_spaces: If True, automatically imports all space modules to register them.
            parallel_encoding: If True, encode observations in parallel threads.
            validate_observations: If True, validates observation keys before encoding.
        """
        self.spaces: OrderedDict[str, BaseObservationSpace] = (
            OrderedDict()
        )  # Preserve space order
        self.config = {}
        self.parallel_encoding = parallel_encoding
        self.validate_observations = validate_observations
        self.space_factory = SpaceFactory

        # Performance optimization: cache required keys per space
        self._space_required_keys: Dict[str, List[str]] = {}

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

        # Simply load spaces in config order (or alphabetically for consistency)
        space_names = sorted(config.keys())  # Alphabetical for consistency

        for space_name in space_names:
            space_config = config[space_name]
            space_instance = self.space_factory.instantiate(space_name, **space_config)
            self.spaces[space_name] = space_instance

            # Cache required keys for performance optimization
            self._space_required_keys[space_name] = list(space_instance.requires.keys())

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

        # Optional validation - can be disabled for performance
        if self.validate_observations:
            validate_observation_spaces(observations, self.spaces)

        if self.parallel_encoding and len(self.spaces) > 1:
            return self._encode_parallel(observations)
        else:
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
        """
        return {key: observations[key] for key in self._space_required_keys[space_name]}

    def _encode_sequential(self, observations: ObservationDict) -> Dict[str, Any]:
        """Encode observations sequentially using new space interface.

        Args:
            observations: Raw observation dictionary

        Returns:
            Encoded observations by space

        Note:
            Space-specific errors are propagated to caller for proper error handling.
        """
        encoded = OrderedDict()

        for space_name, space in self.spaces.items():

            # Call space with safe encoding that handles errors gracefully
            encoded[space_name] = space.safe_encode_observation(
                **self._extract_space_args(space_name, observations)
            )

        # If single space, return value directly
        if len(encoded) == 1:
            return list(encoded.values())[0]

        return dict(encoded)

    def _encode_parallel(self, observations: ObservationDict) -> Dict[str, Any]:
        """Encode observations in parallel using new space interface.

        Args:
            observations: Raw observation dictionary

        Returns:
            Encoded observations by space

        Note:
            Space-specific errors are propagated to caller for proper error handling.
        """

        def encode_single_space(space_name: str, space: BaseObservationSpace) -> tuple:
            """Encode a single space's observation in a separate thread."""
            # Call space with safe encoding that handles errors gracefully
            return (
                space_name,
                space.safe_encode_observation(
                    **self._extract_space_args(space_name, observations)
                ),
            )

        # Use ThreadPoolExecutor for parallel encoding
        with ThreadPoolExecutor(max_workers=min(len(self.spaces), 4)) as executor:
            # Submit all encoding tasks
            future_to_space = {
                executor.submit(encode_single_space, space_name, space): space_name
                for space_name, space in self.spaces.items()
            }

            # Collect results as they complete
            results: Dict[str, Any] = {}
            for future in as_completed(future_to_space):
                try:
                    space_name, result = future.result()
                    results[space_name] = result
                except Exception as e:
                    # Re-raise the first space error encountered
                    # This propagates space-specific errors to the caller
                    raise e

        # If single space, return value directly
        if len(results) == 1:
            return list(results.values())[0]

        return results

    @property
    def space_list(self) -> List[BaseObservationSpace]:
        """Return the list of observation spaces."""
        return list(self.spaces.values())

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
