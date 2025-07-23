"""Simple Observation Space Manager

Clean, lightweight manager for hierarchical observation spaces using SpaceFactory.
No bloat, just essential functionality.
"""

from typing import Any, Dict, List
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from gymnasium import spaces

try:
    from rosnav_rl.utils.type_aliases import ObservationDict
except ImportError:
    # Fallback for when type aliases aren't available
    from typing import Dict as ObservationDict
from .spaces.base_observation_space import BaseObservationSpace


class ObservationSpaceManager:
    """Simple hierarchical observation space manager using SpaceFactory."""

    def __init__(
        self,
        space_factory=None,
        auto_load_spaces=True,
        parallel_encoding=True,
    ):
        """Initialize simple observation space manager.

        Args:
            space_factory: SpaceFactory instance to use. If None, will auto-import.
            auto_load_spaces: If True, automatically imports all space modules to register them.
            parallel_encoding: If True, encode observations in parallel threads.
        """
        self.spaces = OrderedDict()  # Preserve space order
        self.config = {}
        self.parallel_encoding = parallel_encoding
        self.space_factory = space_factory

        # Auto-load SpaceFactory and register all spaces
        if self.space_factory is None:
            self.space_factory = self._auto_load_spacefactory(auto_load_spaces)

    def _auto_load_spacefactory(self, auto_load_spaces=True):
        """Automatically load SpaceFactory and register all spaces."""
        try:
            from .observation_space_factory import SpaceFactory

            if auto_load_spaces:
                # Import all space modules to trigger registration
                try:
                    from .spaces import (  # noqa: F401
                        localization,
                        perception,
                        navigation,
                        dynamics,
                        environment,
                        meta,
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
        """Encode observations for all loaded spaces.
        Can use parallel encoding for better performance.

        Args:
            observations: Raw observation dictionary

        Returns:
            Encoded observations by space
        """
        if not self.spaces:
            return {}

        if self.parallel_encoding and len(self.spaces) > 1:
            return self._encode_parallel(observations)
        else:
            return self._encode_sequential(observations)

    def _encode_sequential(self, observations: ObservationDict) -> Dict[str, Any]:
        """Encode observations sequentially."""
        encoded = OrderedDict()

        for space_name, space in self.spaces.items():
            try:
                encoded[space_name] = space.encode_observation(observations)
            except Exception as e:
                print(f"Error encoding space '{space_name}': {e}")
                # Provide fallback zero observation
                gym_space = space.get_gym_space()
                encoded[space_name] = np.zeros_like(gym_space.sample())

        # If single space, return value directly
        if len(encoded) == 1:
            return list(encoded.values())[0]

        return dict(encoded)

    def _encode_parallel(self, observations: ObservationDict) -> Dict[str, Any]:
        """Encode observations in parallel using ThreadPoolExecutor."""

        def encode_single_space(space_name, space):
            try:
                return space_name, space.encode_observation(observations), None
            except Exception as e:
                return space_name, None, e

        encoded = OrderedDict()

        # Use ThreadPoolExecutor for parallel encoding
        with ThreadPoolExecutor(max_workers=min(len(self.spaces), 4)) as executor:
            # Submit all encoding tasks
            future_to_space = {
                executor.submit(encode_single_space, space_name, space): space_name
                for space_name, space in self.spaces.items()
            }

            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_space):
                space_name, result, error = future.result()
                if error is not None:
                    print(f"Error encoding space '{space_name}': {error}")
                    # Provide fallback zero observation
                    gym_space = self.spaces[space_name].get_gym_space()
                    results[space_name] = np.zeros_like(gym_space.sample())
                else:
                    results[space_name] = result

            # Rebuild in original order
            for space_name in self.spaces.keys():
                encoded[space_name] = results[space_name]

        # If single space, return value directly
        if len(encoded) == 1:
            return list(encoded.values())[0]

        return dict(encoded)

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
