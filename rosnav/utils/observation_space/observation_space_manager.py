from typing import Any, Dict, List, Union

import numpy as np
from gymnasium import spaces

from .spaces.base_observation_space import BaseObservationSpace
from .spaces.feature_maps.base_feature_map_space import BaseFeatureMapSpace
from .utils import stack_spaces


class ObservationSpaceManager:
    """
    A class that manages the observation spaces for a reinforcement learning agent.

    Args:
        space_list (List[Union[str, SPACE_INDEX]]): A list of space names or space indices.
        space_kwargs (Dict[str, Any]): Keyword arguments for configuring the observation spaces.
        frame_stacking (bool, optional): Whether to enable frame stacking. Defaults to False.
        flatten (bool, optional): Whether to flatten the observation spaces. Defaults to True.

    Attributes:
        space_list: The list of space names or space indices.
        observation_space: The combined observation space of all the individual spaces.

    Methods:
        add_observation_space: Add a new observation space to the manager.
        add_multiple_observation_spaces: Add multiple observation spaces to the manager.
        encode_observation: Encode an observation into a numpy array.
        get_space_container: Get the space container for a specific space.
        get_space_index: Get the space index for a specific space name.

    """

    def __init__(
        self,
        ns: str,
        space_list: List[BaseObservationSpace],
        space_kwargs: Dict[str, Any],
        frame_stacking: bool = False,
        flatten: bool = True,
    ) -> None:
        self._ns = ns
        self._space_cls_list = space_list
        self._space_kwargs = space_kwargs
        self._frame_stacking = frame_stacking
        self._flatten = flatten

        self._space_containers = self._setup_spaces()

    @property
    def space_list(self):
        return self._space_cls_list

    @property
    def observation_space(self) -> spaces.Box:
        return stack_spaces(
            *(
                self._space_containers[name].space
                for name in self._space_containers.keys()
            ),
            frame_stacking_enabled=self._frame_stacking,
        )

    def _setup_spaces(self):
        return {
            space_cls.name: space_cls(
                ns=self._ns, flatten=self._flatten, **self._space_kwargs
            )
            for space_cls in self._space_cls_list
        }

    def __getitem__(self, space: Union[str, BaseObservationSpace]) -> spaces.Box:
        space_name = space.name if issubclass(space, BaseObservationSpace) else space
        return self._space_containers[space_name.upper()].space

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        """
        Encode an observation into a numpy array.

        Args:
            observation (dict): The observation to encode.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The encoded observation.

        """
        _concatenated = np.concatenate(
            [
                self._space_containers[space.name].encode_observation(
                    observation, **kwargs
                )
                for space in self._space_cls_list
            ],
        )
        return (
            _concatenated
            if not self._frame_stacking
            else np.expand_dims(_concatenated, axis=0)
        )

    def get_space_container(
        self, space: Union[str, BaseObservationSpace]
    ) -> Union[BaseObservationSpace, BaseFeatureMapSpace]:
        """
        Get the space container for a specific space.

        Args:
            space_name (Union[str, SPACE_INDEX]): The name or index of the space.

        Returns:
            Union[BaseObservationSpace, BaseFeatureMapSpace]: The space container.

        """
        space_name = space.name if issubclass(space, BaseObservationSpace) else space
        return self._space_containers[space_name.upper()]
