from typing import Any, Dict, List, Union

import numpy as np
from gymnasium import spaces

from .space_index import SPACE_INDEX
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
        space_list: List[Union[str, SPACE_INDEX]],
        space_kwargs: Dict[str, Any],
        frame_stacking: bool = False,
        flatten: bool = True,
    ) -> None:
        self._spacelist = space_list
        self._space_kwargs = space_kwargs
        self._frame_stacking = frame_stacking
        self._flatten = flatten

        self._setup_spaces()

    @property
    def space_list(self):
        return self._spacelist

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
        self._space_containers = self._generate_space_container(
            self._spacelist, self._space_kwargs
        )

    def _generate_space_container(
        self,
        space_list: List[Union[str, SPACE_INDEX]],
        space_kwargs: Dict[str, Any],
    ) -> Dict[str, Union[BaseObservationSpace, BaseFeatureMapSpace]]:
        """
        Generates a container for observation spaces.

        Args:
            space_list (List[Union[str, SPACE_INDEX]]): A list of space names or space indices.
            space_kwargs (Dict[str, Any]): Additional keyword arguments for space initialization.

        Returns:
            Dict[str, Union[BaseObservationSpace, BaseFeatureMapSpace]]: A dictionary containing the generated observation spaces.
        """
        space_list = [
            ObservationSpaceManager.get_space_index(space) for space in space_list
        ]
        return {
            space_index.name: space_index.value(flatten=self._flatten, **space_kwargs)
            for space_index in space_list
        }

    def __getitem__(self, space_name: Union[str, SPACE_INDEX]) -> spaces.Box:
        if isinstance(space_name, SPACE_INDEX):
            space_name = space_name.name
        return self._space_containers[space_name.upper()].space

    def add_observation_space(self, space: SPACE_INDEX):
        """
        Add a new observation space to the manager.

        Args:
            space (SPACE_INDEX): The space index of the observation space.

        Raises:
            AssertionError: If the space type is invalid or if the space was already specified.

        """
        assert isinstance(space, SPACE_INDEX), "Invalid Space Type"
        assert (
            space not in self._spacelist
        ), f"{space}-ObservationSpace was already specified!"
        self._spacelist.append(space)
        self._setup_spaces()

    def add_multiple_observation_spaces(self, space_list: List[SPACE_INDEX]):
        """
        Add multiple observation spaces to the manager.

        Args:
            space_list (List[SPACE_INDEX]): A list of space indices.

        """
        for space in space_list:
            self.add_observation_space(space)

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
                for space in self._spacelist
            ],
        )
        return (
            _concatenated
            if not self._frame_stacking
            else np.expand_dims(_concatenated, axis=0)
        )

    def get_space_container(
        self, space_name: Union[str, SPACE_INDEX]
    ) -> Union[BaseObservationSpace, BaseFeatureMapSpace]:
        """
        Get the space container for a specific space.

        Args:
            space_name (Union[str, SPACE_INDEX]): The name or index of the space.

        Returns:
            Union[BaseObservationSpace, BaseFeatureMapSpace]: The space container.

        """
        if isinstance(space_name, SPACE_INDEX):
            space_name = space_name.name
        return self._space_containers[space_name.upper()]

    @staticmethod
    def get_space_index(space_name: Union[str, SPACE_INDEX]) -> SPACE_INDEX:
        """
        Get the space index for a specific space name.

        Args:
            space_name (Union[str, SPACE_INDEX]): The name or index of the space.

        Returns:
            SPACE_INDEX: The space index.

        """
        if isinstance(space_name, SPACE_INDEX):
            return space_name
        return SPACE_INDEX[space_name.upper()]
