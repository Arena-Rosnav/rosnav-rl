from typing import Any, Dict, List, Union

import numpy as np
from gymnasium import spaces

from .space_index import SPACE_INDEX
from .spaces.base_observation_space import BaseObservationSpace
from .utils import stack_spaces


class ObservationSpaceManager:
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

    def add_observation_space(self, space: SPACE_INDEX):
        assert isinstance(space, SPACE_INDEX), "Invalid Space Type"
        assert (
            space not in self._spacelist
        ), f"{space}-ObservationSpace was already specified!"
        self._spacelist.append(space)
        self._setup_spaces()

    def add_multiple_observation_spaces(self, space_list: List[SPACE_INDEX]):
        for space in space_list:
            self.add_observation_space(space)

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        return np.concatenate(
            [
                self._space_containers[space.name].encode_observation(
                    observation, **kwargs
                )
                for space in self._spacelist
            ],
            axis=0 if self._frame_stacking else 1,
        )

    @staticmethod
    def get_space_index(space_name: Union[str, SPACE_INDEX]) -> SPACE_INDEX:
        if isinstance(space_name, SPACE_INDEX):
            return space_name
        return SPACE_INDEX[space_name.upper()]

    def _setup_spaces(self):
        self._space_containers = self._generate_space_container(
            self._spacelist, self._space_kwargs
        )

    def _generate_space_container(
        self,
        space_list: List[Union[str, SPACE_INDEX]],
        space_kwargs: Dict[str, Any],
    ) -> Dict[str, BaseObservationSpace]:
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
