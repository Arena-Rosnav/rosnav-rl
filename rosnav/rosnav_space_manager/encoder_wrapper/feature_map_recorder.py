from typing import Union, List, Dict

import numpy as np

from rosnav.rosnav_space_manager.base_space_encoder import BaseSpaceEncoder
from rosnav.rosnav_space_manager.encoder_wrapper.base_encoder_wrapper import (
    BaseEncoderWrapper,
)

from rosnav.utils.observation_space.space_index import SPACE_INDEX
from rosnav.utils.observation_space.spaces.feature_maps.base_feature_map_space import (
    BaseFeatureMapSpace,
)

_FeatureMapList = List[np.ndarray]


class FeatureMapRecorderWrapper(BaseEncoderWrapper):
    def __init__(
        self,
        encoder: BaseSpaceEncoder,
        record_features: List[SPACE_INDEX] = None,
        save_every_x_obs: int = 10,
    ) -> None:
        super().__init__(encoder)
        self._record_features = record_features if record_features else []
        self._recordings: Dict[str, _FeatureMapList] = {}
        self._save_every_x_obs = save_every_x_obs

        self.__iter = 0

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        encoded_obs_list = []

        for space in self._encoder.observation_space_manager._spacelist:
            space_container = self._encoder.observation_space_manager._space_containers[
                space.name
            ]
            encoded_obs = space_container.encode_observation(observation, **kwargs)
            encoded_obs_list.append(encoded_obs)

            __save_every_map = len(self._record_features) == 0 and isinstance(
                space_container, BaseFeatureMapSpace
            )
            if __save_every_map or space in self._record_features:
                self.record_observation(space_container, encoded_obs)

        _concatenated = np.concatenate(encoded_obs_list, axis=0)

        self.__iter += 1
        if self.__iter % self._save_every_x_obs == 0:
            self.save_feature_map_history()

        return (
            _concatenated
            if not self._encoder.observation_space_manager._frame_stacking
            else np.expand_dims(_concatenated, axis=0)
        )

    def record_observation(
        self, space_container: BaseFeatureMapSpace, observation: np.ndarray
    ):
        space_name = space_container.__class__.__name__
        if space_name not in self._recordings:
            self._recordings[space_name] = []

        self._recordings[space_name].append(
            observation.reshape(
                space_container.feature_map_size, space_container.feature_map_size
            )
        )

    def save_feature_map_history(self):
        for name, history in self._recordings.items():
            np.savez_compressed(f"{name}", history)
