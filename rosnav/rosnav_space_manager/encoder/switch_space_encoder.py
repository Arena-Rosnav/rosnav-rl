from typing import List

from .base_space_encoder import BaseSpaceEncoder
from ...utils.action_space.switch_space import SwitchActionSpace
from ...utils.observation_space.space_index import SPACE_INDEX


class SwitchSpaceEncoder(BaseSpaceEncoder):
    def setup_action_space(self, action_space_kwargs: dict):
        self._action_space_manager = SwitchActionSpace(**action_space_kwargs)
