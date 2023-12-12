from typing import Any, Dict, List

from gymnasium import spaces

from .observation_space_factory import SpaceFactory
from .space_index import SPACE_FACTORY_KEYS
from .spaces.base.goal_space import GoalSpace
from .spaces.base.laser_space import LaserScanSpace
from .spaces.base.last_action_space import LastActionSpace
from .spaces.feature_maps.pedestrian_location_space import PedestrianLocationSpace
from .spaces.feature_maps.pedestrian_type_space import PedestrianTypeSpace
from .spaces.feature_maps.pedestrian_vel_x_space import PedestrianVelXSpace
from .spaces.feature_maps.pedestrian_vel_y_space import PedestrianVelYSpace
from .utils import stack_spaces


class ObservationSpaceManager:
    def __init__(
        self,
        space_list: List[str],
        space_kwargs: Dict[str, Any],
        enable_frame_stacking: bool = False,
    ) -> None:
        self._spacelist = space_list
        self._space_kwargs = space_kwargs
        self._enable_frame_stacking = enable_frame_stacking

        self._setup_spaces()

    @property
    def space_list(self):
        return self._spacelist

    @property
    def unified_observation_space(self) -> spaces.Box:
        return stack_spaces(
            *self._observation_spaces.values(),
            frame_stacking_enabled=self._enable_frame_stacking,
        )

    def _setup_spaces(self):
        self._space_containers = self._generate_space_container(
            self._spacelist, self._space_kwargs
        )
        self._observation_spaces = self._generate_observation_spaces(
            self._space_containers
        )

    def _generate_space_container(
        self,
        names: List[str],
        space_kwargs: Dict[str, Any],
    ) -> Dict[str, "BaseObservationSpace"]:
        return {
            name: SpaceFactory.instantiate(
                SPACE_FACTORY_KEYS[name].value, **space_kwargs
            )
            for name in names
        }

    def _generate_observation_spaces(
        self, space_containers: Dict[str, "BaseObservationSpace"]
    ) -> Dict[str, "BaseObservationSpace"]:
        return {
            name: container.get_gym_space()
            for name, container in space_containers.items()
        }

    def add_observation_space(self, space: str):
        assert (
            space not in self._spacelist
        ), f"{space}-ObservationSpace was already specified!"
        self._spacelist.append(space)
        self._setup_spaces()

    def add_multiple_observation_spaces(self, space_list: List[str]):
        for space in space_list:
            self.add_observation_space(space)

    def get_single_observation_space(self, name: str) -> spaces.Box:
        return self._observation_spaces[name]
