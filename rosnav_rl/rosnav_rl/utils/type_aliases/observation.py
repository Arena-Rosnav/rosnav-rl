from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, Union

from rosnav_rl.observations.data_sources import (
    Collector,
    Generator,
)

if TYPE_CHECKING:
    from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
        BaseObservationSpace,
    )
    from rosnav_rl.spaces.observation_space.spaces.environment.feature_map_spaces import (
        BaseFeatureMapSpace,
    )


ObservationName = str
ObservationDict = dict[ObservationName, Any]
ObservationSpaceUnit = Union["BaseObservationSpace", "BaseFeatureMapSpace"]

ObservationCollector = TypeVar("ObservationCollector", bound=Collector)
ObservationGenerator = TypeVar("ObservationGenerator", bound=Generator)

ObservationSpaceList = list[type[ObservationSpaceUnit]]
ObservationSpaceKwargs = dict[str, Any]
