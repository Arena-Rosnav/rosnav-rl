from __future__ import annotations

from typing import Any, Dict, TypeVar, Union, TYPE_CHECKING, Type, List

from rosnav_rl.observations import (
    ObservationCollectorUnit,
    ObservationGeneratorUnit,
)

if TYPE_CHECKING:
    from rosnav_rl.spaces import BaseFeatureMapSpace, BaseObservationSpace


ObservationName = str
ObservationDict = Dict[ObservationName, Any]
ObservationSpaceUnit = Union["BaseObservationSpace", "BaseFeatureMapSpace"]

ObservationCollector = TypeVar("ObservationCollector", bound=ObservationCollectorUnit)
ObservationGenerator = TypeVar("ObservationGenerator", bound=ObservationGeneratorUnit)

ObservationSpaceList = TypeVar(
    "ObservationSpaceList", bound=List[Type[ObservationSpaceUnit]]
)
