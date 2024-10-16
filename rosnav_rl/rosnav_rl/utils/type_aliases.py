from typing import Any, Dict, Union

from rosnav_rl.spaces import BaseObservationSpace, BaseFeatureMapSpace

ObservationDict = Dict[str, Any]
ObservationSpaceUnit = Union[BaseObservationSpace, BaseFeatureMapSpace]
