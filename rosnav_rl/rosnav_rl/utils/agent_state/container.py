from dataclasses import dataclass
from .states import ActionSpaceState, ObservationSpaceState


@dataclass(frozen=False)
class AgentStateContainer:
    action_space: ActionSpaceState
    observation_space: ObservationSpaceState
