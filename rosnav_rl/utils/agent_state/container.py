from dataclasses import dataclass
from .states import ActionSpaceState, ObservationSpaceState


@dataclass(frozen=False)
class AgentStateContainer:
    """
    AgentStateContainer is a container class that holds the state of an agent's action space and observation space.

    Attributes:
        action_space (ActionSpaceState): Represents the state of the agent's action space.
        observation_space (ObservationSpaceState): Represents the state of the agent's observation space.
    """

    action_space: ActionSpaceState
    observation_space: ObservationSpaceState
