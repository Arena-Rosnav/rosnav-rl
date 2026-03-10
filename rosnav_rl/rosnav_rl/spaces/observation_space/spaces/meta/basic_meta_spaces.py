import numpy as np
from gymnasium import spaces

from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.META)
class IsFirstStepSpace(BaseObservationSpace):
    """Original observation space indicating if this is the first step of an episode.

    This binary observation helps the agent understand episode boundaries and can be
    useful for resetting internal states or special first-step behaviors.
    """

    name = "IsFirstStepSpace"

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the is_first observation.

        Returns:
            spaces.Space: Binary discrete space (0 or 1).
        """
        return spaces.Discrete(2)

    def encode_observation(self, is_first: int = 0, *args, **kwargs) -> int:
        """
        Encodes the is_first observation.

        Args:
            is_first (int): 1 if first step, 0 otherwise.

        Returns:
            int: 1 if first step, 0 otherwise.
        """
        return is_first


@SpaceFactory.register(auto_name=True, category=SpaceCategory.META)
class IsTerminalStepSpace(BaseObservationSpace):
    """Original observation space indicating if this is a terminal step of an episode.

    This binary observation helps the agent understand when an episode is ending,
    which can be important for terminal state value estimation and planning.
    """

    name = "IsTerminalStepSpace"

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the is_terminal observation.

        Returns:
            spaces.Space: Binary discrete space (0 or 1).
        """
        return spaces.Discrete(2)

    def encode_observation(self, is_terminal: int = 0, *args, **kwargs) -> int:
        """
        Encodes the is_terminal observation.

        Args:
            is_terminal (int): 1 if terminal step, 0 otherwise.

        Returns:
            int: 1 if terminal step, 0 otherwise.
        """
        return is_terminal


@SpaceFactory.register(auto_name=True, category=SpaceCategory.META)
class EpisodeStepSpace(BaseObservationSpace):
    """Episode step counter for temporal awareness.

    This space provides the agent with information about how many steps
    have elapsed in the current episode, which can be useful for time-aware
    decision making and episode length normalization.
    """

    name = "EpisodeStepSpace"

    def __init__(self, max_episode_steps: int = 1000, *args, **kwargs):
        """
        Initialize episode step space.

        Args:
            max_episode_steps: Maximum expected episode length
        """
        self.max_episode_steps = max_episode_steps
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the episode step observation.

        Returns:
            spaces.Space: Box space [0, max_episode_steps].
        """
        return spaces.Box(
            low=0, high=self.max_episode_steps, shape=(1,), dtype=np.int32
        )

    def encode_observation(self, episode_step: int = 0, *args, **kwargs) -> np.ndarray:
        """
        Encodes the episode step observation.

        Args:
            episode_step (int): The current episode step.

        Returns:
            np.ndarray: Current episode step number.
        """
        return np.array([episode_step], dtype=np.int32)
