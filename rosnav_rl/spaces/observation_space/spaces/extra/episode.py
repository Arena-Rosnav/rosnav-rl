from ..base_observation_space import BaseObservationSpace
from ...observation_space_factory import SpaceFactory

from gymnasium import spaces
from rosnav_rl.utils.type_aliases import ObservationDict


@SpaceFactory.register("is_first")
class IsFirstStepSpace(BaseObservationSpace):
    name = "is_first"

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the is_first observation.

        Returns:
            spaces.Space: The Gym space for the is_first observation.

        """
        return spaces.Discrete(2)

    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> int:
        """
        Encodes the is_first observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            int: The encoded is_first observation.

        """
        return int(observation["is_first"])


@SpaceFactory.register("is_terminal")
class IsTerminalStepSpace(BaseObservationSpace):
    name = "is_terminal"

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the is_terminal observation.

        Returns:
            spaces.Space: The Gym space for the is_terminal observation.

        """
        return spaces.Discrete(2)

    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> int:
        """
        Encodes the is_terminal observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            int: The encoded is_terminal observation.

        """
        return int(observation["is_terminal"])
