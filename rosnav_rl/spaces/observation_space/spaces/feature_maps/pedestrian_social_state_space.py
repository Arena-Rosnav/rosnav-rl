import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    PedestrianRelativeLocationGenerator,
    PedestrianSocialStateCollector,
)
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("ped_social_state")
class PedestrianSocialStateSpace(BaseFeatureMapSpace):
    """A feature map space representing pedestrian social states in a grid.
    
    This class creates a 2D feature map where each cell contains a social state value 
    for pedestrians detected in that spatial region. The social state is an integer 
    value extracted from semantic data.
    
    Attributes:
        name (str): Identifier for this space type, set to "PEDESTRIAN_SOCIAL_STATE".
        required_observation_units (list): Units required for generating observations:
            - PedestrianSocialStateCollector: Collects social state data from pedestrians
            - PedestrianRelativeLocationGenerator: Provides relative positions of pedestrians
    
    Parameters:
        ped_social_state_num (int): Number of possible pedestrian social states.
        feature_map_size (int): Size of the feature map (width and height in cells).
        roi_in_m (float): Region of interest in meters around the robot.
        *args: Variable length argument list passed to parent class.
        **kwargs: Arbitrary keyword arguments passed to parent class.
    
    The feature map encodes pedestrian social states as integer values in a grid,
    where each cell corresponds to a spatial location. The social state is extracted
    from the 'evidence' field of pedestrian data points by bit-shifting.
    """
    name = "PEDESTRIAN_SOCIAL_STATE"
    required_observation_units = [
        PedestrianSocialStateCollector,
        PedestrianRelativeLocationGenerator,
    ]

    def __init__(
        self,
        ped_social_state_num: int,
        feature_map_size: int,
        roi_in_m: float,
        *args,
        **kwargs
    ) -> None:
        self._social_state_num = ped_social_state_num
        super().__init__(
            feature_map_size=feature_map_size, roi_in_m=roi_in_m, *args, **kwargs
        )

    def get_gym_space(self) -> spaces.Space:
        """
        Get the Gym space representation of the feature map.

        Returns:
            spaces.Space: The Gym space representing the feature map.
        """
        return spaces.Box(
            low=0,
            high=self._social_state_num,
            shape=(1, self._feature_map_size, self._feature_map_size),
            dtype=int,
        )

    def _get_semantic_map(
        self,
        semantic_data: PedestrianSocialStateCollector.data_class,
        relative_pos: PedestrianRelativeLocationGenerator.data_class,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Generates a semantic map representing pedestrian social states.
        
        This method creates a 2D grid map where each cell may contain the social state of a pedestrian 
        located at that position. The social states are extracted from the provided semantic data, and
        positioned on the map according to the relative positions of pedestrians.
        
        Args:
            semantic_data (PedestrianSocialStateCollector.data_class): Data containing pedestrian social state information.
            relative_pos (PedestrianRelativeLocationGenerator.data_class): Data containing pedestrian relative positions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            np.ndarray: A 2D numpy array (feature_map_size × feature_map_size) where values represent 
                pedestrian social states. Zeros indicate empty cells with no pedestrian presence.
        """
        social_state_map = np.zeros((self.feature_map_size, self.feature_map_size))
        social_states = list(
            map(
                lambda x: int(x.evidence) >> 8,
                semantic_data.points,
            )
        )

        if social_states is not None and relative_pos is not None:
            for state, pos in zip(social_states, relative_pos):
                index = self._get_map_index(pos)
                if (
                    0 <= index[0] < self.feature_map_size
                    and 0 <= index[1] < self.feature_map_size
                ):
                    social_state_map[index] = state

        return social_state_map

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Encode the observation into a numpy array.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            np.ndarray: The encoded observation as a numpy array.
        """
        return self._get_semantic_map(
            observation[PedestrianSocialStateCollector.name],
            observation[PedestrianRelativeLocationGenerator.name],
        )
