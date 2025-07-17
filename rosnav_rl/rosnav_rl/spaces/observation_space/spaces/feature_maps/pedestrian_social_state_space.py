import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    PedestrianRelativeLocationGenerator,
    PedestrianSocialStateGenerator,
    PedestrianLocationGenerator,
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
        PedestrianLocationGenerator,
        PedestrianSocialStateGenerator,
        PedestrianRelativeLocationGenerator,
    ]
    background_value = -1

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
            observation[PedestrianSocialStateGenerator.name],
            poses=observation[PedestrianLocationGenerator.name],
            relative_poses=observation[PedestrianRelativeLocationGenerator.name],
            robot_pose=observation[PedestrianLocationGenerator.name],
        )
