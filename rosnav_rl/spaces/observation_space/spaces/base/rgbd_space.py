import numpy as np
from gymnasium import spaces
from numpy import ndarray

from rosnav_rl.observations import (
    ImageColorCollector,
    ImageDepthCollector,
)
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("rgbd")
class RGBDSpace(BaseObservationSpace):
    """A base observation space implementation that combines RGB color images with depth information.
    
    This class implements the RGBD (Red, Green, Blue, Depth) observation space,
    processing both color images and depth data into a combined representation suitable
    for machine learning models.
    
    Attributes:
        name (str): Name identifier for the observation space.
        required_observation_units (list): List of required collector classes for this observation space.
    Parameters:
        rgbd_image_height (int): Height of the RGBD image.
        rgbd_image_width (int): Width of the RGBD image.
        *args: Variable length argument list passed to parent class.
        **kwargs: Arbitrary keyword arguments passed to parent class.
    """
    name: str = "RGBD"
    required_observation_units = [ImageColorCollector, ImageDepthCollector]

    def __init__(
        self, rgbd_image_height: int, rgbd_image_width: int, *args, **kwargs
    ) -> None:
        self._image_height = rgbd_image_height
        self._image_width = rgbd_image_width
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym observation space for the RGBD
        observation space.

        Returns:
            spaces.Space: The Gym observation space.
        """
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(4, self._image_height, self._image_width),
            dtype=np.float32,
        )

    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> ndarray:
        """
        Encodes the RGBD observation by concatenating the observation
        into a 4-channel (4, H, W)-tensor and flattening it.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded RGBD image of shape (4*H*W,).
        """
        depth = observation[ImageDepthCollector.name]  # shape (H, W)
        color = observation[ImageColorCollector.name]  # shape (3, H, W)
        # concatenate channel dimension
        depth = np.expand_dims(depth, axis=0)  # shape (1, H, W)
        image = np.concatenate((color, depth), axis=0)
        return image
