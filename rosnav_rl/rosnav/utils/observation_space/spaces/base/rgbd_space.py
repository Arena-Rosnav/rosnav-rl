import numpy as np
from gymnasium import spaces
from numpy import ndarray

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace

from rl_utils.utils.observation_collector import (
    ImageColorCollector,
    ImageDepthCollector,
)


@SpaceFactory.register("rgbd")
class RGBDSpace(BaseObservationSpace):
    """
    Represents the observation space for laser scan data.

    Args:
        image_height (int): The spatial height of the RGBD input image.
        image_width (int): The spatial width of the RGBD input image.
        *args: Variable length argument list. Might be
            used for BaseObservationSpace init.
        **kwargs: Arbitrary keyword arguments. Might be
            used for BaseObservationSpace init.

    Attributes:
        _image_height (int): RGBD input image height.
        _image_width (int): RGBD input iamge width.
    """

    name: str = "RGBD"
    required_observations = [ImageColorCollector, ImageDepthCollector]

    def __init__(self, image_height: int, image_width: int, *args, **kwargs) -> None:
        self._image_height = image_height
        self._image_width = image_width
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
            shape=(4 * self._image_height * self._image_width,),
            dtype=np.float32,
        )

    def encode_observation(self, observation: dict, *args, **kwargs) -> ndarray:
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
        return image.flatten()
