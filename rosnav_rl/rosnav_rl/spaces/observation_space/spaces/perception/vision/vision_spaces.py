"""Vision Perception Spaces

RGBD and vision-based perception spaces integrated into hierarchical architecture.
"""

import numpy as np
from gymnasium import spaces
from numpy import ndarray

from rosnav_rl.observations import ImageColorCollector, ImageDepthCollector
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from ...base_observation_space import BaseObservationSpace


@SpaceFactory.register("rgbd")
class RGBDSpace(BaseObservationSpace):
    """A base observation space implementation that combines RGB color images with depth information.

    This class implements the RGBD (Red, Green, Blue, Depth) observation space,
    processing both color images and depth data into a combined representation suitable
    for machine learning models.
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
        Returns the Gym observation space for the RGBD observation space.

        The space represents a 4-channel image (RGB + Depth) with shape
        (height, width, 4) and values in range [0, 255].

        Returns:
            spaces.Space: The Gym observation space.
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(self._image_height, self._image_width, 4),
            dtype=np.uint8,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> ndarray:
        """
        Encodes RGBD observation by combining color and depth images.

        Args:
            observation (ObservationDict): Dictionary containing color and depth image data.

        Returns:
            ndarray: Combined RGBD image as a 4-channel array.
        """
        # Extract color image (RGB, 3 channels)
        color_image = observation[ImageColorCollector.name]

        # Extract depth image (1 channel)
        depth_image = observation[ImageDepthCollector.name]

        # Ensure depth image has correct shape (add channel dimension if needed)
        if depth_image.ndim == 2:
            depth_image = np.expand_dims(depth_image, axis=-1)

        # Combine RGB and Depth into RGBD (4 channels)
        rgbd_image = np.concatenate([color_image, depth_image], axis=-1)

        return rgbd_image.astype(np.uint8)
