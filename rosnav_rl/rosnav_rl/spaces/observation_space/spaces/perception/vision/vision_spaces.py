"""Vision Perception Spaces

RGBD and vision-based perception spaces integrated into hierarchical architecture.
"""

import numpy as np
from gymnasium import spaces


from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from ...base_observation_space import BaseObservationSpace
from rosnav_rl.observations.utils.types import ImageData


@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class RGBDSpace(BaseObservationSpace):
    """RGBD observation space combining RGB color and depth images for perception.

    Provides a 4-channel (RGB + Depth) image suitable for deep learning models.

    Technical Specifications:
    - Input: RGB color image, depth image
    - Output: Combined RGBD image (H, W, 4)
    - Normalization: [0, 255] uint8

    Applications: Visual navigation, semantic segmentation, end-to-end learning.
    """

    name = "RGBDSpace"
    requires = {
        "color_image": ImageData,  # RGB color image
        "depth_image": ImageData,  # Depth image
    }

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
        self, color_image: ImageData, depth_image: ImageData, *args, **kwargs
    ) -> ImageData:
        """
        Encodes RGBD observation by combining color and depth images.

        Args:
            color_image (ImageData): RGB color image, shape (H, W, 3) or (3, H, W)
            depth_image (ImageData): Depth image, shape (H, W) or (H, W, 1)

        Returns:
            ImageData: Combined RGBD image as a 4-channel array (H, W, 4)
        """
        # Ensure color image is (H, W, 3)
        if color_image.ndim == 3 and color_image.shape[0] in (3, 4):
            # Convert from CHW to HWC if needed
            color_image = np.transpose(color_image, (1, 2, 0))

        # Ensure depth image has shape (H, W, 1)
        if depth_image.ndim == 2:
            depth_image = np.expand_dims(depth_image, axis=-1)
        elif depth_image.ndim == 3 and depth_image.shape[-1] != 1:
            # If depth image is (1, H, W), convert to (H, W, 1)
            if depth_image.shape[0] == 1:
                depth_image = np.transpose(depth_image, (1, 2, 0))

        # Combine RGB and Depth into RGBD (4 channels)
        rgbd_image = np.concatenate([color_image, depth_image], axis=-1)
        return rgbd_image.astype(np.uint8)
