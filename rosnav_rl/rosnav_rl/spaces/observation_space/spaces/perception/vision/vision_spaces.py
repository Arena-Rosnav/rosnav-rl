"""Vision Perception Spaces

RGBD and vision-based perception spaces integrated into hierarchical architecture.
"""

from typing import Any

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import ImageData
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory

from ...base_observation_space import BaseObservationSpace


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

    def __init__(self, rgbd_image_height: int, rgbd_image_width: int, *args: Any, **kwargs: Any) -> None:
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
        self, color_image: ImageData, depth_image: ImageData, *args: Any, **kwargs: Any
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

@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class RGBSpace(BaseObservationSpace):

    name = "RGBSpace"

    def __init__(
        self, rgb_sources: list[str] | None = None, image_height: int = 128, image_width: int = 128, *args, **kwargs
    ) -> None:

        self._sources = sorted(rgb_sources or [])
        self.requires = {source: ImageData for source in self._sources}
        self._image_height = image_height
        self._image_width = image_width
        self._space = spaces.Box(
            low=0, high=255, shape=(len(self._sources), 3, image_height, image_width), dtype=np.uint8
        )
        super().__init__(*args, **kwargs)

    def get_gym_space(self):
        """
        Returns the Gym observation space for the RGB observation space.

        The space represents a 3-channel image RGB with shape
        (height, width, 3) and values in range [0, 255].

        Returns:
            spaces.Space: The Gym observation space.
        """
        return self._space

    def resize_chw(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize an (C, H, W) image to (height, width) via bilinear interpolation."""
        if image.shape[1] == height and image.shape[2] == width:
            return image
        import torch
        import torch.nn.functional as F

        tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()
        resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        return resized.squeeze(0).numpy()

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, *args, **obs):
        out = np.zeros((len(self._sources), 3, self._image_height, self._image_width), np.uint8)
        for i, s in enumerate(self._sources):
            img = obs.get(s)

            if img is None or img.size == 0:
                continue  # no img recieved -> zeros

            img = np.asarray(img)
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

            if img.shape[-1] != 3:  
                continue  # None RGB format

            img = self.resize_hwc(img, self._image_height, self._image_width)
            out[i] = np.clip(img, 0, 255).astype(np.uint8).transpose(2, 0, 1)
            
        return out
