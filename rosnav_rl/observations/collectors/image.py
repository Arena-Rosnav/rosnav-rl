from __future__ import annotations

from abc import ABC
from typing import List

import numpy as np
import sensor_msgs.msg as sensor_msgs
from cv_bridge import CvBridge
from rl_utils.utils.constants import Simulator

from .base_collector import ObservationCollectorUnit

__all__ = ["ImageColorCollector", "ImageDepthCollector"]


class ImageCollector(ObservationCollectorUnit[sensor_msgs.Image, np.ndarray], ABC):
    """An abstract base class for collecting image observations from ROS topics.
    
    This class extends ObservationCollectorUnit to handle image messages from ROS topics,
    converting them from sensor_msgs.Image format to numpy arrays for use in reinforcement
    learning environments.
    
    Attributes:
        name (str): Name of the observation collector.
        topic (str): ROS topic to subscribe to for image data.
        up_to_date_required (bool): Flag indicating whether the latest message is required.
            Defaults to True.
        applicable_simulators (List[Simulator]): List of simulators this collector works with.
            By default supports Unity and Gazebo simulators.
    """

    name: str
    topic: str
    up_to_date_required: bool = True
    applicable_simulators: List[Simulator] = [
        Simulator.UNITY,
        Simulator.GAZEBO,
    ]

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the ImageCollector object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._image_bridge = CvBridge()


class ImageColorCollector(ImageCollector):
    """
    A class that collects color images as observations.

    Attributes:
        name (str): The name of the collector.
        topic (str): The topic to subscribe to for image messages.
        msg_data_class (Type[sensor_msgs.Image]): The message data class for image messages.
        data_class (Type[np.ndarray]): The data class for the collected images.
        up_to_date_required (bool): Specifies whether value should be kept up to date, i.e. a new message is required for every step.

    Methods:
        __init__(*args, **kwargs): Initializes the ImageColorCollector object.
        preprocess(msg: sensor_msgs.Image) -> np.ndarray: Preprocesses the image message and returns the processed image.
    """

    name: str = "image_color"
    topic: str = "rgbd/image"

    def preprocess(self, msg: sensor_msgs.Image) -> np.ndarray:
        """
        Preprocesses the image message and returns the processed image.

        Args:
            msg (sensor_msgs.Image): The image message to preprocess.

        Returns:
            np.ndarray: The processed image.
        """
        if msg.data == b"":
            # calling image bridge with empty message will crash program
            return None
        cv_mat = self._image_bridge.imgmsg_to_cv2(msg)
        # let the channel dim be the first dim and remove a-channel
        return (np.asarray(cv_mat, dtype=np.float32)[:, :, 0:3]).transpose((2, 0, 1))


class ImageDepthCollector(ImageCollector):
    """
    A class that collects depth images as observations.

    Attributes:
        name (str): The name of the collector.
        topic (str): The topic to subscribe to for depth images.
        msg_data_class (Type[sensor_msgs.Image]): The ROS message data class for depth images.
        data_class (Type[np.ndarray]): The data class for depth images.
        up_to_date_required (bool): Specifies whether value should be kept up to date, i.e. a new message is required for every step.

    Methods:
        __init__(*args, **kwargs): Initializes the ImageDepthCollector object.
        preprocess(msg: sensor_msgs.Image) -> np.ndarray: Preprocesses the depth image message.
    """

    name: str = "image_depth"
    topic: str = "rgbd/depth"

    def preprocess(self, msg: sensor_msgs.Image) -> np.ndarray:
        """
        Preprocesses the depth image message.

        Args:
            msg (sensor_msgs.Image): The depth image message.

        Returns:
            np.ndarray: The preprocessed depth image.

        """
        if msg.data == b"":
            # calling image bridge with empty message will crash program
            return None
        cv_mat = self._image_bridge.imgmsg_to_cv2(msg)
        return np.asarray(cv_mat, dtype=np.float32)
