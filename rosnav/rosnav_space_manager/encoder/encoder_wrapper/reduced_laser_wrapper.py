import numpy as np
import rospy

from ..base_space_encoder import BaseSpaceEncoder
from .base_encoder_wrapper import BaseEncoderWrapper


class ReducedLaserWrapper(BaseEncoderWrapper):
    """
    A wrapper class that reduces the number of laser beams in the observation.

    Args:
        encoder (BaseSpaceEncoder): The base space encoder.
        desired_num_beams (int): The desired number of laser beams.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _reduced_num_laser_beams (int): The number of reduced laser beams.

    """

    def __init__(
        self, encoder: BaseSpaceEncoder, desired_num_beams: int, *args, **kwargs
    ):
        self._reduced_num_laser_beams = desired_num_beams
        super().__init__(encoder)

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        """
        Encodes the observation by reducing the number of laser beams.

        Args:
            observation (dict): The observation dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: The encoded observation.

        """
        observation["laser_scan"] = ReducedLaserWrapper.reduce_laserbeams(
            observation["laser_scan"], self._reduced_num_laser_beams
        )

        return self._encoder.encode_observation(observation)

    @staticmethod
    def reduce_laserbeams(laserbeams: np.ndarray, x: int) -> np.ndarray:
        """
        Reduces the number of laser beams in the given laser scan.

        Args:
            laserbeams (np.ndarray): The laser scan.
            x (int): The number of reduced laser beams.

        Returns:
            np.ndarray: The reduced laser scan.

        """
        if x >= len(laserbeams):
            return laserbeams
        indices = np.round(np.linspace(0, len(laserbeams) - 1, x)).astype(int)[:x]
        if type(laserbeams) == tuple:
            laserbeams = np.array(laserbeams)
        return laserbeams[indices]
