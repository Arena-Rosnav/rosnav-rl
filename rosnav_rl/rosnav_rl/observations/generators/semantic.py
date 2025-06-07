from __future__ import annotations

from warnings import warn

__all__ = [
    "PedestrianLocationGenerator",
    "PedestrianRelativeLocationGenerator",
    "PedestrianRelativeVelGenerator",
    "PedestrianRelativeVelXGenerator",
    "PedestrianRelativeVelYGenerator",
    "PedestrianDistanceGenerator",
    "PedestrianTypeGenerator",
    "PedestrianSocialStateGenerator",
]

from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

if TYPE_CHECKING:
    from rosnav_rl.utils.type_aliases.observation import ObservationDict

from ..collectors import (
    BaseUnit,
    PeopleDataCollector,
    RobotPoseCollector,
)
from ..utils.semantic import get_relative_pos_to_robot, get_relative_vel_to_robot
from .base_generator import ObservationGeneratorUnit


class PedestrianLocationGenerator(ObservationGeneratorUnit[np.ndarray]):
    name: str = "ped_location"
    requires: List[BaseUnit] = [PeopleDataCollector]
    data_class = np.ndarray

    def generate(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate an array of pedestrian locations.

        Args:
            obs_dict (ObservationDict): Dictionary containing observation data including
                pedestrian locations.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: An array of pedestrian locations. Returns an empty array if no pedestrians are detected.
        """
        people_data: PeopleDataCollector.data_class = obs_dict[PeopleDataCollector.name]

        if len(people_data) == 0:
            return np.array([])

        return np.stack([[data.position.x, data.position.y] for data in people_data])


class PedestrianRelativeLocationGenerator(ObservationGeneratorUnit[np.ndarray]):
    """
    Observation generator unit that computes the relative locations of pedestrians with respect to the robot.

    This generator requires the robot's pose and pedestrian locations as inputs. It transforms the pedestrian
    locations from the global frame to the robot's local frame.

    Returns:
        np.ndarray: Empty array if no pedestrians are detected, otherwise an array of relative positions
                    of pedestrians with respect to the robot's frame. Each row represents [x, y] coordinates
                    of a pedestrian in the robot's frame.
    """

    name: str = "ped_relative_location"
    requires: List[BaseUnit] = [RobotPoseCollector, PeopleDataCollector]
    data_class = np.ndarray

    def generate(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Generates an array of pedestrian locations relative to the robot's position.

        Args:
            obs_dict (ObservationDict): Dictionary containing observation data including
                pedestrian locations and robot pose.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: An array of pedestrian positions relative to the robot.
                Returns an empty array if no pedestrians are detected.
        """
        people_data: PeopleDataCollector.data_class = obs_dict[PeopleDataCollector.name]

        if len(people_data) == 0:
            return np.array([])

        return get_relative_pos_to_robot(
            robot_pose=obs_dict[RobotPoseCollector.name],
            distant_poses=np.stack(
                [[data.position.x, data.position.y, 1] for data in people_data]
            ),
        )


class PedestrianRelativeVelGenerator(ObservationGeneratorUnit[np.ndarray]):
    """
    Observation generator unit for calculating the relative velocities of pedestrians with respect to the robot.

    This class collects pedestrian velocity components (x and y) and transforms them into the robot's reference frame.
    It returns an array of relative velocity vectors for detected pedestrians.

    Attributes:
        name (str): Identifier for this observation generator unit.
        requires (List[BaseUnit]): List of collectors this generator depends on.
        data_class: The numpy.ndarray class used for the output data.

    Returns:
        np.ndarray: Array of relative velocity vectors of pedestrians in robot's reference frame.
                   Returns empty array if no pedestrians are detected.
                   Shape: (num_pedestrians, 2) where each row is [rel_vel_x, rel_vel_y]
    """

    name: str = "ped_relative_vel"
    requires: List[BaseUnit] = [RobotPoseCollector, PeopleDataCollector]
    data_class = np.ndarray

    def generate(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate relative velocities of pedestrians with respect to the robot.

        This method extracts pedestrian velocity data from the observation dictionary,
        combines the x and y components, and computes the relative velocities
        with respect to the robot's frame of reference.

        Args:
            obs_dict: Dictionary containing observation data including pedestrian
                     velocities and robot pose.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of relative pedestrian velocities with respect to the robot.
                       Returns an empty array if no pedestrian data is available.
        """
        ped_vel_x: PeopleDataCollector.data_class = [
            data.velocity.x for data in obs_dict[PeopleDataCollector.name]
        ]
        ped_vel_y: PeopleDataCollector.data_class = [
            data.velocity.y for data in obs_dict[PeopleDataCollector.name]
        ]

        if len(ped_vel_x) == 0 or len(ped_vel_y) == 0:
            return np.array([])

        ped_vel = np.stack(
            [[data_x, data_y] for data_x, data_y in zip(ped_vel_x, ped_vel_y)]
        )

        return get_relative_vel_to_robot(
            robot_pose=obs_dict[RobotPoseCollector.name],
            pedestrian_vel_vector=ped_vel,
        )


class PedestrianRelativeVelXGenerator(ObservationGeneratorUnit[np.ndarray]):
    name: str = "ped_relative_vel_x"
    requires: List[BaseUnit] = [PedestrianRelativeVelGenerator]
    data_class = np.ndarray

    def generate(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> np.ndarray:
        ped_rel_vel: PedestrianRelativeVelGenerator.data_class = obs_dict[
            PedestrianRelativeVelGenerator.name
        ]
        return ped_rel_vel[:, 0] if len(ped_rel_vel) > 0 else ped_rel_vel


class PedestrianRelativeVelYGenerator(ObservationGeneratorUnit[np.ndarray]):
    name: str = "ped_relative_vel_y"
    requires: List[BaseUnit] = [PedestrianRelativeVelGenerator]
    data_class = np.ndarray

    def generate(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> np.ndarray:
        ped_rel_vel: PedestrianRelativeVelGenerator.data_class = obs_dict[
            PedestrianRelativeVelGenerator.name
        ]

        return ped_rel_vel[:, 1] if len(ped_rel_vel) > 0 else ped_rel_vel


class PedestrianDistanceGenerator(
    ObservationGeneratorUnit[Dict[Union[str, int], float]]
):
    """ObservationGeneratorUnit that calculates the minimum distance to pedestrians of each type.

    This generator uses pedestrian relative locations and types to compute the minimum
    distance to pedestrians of each unique type. The output is a dictionary mapping
    pedestrian type IDs to their corresponding minimum distances.

    Returns:
        Dict[Union[str, int], float]: A dictionary where keys are pedestrian type IDs and
            values are the minimum distances to pedestrians of that type. Returns an empty
            dictionary if no pedestrian data is available or if there's a mismatch between
            pedestrian types and locations.
    """

    name: str = "ped_type_distances"
    requires: List[BaseUnit] = [
        PedestrianRelativeLocationGenerator,
        PeopleDataCollector,
    ]
    data_class = Dict[Union[str, int], float]

    def generate(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[Union[str, int], float]]:
        """
        Generate a dictionary that maps each unique pedestrian type to the minimum distance
        to a pedestrian of that type.

        Args:
            obs_dict (ObservationDict): Dictionary containing observation data, should include
                                      pedestrian relative locations and types.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, float]: Dictionary mapping pedestrian types to minimum distances.
                             Returns empty dictionary if required data is not available or inconsistent.

        Note:
            - Requires 'PedestrianRelativeLocationGenerator.name' and 'PedestrianTypeCollector.name' in obs_dict.
            - Warns if the number of pedestrian types and locations don't match.
        """
        ped_distances = {}

        relative_locations = obs_dict.get(
            PedestrianRelativeLocationGenerator.name, None
        )

        # determine index of 'group_id' in data.tagnames
        try:
            group_id_index = obs_dict[PeopleDataCollector.name].tagnames.index(
                "group_id"
            )
        except ValueError:
            warn(
                "Pedestrian group ID not found in the data. Returning empty dictionary."
            )
            return ped_distances

        ped_groups = [
            data.tags[group_id_index] for data in obs_dict[PeopleDataCollector.name]
        ]

        if relative_locations is None or ped_groups is None:
            return ped_distances

        if len(relative_locations) == 0 or len(ped_groups) == 0:
            return ped_distances

        if len(ped_groups) != len(relative_locations):
            warn("Number of pedestrian types and locations do not match!")
            return ped_distances

        distances = np.linalg.norm(relative_locations, axis=1)
        groups = np.array(ped_groups)

        # get the unique types
        for _type in groups:
            ped_distances[_type] = np.min(distances[groups == _type])

        return ped_distances


class PedestrianTypeGenerator(ObservationGeneratorUnit[np.ndarray]):
    name: str = "ped_types"
    requires: List[BaseUnit] = [PeopleDataCollector]
    data_class = np.ndarray

    def generate(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate the social state of pedestrians.

        Args:
            obs_dict (ObservationDict): Dictionary containing observation data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: Array of social states for each pedestrian.
        """
        try:
            behavior_idx = obs_dict[PeopleDataCollector.name].tagnames.index("group_id")
            return np.array(
                [data.tags[behavior_idx] for data in obs_dict[PeopleDataCollector.name]]
            )
        except ValueError:
            warn(
                "Pedestrian social state not found in the data. Returning empty array."
            )
            return np.array([])


class PedestrianSocialStateGenerator(ObservationGeneratorUnit[np.ndarray]):
    name: str = "ped_social_state"
    requires: List[BaseUnit] = [PeopleDataCollector]
    data_class = np.ndarray

    def generate(
        self,
        obs_dict: ObservationDict,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate the social state of pedestrians.

        Args:
            obs_dict (ObservationDict): Dictionary containing observation data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: Array of social states for each pedestrian.
        """
        try:
            behavior_idx = obs_dict[PeopleDataCollector.name].tagnames.index("behavior")
            return np.array(
                [data.tags[behavior_idx] for data in obs_dict[PeopleDataCollector.name]]
            )
        except ValueError:
            warn(
                "Pedestrian social state not found in the data. Returning empty array."
            )
            return np.array([])
