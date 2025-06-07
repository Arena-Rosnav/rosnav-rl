import numpy as np


def get_relative_pos_to_robot(
    robot_pose: np.ndarray, distant_poses: np.ndarray
) -> np.ndarray:
    """Transforms distant poses from map frame to robot frame coordinates.

    This function calculates the relative positions of a set of points with respect to
    the robot's current pose. It creates a homogeneous transformation matrix from map
    to robot coordinates, inverts it, and applies it to the distant poses.

    Args:
        robot_pose: A numpy array or dictionary containing the robot's pose with keys:
            - "x": x-coordinate of the robot in the map frame
            - "y": y-coordinate of the robot in the map frame
            - "yaw": orientation of the robot in the map frame (in radians)
        distant_poses: A numpy array of shape (N, 3) where each row represents a pose
            in homogeneous coordinates [x, y, 1] in the map frame

    Returns:
        A numpy array of shape (N, 2) containing the x and y coordinates of the distant
        poses in the robot's frame of reference (excluding the homogeneous component)
    """
    # Create the homogeneous transformation matrix map_T_robot
    map_T_robot = np.array(
        [
            [
                np.cos(robot_pose["yaw"]),  # Rotation component: cos(yaw)
                -np.sin(robot_pose["yaw"]),  # Rotation component: -sin(yaw)
                robot_pose["x"],  # Translation component: x
            ],
            [
                np.sin(robot_pose["yaw"]),  # Rotation component: sin(yaw)
                np.cos(robot_pose["yaw"]),  # Rotation component: cos(yaw)
                robot_pose["y"],  # Translation component: y
            ],
            [0, 0, 1],  # Homogeneous component: 0, 0, 1
        ]
    )

    # Calculate the inverse transformation matrix robot_T_map
    robot_T_map = np.linalg.inv(map_T_robot)

    # Apply the transformation to the distant poses using einsum, return the transformed poses, excluding the homogeneous component
    return np.einsum("ij,kj->ki", robot_T_map, distant_poses)[:, :2]


def get_relative_vel_to_robot(
    robot_pose: np.ndarray,
    pedestrian_vel_vector: np.ndarray,
) -> np.ndarray:
    """Transforms pedestrian velocity vectors from a global map frame to the robot's local frame.

    This function transforms the velocity vectors of pedestrians from the map coordinate
    frame to the robot's local coordinate frame using a rotation matrix derived from the
    robot's orientation (yaw).

    Parameters:
        robot_pose (np.ndarray): A numpy array containing the robot's pose information,
                                 which must include a 'yaw' key representing the robot's
                                 orientation in radians.
        pedestrian_vel_vector (np.ndarray): A numpy array of shape (n, 2) containing
                                            pedestrian velocity vectors [vx, vy] in
                                            the map coordinate frame.
    Returns:
        np.ndarray: A numpy array of shape (n, 2) containing the pedestrian velocity
                    vectors transformed to the robot's coordinate frame. Returns an
                    empty array with shape (0, 2) if the input pedestrian_vel_vector
                    is empty.
    """
    # Create the rotation matrix to transform from map frame to robot frame
    map_r_robot = np.array(
        [
            [np.cos(robot_pose["yaw"]), -np.sin(robot_pose["yaw"])],
            [np.sin(robot_pose["yaw"]), np.cos(robot_pose["yaw"])],
        ]
    )

    # Use the transpose for transformation
    robot_r_map = map_r_robot.T

    if len(pedestrian_vel_vector) > 0:
        # Transform pedestrian velocities to the robot's coordinate frame
        rel_vel = np.matmul(robot_r_map, pedestrian_vel_vector.T).T
        return rel_vel

    return np.empty((0, 2))
