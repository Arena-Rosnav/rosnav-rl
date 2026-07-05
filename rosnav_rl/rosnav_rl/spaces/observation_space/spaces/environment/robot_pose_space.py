"""World-frame robot pose observation space for SE(2) frame canonicalization.

RobotPoseSpace emits the robot's world-frame pose (x, y, theta) at each timestep.
It is NOT added to the encoder's mlp_keys/cnn_keys — it is a training-time
auxiliary target used exclusively by WorldModel._train to compute per-step
relative poses P_k = se2_between(anchor_pose, pose_k) for the SE(2) ped-target
transform. During inference/imagination, pose_accum is integrated from actions.

The registered name is "RobotPoseSpace" so data["RobotPoseSpace"] carries shape
(B, T, 3) in training batches.
"""

from typing import ClassVar, Dict, Any

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.observation_types import Pose2D
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class RobotPoseSpace(BaseObservationSpace):
    """World-frame robot pose (x, y, theta) for SE(2) target-frame correction.

    Shape: (3,) = [x_world, y_world, theta_world].

    Used only at training time to compute per-step relative poses for
    transforming decoded pedestrian positions from anchor frame to current frame.
    NOT consumed by the encoder; NOT in mlp_keys.
    """

    name: ClassVar[str] = "RobotPoseSpace"
    requires: ClassVar[Dict[str, Any]] = {
        "robot_pose": Pose2D,
    }

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    def encode_observation(
        self,
        robot_pose: Pose2D,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.asarray(robot_pose, dtype=np.float32)
