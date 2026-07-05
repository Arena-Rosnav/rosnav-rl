"""Robust Navigation Spaces - Production Ready

Goal and navigation spaces with only proven, reliable features.
"""

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.observation_types import (
    DistanceAngleMetrics,
)
from ...observation_space_factory import SpaceFactory
from ...space_categories import SpaceCategory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register(auto_name=True, category=SpaceCategory.NAVIGATION)
class RobustGoalSpace(BaseObservationSpace):
    """Advanced navigation goal observation space with robust normalization and progress tracking.

    Provides a production-ready, normalized representation of the robot's position relative to its navigation goal.
    Features robust distance normalization, angle normalization, and optional progress tracking for curriculum learning
    and reward shaping.

    Technical Specifications:
    - Distance Normalization: Configurable (tanh, linear, log) for smooth, bounded output
    - Angle Normalization: Maps angle to [-1, 1] for stable learning
    - Progress Tracking: Optional, measures normalized distance reduction

    Configuration:
    - goal_max_dist: Reference distance for normalization (meters)
    - include_progress: Whether to include progress metric
    - distance_scaling: Normalization method ("tanh", "linear", "log")

    Output Format: 2- or 3-dimensional normalized vector [distance, angle, progress?] for navigation control.

    Applications: Goal-reaching, reward shaping, curriculum learning, and progress-based exploration.
    """

    name = "RobustGoalSpace"
    requires = {
        "dist_angle_to_goal": DistanceAngleMetrics,
    }

    def __init__(
        self,
        goal_max_dist: float = 50.0,
        include_progress: bool = True,
        distance_scaling: str = "tanh",  # "tanh", "linear", or "log"
        *args,
        **kwargs,
    ):
        """Initialize robust goal space.

        Args:
            goal_max_dist: Reference distance for normalization
            include_progress: Include simple progress tracking
            distance_scaling: Normalization method ("tanh" recommended)
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_distance = goal_max_dist
        self.include_progress = include_progress
        self.distance_scaling = distance_scaling

        # Simple progress tracking
        self.last_distance = None

        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        """Reset episode-local state."""
        self.last_distance = None

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for goal representation."""
        dims = 3 if self.include_progress else 2  # [distance, angle, progress]

        return spaces.Box(
            low=np.array([-1.0] * dims), high=np.array([1.0] * dims), dtype=np.float32
        )

    def _normalize_distance(self, distance: float) -> float:
        """Normalize distance using selected method."""
        if self.distance_scaling == "tanh":
            # Smooth, bounded normalization
            return np.tanh(distance / self.max_distance)
        elif self.distance_scaling == "linear":
            # Simple linear clipping
            return np.clip(distance / self.max_distance, 0.0, 1.0)
        elif self.distance_scaling == "log":
            # Logarithmic scaling for wide distance ranges
            return np.log(1 + distance) / np.log(1 + self.max_distance)
        else:
            raise ValueError(f"Unknown distance scaling: {self.distance_scaling}")

    def _compute_progress(self, current_distance: float) -> float:
        """Compute simple progress metric."""
        if self.last_distance is None:
            progress = 0.0
        else:
            # Progress as normalized distance reduction
            distance_reduction = max(0.0, self.last_distance - current_distance)
            progress = np.tanh(
                distance_reduction / self.max_distance * 10.0
            )  # Scale up for sensitivity

        self.last_distance = current_distance
        return progress

    def encode_observation(
        self, dist_angle_to_goal: DistanceAngleMetrics, *args, **kwargs
    ) -> DistanceAngleMetrics:
        """Encode robust navigation goal with configurable normalization and progress tracking.

        Args:
            dist_angle_to_goal (DistanceAngleMetrics): Distance and angle measurements to navigation goal
                - Shape: (2,) - [distance, angle]
                - Dtype: np.float32
                - Units: [meters, radians]
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [2.5, 0.785] (2.5m away, 45° to the right)

        Returns:
            DistanceAngleMetrics: Normalized goal representation.
                - Shape: (2,) or (3,) if include_progress=True
                - Dtype: np.float32
                - Elements: [norm_distance, norm_angle, progress?]
                - Units: [normalized, normalized, normalized?]
                - Range: all elements ∈ [-1,1]
                - Example: [0.8, 0.25, 0.1] (distance, angle, progress)
        """
        distance = float(dist_angle_to_goal[0])
        angle = float(dist_angle_to_goal[1])

        # Normalize distance using selected method
        normalized_distance = self._normalize_distance(distance)

        # Normalize angle to [-1, 1]
        normalized_angle = angle / np.pi

        result = [normalized_distance, normalized_angle]

        if self.include_progress:
            progress = self._compute_progress(distance)
            result.append(progress)

        return np.array(result, dtype=np.float32)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.NAVIGATION)
class MultiScaleGoalSpace(BaseObservationSpace):
    """Multi-scale navigation goal observation space for distance sensitivity.

    Represents the navigation goal at multiple distance scales, enabling the agent to reason about
    both close and far goals for improved planning and curriculum learning.

    Technical Specifications:
    - Multi-Scale Distance: Configurable list of distance scales for normalization
    - Angle Normalization: Maps angle to [-1, 1] for stable learning

    Configuration:
    - distance_scales: List of scale factors (e.g., [0.5, 1.0, 2.0])
    - goal_max_dist: Base maximum distance (meters)

    Output Format: (num_scales * 2)-dimensional normalized vector [distance, angle, ...] for each scale.

    Applications: Curriculum learning, multi-scale planning, and reward shaping.
    """

    name = "MultiScaleGoalSpace"
    requires = {
        "dist_angle_to_goal": DistanceAngleMetrics,  # Distance and angle to goal for multi-scale analysis
    }

    def __init__(
        self, distance_scales: list = None, goal_max_dist: float = 50.0, *args, **kwargs
    ):
        """Initialize multi-scale goal space.

        Args:
            distance_scales: List of scale factors [0.5, 1.0, 2.0] = [close, medium, far]
            goal_max_dist: Base maximum distance
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.distance_scales = distance_scales or [0.5, 1.0, 2.0]
        self.base_max_distance = goal_max_dist

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for multi-scale goal representation."""
        # Each scale contributes 2 dimensions (distance, angle)
        total_dims = len(self.distance_scales) * 2

        return spaces.Box(
            low=np.array([-1.0] * total_dims),
            high=np.array([1.0] * total_dims),
            dtype=np.float32,
        )

    def encode_observation(
        self, dist_angle_to_goal: DistanceAngleMetrics, *args, **kwargs
    ) -> DistanceAngleMetrics:
        """Encode navigation goal at multiple distance scales for curriculum learning.

        Args:
            dist_angle_to_goal (DistanceAngleMetrics): Distance and angle measurements to navigation goal
                - Shape: (2,) - [distance, angle]
                - Dtype: np.float32
                - Units: [meters, radians]
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [2.5, 0.785] (will be represented at multiple scales)

        Returns:
            DistanceAngleMetrics: Multi-scale goal representation.
                - Shape: (num_scales * 2,) - [scale1_distance, scale1_angle, scale2_distance, scale2_angle, ...]
                - Dtype: np.float32
                - Elements: Repeated [norm_distance, norm_angle] pairs for each scale
                - Units: [normalized, normalized] per scale
                - Range: all elements ∈ [-1,1]
                - Example: [0.8, 0.25, 0.4, 0.25, 0.2, 0.25] for 3 scales
        """
        distance, angle = float(dist_angle_to_goal[0]), float(dist_angle_to_goal[1])

        # Normalize angle (same for all scales)
        normalized_angle = angle / np.pi

        result = []

        for scale in self.distance_scales:
            max_distance = self.base_max_distance * scale

            # Different distance normalization for each scale
            normalized_distance = np.tanh(distance / max_distance)

            result.extend([normalized_distance, normalized_angle])

        return np.array(result, dtype=np.float32)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.NAVIGATION)
class SubgoalContextSpace(BaseObservationSpace):
    """Subgoal context observation space with hierarchical goal-subgoal relationship.

    Provides both subgoal and main goal information, enabling the agent to reason about navigation
    hierarchies and relationships for advanced planning and curriculum learning.

    Technical Specifications:
    - Subgoal and Goal Encoding: Normalized distance and angle for both subgoal and main goal
    - Goal-Subgoal Relationship: Optional relational features (distance and angle difference)

    Configuration:
    - subgoal_max_dist: Maximum distance for normalization (meters)
    - include_goal_subgoal_relation: Whether to include relational features

    Output Format: 4- or 6-dimensional normalized vector [subgoal, goal, relation?] for hierarchical navigation.

    Applications: Hierarchical navigation, curriculum learning, and relational planning.
    """

    name = "SubgoalContextSpace"

    # Schema-based requirements: defines the data sources needed from observations.yaml
    # Each key corresponds to a data source name, each value provides rich type metadata
    requires = {
        "dist_angle_to_subgoal": DistanceAngleMetrics,  # Distance and angle to subgoal
        "dist_angle_to_goal": DistanceAngleMetrics,  # Distance and angle to main goal
    }

    def __init__(
        self,
        subgoal_max_dist: float = 30.0,
        include_goal_subgoal_relation: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize subgoal context space.

        Args:
            subgoal_max_dist: Maximum distance for normalization
            include_goal_subgoal_relation: Include relative goal-subgoal information
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_distance = subgoal_max_dist
        self.include_goal_subgoal_relation = include_goal_subgoal_relation

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for subgoal context."""
        base_dims = 4  # [subgoal_dist, subgoal_angle, goal_dist, goal_angle]

        if self.include_goal_subgoal_relation:
            base_dims += 2  # [distance_diff, angle_diff]

        return spaces.Box(
            low=np.array([-1.0] * base_dims),
            high=np.array([1.0] * base_dims),
            dtype=np.float32,
        )

    def encode_observation(
        self,
        dist_angle_to_subgoal: DistanceAngleMetrics,
        dist_angle_to_goal: DistanceAngleMetrics,
        *args,
        **kwargs,
    ) -> DistanceAngleMetrics:
        """Encode subgoal context with hierarchical goal-subgoal relationship features.

        Args:
            dist_angle_to_subgoal (DistanceAngleMetrics): Distance and angle measurements to navigation subgoal
                - Shape: (2,) - [distance, angle]
                - Dtype: np.float32
                - Units: [meters, radians]
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [1.2, -0.523] (1.2m away, 30° to the left)
            dist_angle_to_goal (DistanceAngleMetrics): Distance and angle measurements to main navigation goal
                - Shape: (2,) - [distance, angle]
                - Dtype: np.float32
                - Units: [meters, radians]
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [8.5, 0.262] (8.5m away, 15° to the right)

        Returns:
            DistanceAngleMetrics: Subgoal context representation.
                - Shape: (4,) or (6,) if include_goal_subgoal_relation=True
                - Dtype: np.float32
                - Elements: [norm_subgoal_dist, norm_subgoal_angle, norm_goal_dist, norm_goal_angle, distance_diff?, angle_diff?]
                - Units: [normalized, normalized, normalized, normalized, normalized?, normalized?]
                - Range: all elements ∈ [-1,1]
                - Example: [0.3, -0.17, 0.85, 0.08, 0.55, 0.25] (subgoal context with relations)
        """
        # Directly unpack and convert inputs to a NumPy array for vectorized operations
        data = np.array([dist_angle_to_subgoal, dist_angle_to_goal], dtype=np.float32)
        distances = data[:, 0]
        angles = data[:, 1]

        # Pre-calculate constants to avoid repeated division
        inv_max_dist = 1.0 / self.max_distance
        inv_pi = 1.0 / np.pi

        # Vectorized normalization for distances and angles
        norm_distances = np.tanh(distances * inv_max_dist)
        norm_angles = angles * inv_pi

        # Assemble the base result array
        result_list = [
            norm_distances[0],
            norm_angles[0],
            norm_distances[1],
            norm_angles[1],
        ]

        if self.include_goal_subgoal_relation:
            # Calculate relational features
            distance_diff = np.tanh((distances[1] - distances[0]) * inv_max_dist)
            angle_diff = np.clip((angles[1] - angles[0]) * inv_pi, -1.0, 1.0)
            result_list.extend([distance_diff, angle_diff])

        return np.array(result_list, dtype=np.float32)
