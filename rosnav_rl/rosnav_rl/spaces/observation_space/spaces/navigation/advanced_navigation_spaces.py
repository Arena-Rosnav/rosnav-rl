"""Robust Navigation Spaces - Production Ready

Goal and navigation spaces with only proven, reliable features.
"""

from typing import Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import DistAngleToGoalGenerator, DistAngleToSubgoalGenerator
from rosnav_rl.utils.type_aliases import ObservationDict
from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("robust_goal")
class RobustGoalSpace(BaseObservationSpace):
    """Production-ready goal space with reliable distance normalization.
    
    Features only proven techniques:
    - Hyperbolic tangent normalization (smooth, bounded)
    - Simple progress tracking (distance reduction)
    - Configurable distance scaling
    """
    
    name = "ROBUST_GOAL"
    required_observation_units = [DistAngleToGoalGenerator]
    
    def __init__(self,
                 max_distance: float = 50.0,
                 include_progress: bool = True,
                 distance_scaling: str = "tanh",  # "tanh", "linear", or "log"
                 *args, **kwargs):
        """Initialize robust goal space.
        
        Args:
            max_distance: Reference distance for normalization
            include_progress: Include simple progress tracking
            distance_scaling: Normalization method ("tanh" recommended)
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_distance = max_distance
        self.include_progress = include_progress
        self.distance_scaling = distance_scaling
        
        # Simple progress tracking
        self.last_distance = None
        
        super().__init__(*args, **kwargs)
    
    def get_gym_space(self) -> spaces.Space:
        """Return gym space for goal representation."""
        dims = 3 if self.include_progress else 2  # [distance, angle, progress]
        
        return spaces.Box(
            low=np.array([-1.0] * dims),
            high=np.array([1.0] * dims),
            dtype=np.float32
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
            progress = np.tanh(distance_reduction / self.max_distance * 10.0)  # Scale up for sensitivity
        
        self.last_distance = current_distance
        return progress
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode goal observation with robust normalization.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Normalized goal representation
        """
        goal_data = observation[DistAngleToGoalGenerator.name]
        distance = float(goal_data[0])
        angle = float(goal_data[1])
        
        # Normalize distance using selected method
        normalized_distance = self._normalize_distance(distance)
        
        # Normalize angle to [-1, 1]
        normalized_angle = angle / np.pi
        
        result = [normalized_distance, normalized_angle]
        
        if self.include_progress:
            progress = self._compute_progress(distance)
            result.append(progress)
        
        return np.array(result, dtype=np.float32)


@SpaceFactory.register("multi_scale_goal")
class MultiScaleGoalSpace(BaseObservationSpace):
    """Multi-scale goal representation for better distance sensitivity.
    
    Represents the goal at multiple distance scales to help the agent
    make better decisions at different distances from the goal.
    """
    
    name = "MULTI_SCALE_GOAL"
    required_observation_units = [DistAngleToGoalGenerator]
    
    def __init__(self,
                 distance_scales: list = None,
                 base_max_distance: float = 50.0,
                 *args, **kwargs):
        """Initialize multi-scale goal space.
        
        Args:
            distance_scales: List of scale factors [0.5, 1.0, 2.0] = [close, medium, far]
            base_max_distance: Base maximum distance
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.distance_scales = distance_scales or [0.5, 1.0, 2.0]
        self.base_max_distance = base_max_distance
        
        super().__init__(*args, **kwargs)
    
    def get_gym_space(self) -> spaces.Space:
        """Return gym space for multi-scale goal representation."""
        # Each scale contributes 2 dimensions (distance, angle)
        total_dims = len(self.distance_scales) * 2
        
        return spaces.Box(
            low=np.array([-1.0] * total_dims),
            high=np.array([1.0] * total_dims),
            dtype=np.float32
        )
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode goal at multiple scales.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Multi-scale goal representation
        """
        goal_data = observation[DistAngleToGoalGenerator.name]
        distance = float(goal_data[0])
        angle = float(goal_data[1])
        
        # Normalize angle (same for all scales)
        normalized_angle = angle / np.pi
        
        result = []
        
        for scale in self.distance_scales:
            max_distance = self.base_max_distance * scale
            
            # Different distance normalization for each scale
            normalized_distance = np.tanh(distance / max_distance)
            
            result.extend([normalized_distance, normalized_angle])
        
        return np.array(result, dtype=np.float32)


@SpaceFactory.register("subgoal_context")
class SubgoalContextSpace(BaseObservationSpace):
    """Subgoal context with goal relationship.
    
    Provides both subgoal and main goal information to give
    the agent context about its navigation hierarchy.
    """
    
    name = "SUBGOAL_CONTEXT"
    required_observation_units = [DistAngleToSubgoalGenerator, DistAngleToGoalGenerator]
    
    def __init__(self,
                 max_distance: float = 30.0,
                 include_goal_subgoal_relation: bool = True,
                 *args, **kwargs):
        """Initialize subgoal context space.
        
        Args:
            max_distance: Maximum distance for normalization
            include_goal_subgoal_relation: Include relative goal-subgoal information
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_distance = max_distance
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
            dtype=np.float32
        )
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode subgoal context.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Subgoal context representation
        """
        subgoal_data = observation.get(DistAngleToSubgoalGenerator.name, np.array([0.0, 0.0]))
        goal_data = observation.get(DistAngleToGoalGenerator.name, np.array([0.0, 0.0]))
        
        subgoal_distance = float(subgoal_data[0])
        subgoal_angle = float(subgoal_data[1])
        goal_distance = float(goal_data[0])
        goal_angle = float(goal_data[1])
        
        # Normalize distances and angles
        norm_subgoal_dist = np.tanh(subgoal_distance / self.max_distance)
        norm_subgoal_angle = subgoal_angle / np.pi
        norm_goal_dist = np.tanh(goal_distance / self.max_distance)
        norm_goal_angle = goal_angle / np.pi
        
        result = [norm_subgoal_dist, norm_subgoal_angle, norm_goal_dist, norm_goal_angle]
        
        if self.include_goal_subgoal_relation:
            # Distance and angle differences (how much further is goal vs subgoal)
            distance_diff = np.tanh((goal_distance - subgoal_distance) / self.max_distance)
            angle_diff = np.clip((goal_angle - subgoal_angle) / np.pi, -1.0, 1.0)
            
            result.extend([distance_diff, angle_diff])
        
        return np.array(result, dtype=np.float32)
