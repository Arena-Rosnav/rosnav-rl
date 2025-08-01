"""
Rich type annotations for the observation system.

This module provides schema-based type annotations inspired by Albumentations,
offering rich metadata for observation data types used throughout the system.
The annotations include descriptions, shapes, units, constraints, and examples
to improve developer experience and documentation.
"""

from __future__ import annotations

from typing import Annotated, Dict, Union
from dataclasses import dataclass

import numpy as np
import people_msgs.msg as people_msgs


# Schema-based data requirement specification (inspired by Albumentations)
@dataclass(frozen=True)
class DataSpec:
    """
    Schema for specifying data requirements with rich metadata.
    Similar to Albumentations' parameter specifications.

    This provides a structured way to document observation data types with
    comprehensive metadata that helps developers understand the expected
    format, constraints, and usage of each data type.
    """

    description: str
    shape: str = ""
    units: str = ""
    source: str = ""
    constraints: str = ""
    example: str = ""


# Core robot and sensor data types
Pose2D = Annotated[
    np.ndarray,
    DataSpec(
        description="Pose 2D in world coordinates",
        shape="(3,)",
        units="meters, meters, radians",
        source="odometry or SLAM",
        constraints="theta ∈ [-π, π]",
        example="[2.5, 1.2, 0.785]",
    ),
]

LidarRanges = Annotated[
    np.ndarray,
    DataSpec(
        description="Preprocessed laser scan ranges",
        shape="(n,)",
        units="meters",
        source="lidar sensor",
        constraints="ranges ∈ [0, max_range], NaN replaced with max_range",
        example="[0.5, 1.2, 3.4, ..., 2.1]",
    ),
]

RobotRelativePosition = Annotated[
    np.ndarray,
    DataSpec(
        description="Position relative to robot's local frame",
        shape="(2,)",
        units="meters",
        constraints="x: forward/backward, y: left/right from robot perspective",
        example="[1.5, -0.8]",
    ),
]

# Navigation and planning data types
GoalLocation = Annotated[
    np.ndarray,
    DataSpec(
        description="Goal or waypoint position in world coordinates",
        shape="(3,)",
        units="meters, meters, radians",
        source="navigation planner",
        constraints="theta ∈ [-π, π]",
        example="[5.0, 3.2, 1.57]",
    ),
]

SubgoalLocation = Annotated[
    np.ndarray,
    DataSpec(
        description="Subgoal or intermediate waypoint position in world coordinates",
        shape="(3,)",
        units="meters, meters, radians",
        source="navigation planner",
        constraints="theta ∈ [-π, π]",
        example="[3.0, 2.1, 0.78]",
    ),
]

RobotVelocity = Annotated[
    np.ndarray,
    DataSpec(
        description="Robot velocity command in base frame",
        shape="(3,)",
        units="meters/second, meters/second, radians/second",
        source="robot controller or action",
        constraints="linear.x, linear.y, angular.z",
        example="[0.5, 0.0, 0.2]",
    ),
]

NavigationPath = Annotated[
    np.ndarray,
    DataSpec(
        description="Planned path waypoints in world coordinates",
        shape="(N, 2)",
        units="meters",
        source="navigation planner",
        constraints="N = number of waypoints, x,y coordinates",
        example="[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]",
    ),
]

ImageData = Annotated[
    np.ndarray,
    DataSpec(
        description="Preprocessed image data from camera sensor",
        shape="(C, H, W) or (H, W)",
        units="normalized or raw pixel values",
        source="camera sensor",
        constraints="CHW format for multi-channel, HW for single channel",
        example="CHW: (3, 480, 640) or HW: (480, 640)",
    ),
]

DistanceAngleMetrics = Annotated[
    np.ndarray,
    DataSpec(
        description="Distance and angle measurements to target",
        shape="(2,)",
        units="meters, radians",
        constraints="distance ≥ 0, angle ∈ [-π, π]",
        example="[2.5, 0.785]",
    ),
]

SafetyStatus = Annotated[
    bool,
    DataSpec(
        description="Safety check result for collision avoidance",
        constraints="True if within safety distance, False otherwise",
        source="laser scan collision detection",
        example="False",
    ),
]

IsTerminal = Annotated[
    bool,
    DataSpec(
        description="Flag indicating if the episode has ended",
        constraints="True if episode is done, False otherwise",
        source="environment termination conditions",
        example="False",
    ),
]

# Pedestrian detection and tracking data types
PedestrianDetections = Annotated[
    people_msgs.People,
    DataSpec(
        description="Detected pedestrians with poses and velocities",
        source="people detector/tracker",
        constraints="Global coordinate frame",
        example="People(people=[Person(position=Point(x=1.0, y=2.0), velocity=Vector3(x=0.5, y=0.0))])",
    ),
]

PedestrianWorldLocations = Annotated[
    np.ndarray,
    DataSpec(
        description="Pedestrian positions in world/global coordinates",
        shape="(N, 2)",
        units="meters",
        constraints="N = number of pedestrians, x,y coordinates",
        example="[[1.5, 2.3], [4.1, -1.2], [0.8, 3.7]]",
    ),
]

PedestrianRelativeLocations = Annotated[
    np.ndarray,
    DataSpec(
        description="Pedestrian positions relative to robot's local frame",
        shape="(N, 2)",
        units="meters",
        constraints="N = number of pedestrians, x: forward/back, y: left/right from robot",
        example="[[1.5, -0.8], [2.1, 1.3], [-0.5, 2.0]]",
    ),
]

PedestrianRelativeVelocities = Annotated[
    np.ndarray,
    DataSpec(
        description="Velocity vectors in robot's reference frame",
        shape="(N, 2)",
        units="meters/second",
        constraints="N = number of pedestrians, x: forward/back, y: left/right",
        example="[[0.5, 0.2], [-0.3, 0.8]]",
    ),
]

# Pedestrian analysis and social navigation data types
PedestrianTypeMinDistances = Annotated[
    Dict[Union[str, int], float],
    DataSpec(
        description="Minimum distances to pedestrians grouped by type/group ID",
        units="meters",
        constraints="Keys are group IDs, values are minimum distances ≥ 0",
        example="{1: 2.5, 2: 4.1, 3: 1.8}",
    ),
]

PedestrianTypeArray = Annotated[
    np.ndarray,
    DataSpec(
        description="Array of pedestrian group IDs/types, ordered by detection",
        shape="(N,)",
        constraints="N = number of pedestrians, values are integer group IDs",
        example="[1, 2, 1, 3, 2]",
    ),
]

PedestrianSocialStates = Annotated[
    np.ndarray,
    DataSpec(
        description="Array of pedestrian social/behavior states",
        shape="(N,)",
        constraints="N = number of pedestrians, values are integer behavior codes",
        example="[0, 1, 0, 2, 1]",
    ),
]


# Additional encoded output types for observation spaces
MotionStateVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Robot motion state representation including velocity magnitude and direction",
        shape="(4,)",
        units="normalized, radians, normalized, normalized",
        constraints="[velocity_magnitude, velocity_direction, consistency_score, stability_metric]",
        example="[0.75, 0.785, 0.92, 0.88]",
    ),
]

KinematicStateVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Complete kinematic state including normalized pose and velocities",
        shape="(7,)",
        units="normalized, normalized, normalized, normalized, normalized, normalized, normalized",
        constraints="[norm_x, norm_y, cos_yaw, sin_yaw, norm_linear_vel, norm_angular_vel, speed_magnitude]",
        example="[0.5, -0.3, 0.707, 0.707, 0.4, 0.1, 0.41]",
    ),
]

TrajectoryStateVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Trajectory state with path smoothness and curvature metrics",
        shape="(8,)",
        units="normalized, normalized, normalized, normalized, normalized, normalized, normalized, normalized",
        constraints="[norm_x, norm_y, cos_yaw, sin_yaw, norm_linear_vel, norm_angular_vel, "
        "path_smoothness, path_curvature]",
        example="[0.5, -0.3, 0.707, 0.707, 0.4, 0.1, 0.85, 0.62]",
    ),
]

RobotActionVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Robot action command vector (last executed action)",
        shape="(2,) or (3,)",
        units="meters/second, radians/second",
        constraints="2D: [linear_vel, angular_vel], 3D: [linear_vel, translational_vel, angular_vel]",
        example="[0.5, 0.2] or [0.5, 0.1, 0.2]",
    ),
]

GoalRelativePosition = Annotated[
    np.ndarray,
    DataSpec(
        description="Subgoal position in robot's local coordinate frame",
        shape="(2,)",
        units="meters",
        constraints="x: forward/backward, y: left/right from robot perspective",
        example="[2.0, -1.5]",
    ),
]

SubgoalRelativePosition = Annotated[
    np.ndarray,
    DataSpec(
        description="Subgoal position in robot's local coordinate frame",
        shape="(2,)",
        units="meters",
        constraints="x: forward/backward, y: left/right from robot perspective",
        example="[2.0, -1.5]",
    ),
]

EnvironmentContextVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Environment analysis with obstacle density per sector and safety metrics",
        shape="(num_sectors + 3,)",
        units="normalized density values and metrics",
        constraints="[sector_densities..., total_density, min_clearance, obstacle_count]",
        example="[0.2, 0.8, 0.1, 0.0, 0.4, 0.35, 1.5, 12.0]",
    ),
]

SpatialAwarenessVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Spatial awareness with directional clearances and corridor detection",
        shape="(num_directions + 2,)",
        units="meters and boolean",
        constraints="[directional_clearances..., corridor_detected, corridor_direction]",
        example="[2.5, 1.8, 0.5, 3.2, 1.0, 0.785]",
    ),
]

ObstacleProximityVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Obstacle proximity analysis with zone occupancy and emergency indicators",
        shape="(num_zones + 2,)",
        units="normalized occupancy and boolean flags",
        constraints="[zone_occupancies..., emergency_stop_needed, has_valid_readings]",
        example="[0.2, 0.8, 0.0, 0.1, 0.0, 1.0]",
    ),
]

# Feature map output types for environment spaces
PedestrianFeatureMap = Annotated[
    np.ndarray,
    DataSpec(
        description="2D feature map representing pedestrian positions and velocities",
        shape="(map_size, map_size)",
        units="normalized density or velocity values",
        constraints="Grid-based representation with robot at center",
        example="64x64 grid with pedestrian velocity magnitudes",
    ),
]

LaserFeatureMap = Annotated[
    np.ndarray,
    DataSpec(
        description="Stacked laser scan feature maps for temporal analysis",
        shape="(map_size, map_size)",
        units="normalized range values",
        constraints="Multi-frame aggregation of laser data in grid format",
        example="32x32 grid showing obstacle history",
    ),
]

SemanticFeatureMap = Annotated[
    np.ndarray,
    DataSpec(
        description="Semantic feature map with classified environmental elements",
        shape="(map_size, map_size)",
        units="classification values or binary occupancy",
        constraints="Grid representation with semantic labels",
        example="Feature map with pedestrian types or social states",
    ),
]

SocialStateFeatureMap = Annotated[
    np.ndarray,
    DataSpec(
        description="Feature map encoding pedestrian social states in spatial grid",
        shape="(map_size, map_size)",
        units="integer social state codes",
        constraints="Grid with state codes for pedestrians, 0 elsewhere",
        example="32x32 grid with sparse integer values representing social behaviors",
    ),
]

PedestrianTypeFeatureMap = Annotated[
    np.ndarray,
    DataSpec(
        description="Feature map encoding pedestrian types in spatial grid",
        shape="(map_size, map_size)",
        units="integer type codes",
        constraints="Grid with type codes for pedestrians, -1 for empty cells",
        example="32x32 grid with sparse integer values representing pedestrian types",
    ),
]

# Localization output types
FilteredOdometryVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Filtered and normalized odometry data with optional acceleration",
        shape="(2,) or (4,)",
        units="normalized velocity and acceleration",
        constraints="[linear_vel, angular_vel] or [linear_vel, angular_vel, linear_acc, angular_acc]",
        example="[0.5, 0.2] or [0.5, 0.2, 0.1, 0.05]",
    ),
]

StabilizedPoseVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Stabilized pose representation with trigonometric orientation encoding",
        shape="(4,) or (5,)",
        units="normalized position and unitless trigonometric components",
        constraints="[norm_x, norm_y, cos_yaw, sin_yaw, confidence?] with optional confidence metric",
        example="[0.5, 0.2, 0.707, 0.707] or [0.5, 0.2, 0.707, 0.707, 0.9]",
    ),
]

CombinedLocalizationVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Combined localization vector integrating pose and velocity information",
        shape="(6,)",
        units="normalized position, trigonometric orientation, and normalized velocities",
        constraints="[norm_x, norm_y, cos_yaw, sin_yaw, norm_linear_vel, norm_angular_vel]",
        example="[0.5, 0.2, 0.707, 0.707, 0.3, 0.1]",
    ),
]

MissionContextVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Mission context information with progress tracking and temporal awareness",
        shape="(4,)",
        units="normalized progress, time, phase, and urgency metrics",
        constraints="[progress, time_normalized, mission_phase, urgency] all ∈ [0,1]",
        example="[0.6, 0.3, 0.8, 0.2]",
    ),
]

PerformanceContextVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Performance metrics including efficiency and smoothness analysis",
        shape="(4,)",
        units="normalized efficiency and performance metrics",
        constraints="[path_efficiency, energy_efficiency, smoothness, overall_performance] all ∈ [0,1]",
        example="[0.85, 0.7, 0.9, 0.82]",
    ),
]

SafetyContextVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Safety context with risk assessment and collision monitoring",
        shape="(4,)",
        units="normalized risk and safety metrics",
        constraints="[current_risk, avg_risk, violation_rate, safety_margin] all ∈ [0,1]",
        example="[0.2, 0.15, 0.05, 0.85]",
    ),
]

RobotPoseVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Normalized robot pose in world coordinates",
        shape="(3,)",
        units="normalized position and orientation",
        constraints="[norm_x, norm_y, normalized_theta] with theta ∈ [-1,1]",
        example="[0.5, -0.3, 0.25]",
    ),
]

LocalizationQualityVector = Annotated[
    np.ndarray,
    DataSpec(
        description="Localization quality assessment with pose uncertainty and consistency metrics",
        shape="(6,)",
        units="normalized metrics",
        constraints="[pose_uncertainty, velocity_consistency, position_drift, angular_drift, "
        "filter_confidence, measurement_quality]",
        example="[0.1, 0.95, 0.02, 0.01, 0.98, 0.88]",
    ),
]


# Export all type annotations for easy importing
__all__ = [
    "DataSpec",
    "Pose2D",
    "LidarRanges",
    "RobotRelativePosition",
    "GoalLocation",
    "SubgoalLocation",
    "RobotVelocity",
    "NavigationPath",
    "ImageData",
    "DistanceAngleMetrics",
    "SafetyStatus",
    "IsTerminal",
    "PedestrianDetections",
    "PedestrianWorldLocations",
    "PedestrianRelativeLocations",
    "PedestrianRelativeVelocities",
    "PedestrianTypeMinDistances",
    "PedestrianTypeArray",
    "PedestrianSocialStates",
    "MotionStateVector",
    "KinematicStateVector",
    "TrajectoryStateVector",
    "RobotActionVector",
    "SubgoalRelativePosition",
    "EnvironmentContextVector",
    "SpatialAwarenessVector",
    "ObstacleProximityVector",
    "PedestrianFeatureMap",
    "LaserFeatureMap",
    "SemanticFeatureMap",
    "SocialStateFeatureMap",
    "PedestrianTypeFeatureMap",
    "FilteredOdometryVector",
    "StabilizedPoseVector",
    "CombinedLocalizationVector",
    "MissionContextVector",
    "PerformanceContextVector",
    "SafetyContextVector",
    "RobotPoseVector",
    "LocalizationQualityVector",
]
