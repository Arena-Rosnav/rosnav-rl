"""
Example usage of the improved ObservationManager with the new base classes.

This example shows how to:
1. Create collectors and generators using the new base classes
2. Set up the ObservationManager with data sources
3. Use the factory to create from YAML configuration
"""

import yaml
from typing import Dict, Any
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from .observation_manager import ObservationManager
from .observation_factory import create_observation_manager_from_config
from .collectors import LaserScanCollector, OdometryCollector, PoseStampedCollector
from .generators import GoalLocationInRobotFrameGenerator, DistAngleToGoalGenerator
from ..states import SimulationStateContainer
from ..utils.rostopic import Namespace


def create_observation_manager_programmatically(node: Node) -> ObservationManager:
    """Example of creating an ObservationManager programmatically."""

    # Create simulation state container
    simulation_state = SimulationStateContainer()

    # Create namespace
    ns = Namespace("robot1")

    # Create data sources manually
    data_sources = {
        # Collectors
        "laser_scan": LaserScanCollector(
            name="laser_scan",
            topic="scan",
            node=node,
            up_to_date_required=True,  # Critical sensor requiring fresh data
        ),
        "robot_pose": OdometryCollector(
            name="robot_pose",
            topic="odom",
            node=node,
            up_to_date_required=True,  # Critical for navigation
        ),
        "goal_pose": PoseStampedCollector(
            name="goal_pose",
            topic="goal",
            node=node,
            up_to_date_required=False,  # Goal updates less frequently
        ),
        # Generators
        "goal_in_robot_frame": GoalLocationInRobotFrameGenerator(
            name="goal_in_robot_frame",
            inputs={"robot_pose": "robot_pose", "goal_pose": "goal_pose"},
        ),
        "distance_to_goal": DistAngleToGoalGenerator(
            name="distance_to_goal",
            inputs={"goal_in_robot_frame": "goal_in_robot_frame"},
        ),
    }

    # Create QoS profile
    qos_profile = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
    )

    # Create observation manager
    return ObservationManager(
        node=node,
        ns=ns,
        data_sources=data_sources,
        simulation_state_container=simulation_state,
        qos_profile=qos_profile,
        wait_for_obs=True,
        enable_synchronization=True,
    )


def create_observation_manager_from_yaml(
    node: Node, config_path: str
) -> ObservationManager:
    """Example of creating an ObservationManager from YAML configuration."""

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create simulation state container
    simulation_state = SimulationStateContainer()

    # Create namespace
    ns = Namespace("robot1")

    # Create QoS profile
    qos_profile = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
    )

    return create_observation_manager_from_config(
        config=config,
        node=node,
        ns=ns,
        simulation_state_container=simulation_state,
        qos_profile=qos_profile,
        wait_for_obs=True,
        enable_synchronization=True,
    )


def example_observation_loop(obs_manager: ObservationManager):
    """Example of how to use the observation manager in a control loop."""

    try:
        # Get all observations
        observations = obs_manager.get_observations()

        # Access specific observations
        laser_data = observations.get("laser_scan")
        robot_pose = observations.get("robot_pose")
        goal_distance = observations.get("distance_to_goal")

        # Check health status
        health = obs_manager.get_health_status()

        # Print some info
        if laser_data is not None:
            print(f"Laser scan has {len(laser_data)} points")

        if robot_pose is not None:
            print(
                f"Robot position: x={robot_pose[0]:.2f}, y={robot_pose[1]:.2f}, θ={robot_pose[2]:.2f}"
            )

        if goal_distance is not None:
            print(
                f"Distance to goal: {goal_distance[0]:.2f}m, angle: {goal_distance[1]:.2f}rad"
            )

        # Check for unhealthy collectors
        unhealthy = [
            name
            for name, status in health.items()
            if not status.get("is_healthy", True)
        ]
        if unhealthy:
            print(f"Warning: Unhealthy collectors: {unhealthy}")

    except Exception as e:
        print(f"Error getting observations: {e}")


# Example YAML configuration that would work with this setup
EXAMPLE_CONFIG = """
aliases:
  robot_pose: robot_pose_odom
  laser: front_laser
  goal: goal_pose

datasources:
  # Collectors
  robot_pose_odom:
    type: OdometryCollector
    params:
      topic: "odom"
      up_to_date_required: true      # Critical sensor data
      
  front_laser:
    type: LaserScanCollector
    params:
      topic: "scan"
      up_to_date_required: true      # Essential for obstacle avoidance
      
  goal_pose:
    type: PoseStampedCollector
    params:
      topic: "goal"
      up_to_date_required: false     # Goal updates less frequently
  
  # Generators  
  goal_in_robot_frame:
    type: GoalLocationInRobotFrameGenerator
    params:
      inputs:
        robot_pose: robot_pose_odom
        goal_pose: goal_pose
        
  distance_to_goal:
    type: DistAngleToGoalGenerator
    params:
      inputs:
        goal_in_robot_frame: goal_in_robot_frame
"""


def save_example_config(path: str):
    """Save the example configuration to a file."""
    with open(path, "w") as f:
        f.write(EXAMPLE_CONFIG)


if __name__ == "__main__":
    # This would be used in a ROS node
    import rclpy

    rclpy.init()
    node = Node("observation_example")

    try:
        # Method 1: Programmatic creation
        obs_manager1 = create_observation_manager_programmatically(node)

        # Method 2: From YAML (save config first)
        config_path = "/tmp/observation_config.yaml"
        save_example_config(config_path)
        obs_manager2 = create_observation_manager_from_yaml(node, config_path)

        # Example usage
        example_observation_loop(obs_manager1)

    finally:
        obs_manager1.shutdown()
        obs_manager2.shutdown()
        node.destroy_node()
        rclpy.shutdown()
