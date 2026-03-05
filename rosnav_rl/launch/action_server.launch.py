"""
ROS2 launch file for the rosnav_rl action server.

Launches the DRL action server node that loads a trained RL model and serves
velocity commands via the GetCommand service. The DRLController nav2 plugin
calls this service to get velocity commands.

Launch arguments:
    agent_name (str): Name of the trained agent directory to load.
    namespace (str): ROS namespace for the node (default: "").
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    agent_name_arg = DeclareLaunchArgument(
        "agent_name",
        default_value="",
        description="Name of the trained DRL agent to deploy",
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="ROS namespace for the action server node",
    )

    action_server_node = Node(
        package="rosnav_rl",
        executable="action_server.py",
        name="rosnav_action_server",
        output="screen",
        parameters=[
            {
                "agent_name": LaunchConfiguration("agent_name"),
                "namespace": LaunchConfiguration("namespace"),
            }
        ],
    )

    return LaunchDescription(
        [
            agent_name_arg,
            namespace_arg,
            action_server_node,
        ]
    )
