"""
ROS2 launch file for the rosnav_rl action server.

Launches the DRL action server node that loads a trained RL model and serves
velocity commands via the GetCommand service. The DRLController nav2 plugin
calls this service to get velocity commands.

Launch arguments:
    agent_name (str): Name of the trained agent directory to load.
    namespace (str): ROS namespace for the node (default: "").
    agents_dir (str): Base directory containing agent sub-folders.
        Sets ROSNAV_AGENTS_DIR on the process. Defaults to the
        ROSNAV_AGENTS_DIR env var, with a fallback to arena_training/agents/.

Examples:
    ros2 launch rosnav_rl action_server.launch.py agent_name:=my_ppo_agent
    ros2 launch rosnav_rl action_server.launch.py agent_name:=my_ppo_agent agents_dir:=/data/agents
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
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

    agents_dir_arg = DeclareLaunchArgument(
        "agents_dir",
        default_value=EnvironmentVariable("ROSNAV_AGENTS_DIR", default_value=""),
        description=(
            "Base directory that contains agent sub-folders. "
            "Sets ROSNAV_AGENTS_DIR for the launched process. "
            "Defaults to the ROSNAV_AGENTS_DIR env var, then to arena_training/agents/."
        ),
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
        additional_env={"ROSNAV_AGENTS_DIR": LaunchConfiguration("agents_dir")},
    )

    return LaunchDescription(
        [
            agent_name_arg,
            namespace_arg,
            agents_dir_arg,
            action_server_node,
        ]
    )
