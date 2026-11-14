"""
ROS2 launch file for the rosnav_rl action server.

Launches the DRL action server node that loads a trained RL model and serves
velocity commands via the GetCommand service. The DRLController nav2 plugin
calls this service to get velocity commands.

Launch arguments:
    agent_name (str): Name of the trained agent directory to load.
    namespace (str): ROS namespace for the node (default: "").
    robot (str): Arena robot model name, its model_params.yaml supplies the base frame.
    frame (str): Frame prefix of the robot (default: "").
    base_frame (str): Base frame override, defaults to the robot's base_frame.
    arena_env_node (str): Env node whose task_reset topic resets the agent (default: "", no reset).
    agents_dir (str): Base directory containing agent sub-folders.
        Sets ROSNAV_AGENTS_DIR on the process. Defaults to the
        ROSNAV_AGENTS_DIR env var, with a fallback to arena_training/agents/.

Examples:
    ros2 launch rosnav_rl action_server.launch.py agent_name:=my_ppo_agent
    ros2 launch rosnav_rl action_server.launch.py agent_name:=my_ppo_agent agents_dir:=/data/agents
"""

from pathlib import Path

from rosnav_rl.utils.yaml_io import load_yaml
from ament_index_python.packages import get_package_share_directory

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def _action_server_node(context: LaunchContext) -> list[Node]:
    base_frame = LaunchConfiguration("base_frame").perform(context)
    robot = LaunchConfiguration("robot").perform(context)
    if not base_frame and robot:
        model_params = Path(get_package_share_directory("arena_robots")) / "robots" / robot / "model_params.yaml"
        base_frame = load_yaml(model_params)["base_frame"]
    return [
        Node(
            package="rosnav_rl",
            executable="action_server.py",
            name="rosnav_action_server",
            output="screen",
            parameters=[
                {
                    "agent_name": LaunchConfiguration("agent_name"),
                    "namespace": LaunchConfiguration("namespace"),
                    "frame": LaunchConfiguration("frame"),
                    "base_frame": base_frame or "base_link",
                    "arena_env_node": LaunchConfiguration("arena_env_node"),
                }
            ],
            additional_env={"ROSNAV_AGENTS_DIR": LaunchConfiguration("agents_dir")},
        )
    ]


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

    robot_arg = DeclareLaunchArgument(
        "robot",
        default_value="",
        description="Arena robot model name whose model_params.yaml base_frame the server uses",
    )

    frame_arg = DeclareLaunchArgument(
        "frame",
        default_value="",
        description="Frame prefix of the robot",
    )

    base_frame_arg = DeclareLaunchArgument(
        "base_frame",
        default_value="",
        description="Base frame override, defaults to the robot's base_frame, else base_link",
    )

    arena_env_node_arg = DeclareLaunchArgument(
        "arena_env_node",
        default_value="",
        description="Env node whose task_reset topic resets the agent, empty disables the reset",
    )

    return LaunchDescription(
        [
            agent_name_arg,
            namespace_arg,
            agents_dir_arg,
            robot_arg,
            frame_arg,
            base_frame_arg,
            arena_env_node_arg,
            OpaqueFunction(function=_action_server_node),
        ]
    )
