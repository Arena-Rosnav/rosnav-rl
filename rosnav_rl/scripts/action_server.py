#!/usr/bin/env python3
"""
ROS2 action server node for rosnav_rl model inference.

Loads a trained RL agent and serves velocity commands via the GetCommand service,
which is called by the DRLController nav2 plugin.

Usage:
    ros2 run rosnav_rl action_server --ros-args -p agent_name:=my_agent -p namespace:=""

    Or via launch file:
    ros2 launch rosnav_rl action_server.launch.py agent_name:=my_agent
"""

import rclpy
from rclpy.node import Node

from rosnav_rl.action_server.arena_server import ArenaActionServer


def main(args=None):
    rclpy.init(args=args)

    # Create a temporary node to read parameters
    param_node = rclpy.create_node("_action_server_param_reader")

    param_node.declare_parameter("agent_name", "")
    param_node.declare_parameter("namespace", "")

    agent_name = param_node.get_parameter("agent_name").get_parameter_value().string_value
    namespace = param_node.get_parameter("namespace").get_parameter_value().string_value

    param_node.destroy_node()

    if not agent_name:
        # Fallback: try environment variable
        import os
        agent_name = os.environ.get("ROSNAV_AGENT_NAME", "")

    if not agent_name:
        print("[rosnav_rl] ERROR: No agent_name specified. "
              "Use --ros-args -p agent_name:=<name> or set ROSNAV_AGENT_NAME env var.")
        rclpy.shutdown()
        return

    action_server = ArenaActionServer(
        agent_name=agent_name,
        namespace=namespace,
    )

    try:
        action_server.start()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
