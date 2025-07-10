"""
This script initializes and starts an action server node for the ROS navigation framework.

Functions:
    parse_args(): Parses command-line arguments for the simulation framework, agent name, and namespace.

Usage:
    Run this script with optional command-line arguments to specify the simulation framework, agent name, and namespace.
    Example:
        python action_server.py -sim arena -mp agent_name -ns namespace

Command-line Arguments:
    -sim, --simulation_framework: The simulation framework to use (default: "arena").
    -mp, --agent_name: The name of the agent (default: None).
    -ns, --namespace: The namespace for the ROS node (default: "").

Raises:
    ValueError: If the specified simulation framework is not supported.
"""

import argparse
from rosnav_rl.action_server.arena_server import ArenaActionServer
import rclpy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-sim", "--simulation_framework", type=str, default="arena")
    parser.add_argument("-mp", "--agent_name", type=str, default=None)
    parser.add_argument("-ns", "--namespace", type=str, default="")

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    rospy.init_node("action_server_node")
    args = parse_args()

    if args.simulation_framework == "arena":
        action_server = ArenaActionServer(
            agent_name=args.agent_name, namespace=args.namespace
        )
    else:
        raise ValueError(
            f"Simulation framework {args.simulation_framework} not supported!"
        )
    action_server.start()
