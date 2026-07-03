"""Standalone in-process rosnav_rl inference node.

Runs the policy at a fixed rate, publishes `subgoal` (lookahead along the
straight robot→goal line) for the observation pipeline, and emits the
decoded command as `TwistStamped` on `~/cmd_vel`.

When `train_mode=true`, only the subgoal pipeline runs; the agent, ObservationManager,
and cmd_vel publisher are skipped — the training loop owns the policy in that case.
"""

from __future__ import annotations

import importlib.resources
import math
from pathlib import Path

import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int16

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.factory.factory import (
    create_observation_manager_from_config,
)
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.agent_paths import load_agent_spec, resolve_agent_dir
from rosnav_rl.utils.rostopic import Namespace


_NODE_NAME = "rosnav_rl_inference"


class ArenaInferenceNode:
    """Owns the per-robot RL agent and drives cmd_vel from observations."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.logger = node.get_logger()

        node.declare_parameter("agent", "")
        node.declare_parameter("namespace", "")
        node.declare_parameter("frame", "map")
        node.declare_parameter("base_frame", "base_link")
        node.declare_parameter("control_rate", 10.0)
        node.declare_parameter("min_lookahead_dist", 0.5)
        node.declare_parameter("max_lookahead_dist", 2.5)
        node.declare_parameter("lookahead_time", 1.5)
        node.declare_parameter("train_mode", False)

        self.agent_name = node.get_parameter("agent").get_parameter_value().string_value
        self.train_mode = bool(node.get_parameter("train_mode").get_parameter_value().bool_value)
        if not self.agent_name and not self.train_mode:
            raise RuntimeError("rosnav_rl_inference: parameter 'agent' is required")
        self.namespace = Namespace(node.get_parameter("namespace").get_parameter_value().string_value)
        self.frame = node.get_parameter("frame").get_parameter_value().string_value or "map"
        self.base_frame = node.get_parameter("base_frame").get_parameter_value().string_value or "base_link"
        self.control_rate = float(node.get_parameter("control_rate").get_parameter_value().double_value or 10.0)
        self.min_lookahead = float(node.get_parameter("min_lookahead_dist").get_parameter_value().double_value or 0.5)
        self.max_lookahead = float(node.get_parameter("max_lookahead_dist").get_parameter_value().double_value or 2.5)
        self.lookahead_time = float(node.get_parameter("lookahead_time").get_parameter_value().double_value or 1.5)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        self._goal: PoseStamped | None = None
        self._last_speed: float = 0.0

        self._subgoal_pub = node.create_publisher(PoseStamped, "subgoal", 1)
        self._goal_sub = node.create_subscription(
            PoseStamped,
            "goal_pose",
            self._on_goal,
            1,
        )

        if not self.train_mode:
            self._cmd_vel_pub = node.create_publisher(TwistStamped, "cmd_vel", 1)
            self._reset_sub = node.create_subscription(
                Int16,
                "/scenario_reset",
                self._on_scenario_reset,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
            )
            self.logger.info(f"[rosnav_rl] loading agent '{self.agent_name}'")
            self.agent = self._load_agent()
            self.logger.info("[rosnav_rl] agent loaded")
            self.observation_manager = self._load_observation_manager()
            self.logger.info("[rosnav_rl] observation manager ready")
        else:
            self.logger.info(
                "[rosnav_rl] subgoal-only mode (train_mode=true); agent and cmd_vel publisher skipped"
            )

        period = 1.0 / max(self.control_rate, 1e-3)
        self._timer = node.create_timer(period, self._step)
        self.logger.info(f"[rosnav_rl] running at {self.control_rate:.1f} Hz")

    def _load_agent(self) -> RL_Agent:
        model_dir = resolve_agent_dir(self.agent_name)
        spec = load_agent_spec(model_dir)
        self._agent_parameters: AgentParameters = spec.parameters
        agent = RL_Agent(spec)
        agent.load_model(path=model_dir / "best_model.zip")
        return agent

    def _load_observation_manager(self):
        agent_dir = resolve_agent_dir(self.agent_name)
        obs_config_path: Path | str = agent_dir / "observations.yaml"
        if not Path(obs_config_path).exists():
            obs_config_path = str(
                importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml"
            )
        with open(obs_config_path) as f:
            config = yaml.safe_load(f)
        return create_observation_manager_from_config(
            config=config,
            node=self.node,
            ns=str(self.namespace),
            simulation_state_container=self._agent_parameters,
            wait_for_obs=False,
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg

    def _on_scenario_reset(self, _msg: Int16) -> None:
        self.agent.model.reset()
        self._last_speed = 0.0

    def _step(self) -> None:
        if self._goal is None:
            return

        robot_xy = self._lookup_robot_xy()
        if robot_xy is None:
            return

        goal_xy = (self._goal.pose.position.x, self._goal.pose.position.y)
        self._publish_subgoal(robot_xy, goal_xy)

        if self.train_mode:
            return

        try:
            observations = self.observation_manager.get_observations()
        except Exception as exc:
            self.logger.warn(
                f"[rosnav_rl] get_observations failed: {type(exc).__name__}: {exc}"
            )
            return

        try:
            raw_action = self.agent.get_action(observations)
        except Exception as exc:
            self.logger.warn(f"[rosnav_rl] get_action failed: {type(exc).__name__}: {exc}")
            return

        try:
            cmd = self.agent.space_manager.action_space_manager.decode_action(raw_action)
        except Exception as exc:
            self.logger.warn(f"[rosnav_rl] decode_action failed: {type(exc).__name__}: {exc}")
            return

        self._publish_cmd_vel(float(cmd[0]), float(cmd[1]), float(cmd[2]))

    def _lookup_robot_xy(self) -> tuple[float, float] | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                self.frame,
                self.base_frame,
                rclpy.time.Time(),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def _publish_subgoal(self, robot_xy: tuple[float, float], goal_xy: tuple[float, float]) -> None:
        dx = goal_xy[0] - robot_xy[0]
        dy = goal_xy[1] - robot_xy[1]
        dist = math.hypot(dx, dy)
        lookahead = max(self.min_lookahead, min(self.max_lookahead, self._last_speed * self.lookahead_time))
        if dist <= lookahead or dist < 1e-6:
            sx, sy = goal_xy
        else:
            scale = lookahead / dist
            sx = robot_xy[0] + dx * scale
            sy = robot_xy[1] + dy * scale

        msg = PoseStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = sx
        msg.pose.position.y = sy
        msg.pose.orientation.w = 1.0
        self._subgoal_pub.publish(msg)

    def _publish_cmd_vel(self, vx: float, vy: float, wz: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.angular.z = wz
        self._cmd_vel_pub.publish(msg)
        self._last_speed = math.hypot(vx, vy)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node(_NODE_NAME)
    try:
        ArenaInferenceNode(node)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
