"""Standalone in-process rosnav_rl inference node.

Runs the policy at a fixed rate, publishes `subgoal` (lookahead along the
straight robot→goal line) for the observation pipeline, and emits the
decoded command as `TwistStamped` on `~/cmd_vel`.

When `train_mode=true`, only the subgoal pipeline runs; the agent, ObservationManager,
and cmd_vel publisher are skipped — the training loop owns the policy in that case.
"""

from __future__ import annotations

import math
import time
from collections import deque

import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int16

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.dreamerv3.safety import SafetyCalibration, attenuation_factor
from rosnav_rl.model.model import RL_Model
from rosnav_rl.observations import ObservationManager
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.agent_paths import (
    resolve_agent_dir,
    resolve_observations_config_path,
)
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
        node.declare_parameter("log_step_latency", False)
        node.declare_parameter("safety_layer_enabled", False)
        node.declare_parameter("safety_calibration_path", "")
        node.declare_parameter("safety_lam", 1.0)
        node.declare_parameter("safety_gamma_min", 0.3)

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
        self._log_step_latency = bool(node.get_parameter("log_step_latency").get_parameter_value().bool_value)
        self._step_latencies: deque[float] = deque(maxlen=100)
        self._safety_enabled = bool(node.get_parameter("safety_layer_enabled").get_parameter_value().bool_value)
        self._safety_lam = float(node.get_parameter("safety_lam").get_parameter_value().double_value or 1.0)
        self._safety_gamma_min = float(node.get_parameter("safety_gamma_min").get_parameter_value().double_value or 0.3)
        self._safety_calib: SafetyCalibration | None = None
        self._u_ema: float | None = None

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
            if self._safety_enabled:
                self._safety_calib = self._load_safety_calibration()
        else:
            self.logger.info(
                "[rosnav_rl] subgoal-only mode (train_mode=true); agent and cmd_vel publisher skipped"
            )

        self._period = 1.0 / max(self.control_rate, 1e-3)
        self._timer = node.create_timer(self._period, self._step)
        self.logger.info(f"[rosnav_rl] running at {self.control_rate:.1f} Hz")

    def _load_agent(self) -> RL_Agent:
        model_dir = resolve_agent_dir(self.agent_name)
        agent = RL_Agent.from_agent_dir(model_dir)
        self._agent_spec = agent.spec
        self._agent_parameters: AgentParameters = agent.spec.parameters
        return agent

    def _load_observation_manager(self):
        obs_config_path = resolve_observations_config_path(self._agent_spec)
        with open(obs_config_path) as f:
            config = yaml.safe_load(f)
        return ObservationManager.from_config(
            config=config,
            node=self.node,
            ns=str(self.namespace),
            simulation_state_container=self._agent_parameters,
            wait_for_obs=False,
        )

    def _load_safety_calibration(self) -> SafetyCalibration:
        if type(self.agent.model).get_safety_signal is RL_Model.get_safety_signal:
            raise RuntimeError(
                f"rosnav_rl_inference: safety_layer_enabled=true but backend "
                f"'{type(self.agent.model).__name__}' does not override get_safety_signal() "
                "(SB3 backends never will; DreamerV3 needs behavior.expose_kl_surprise=true) "
                "-- the safety layer would silently no-op, so refusing to start instead."
            )
        path = self.node.get_parameter("safety_calibration_path").get_parameter_value().string_value
        if not path:
            raise RuntimeError(
                "rosnav_rl_inference: safety_layer_enabled=true requires 'safety_calibration_path'"
            )
        calib = SafetyCalibration.from_json(path)
        self.logger.info(f"[rosnav_rl] safety layer enabled, deployed threshold={calib.deployed:.4f}")
        return calib

    def _safety_gamma(self) -> float:
        """EMA-smooth the deploy-time uncertainty signal and convert it to a velocity scale.

        No-op (returns 1.0) unless the safety layer is enabled. Backend support for
        ``get_safety_signal`` is verified once at load time (`_load_safety_calibration`),
        so reaching here with it enabled means the signal is merely unavailable *this
        step* (e.g. before the first ``get_action`` call), not unsupported.
        """
        if not self._safety_enabled:
            return 1.0
        u = self.agent.model.get_safety_signal()
        if u is None:
            return 1.0
        beta = self._safety_calib.ema_beta
        u = float(u)
        self._u_ema = u if self._u_ema is None else beta * u + (1.0 - beta) * self._u_ema
        return attenuation_factor(
            self._u_ema, self._safety_calib.deployed, lam=self._safety_lam, gamma_min=self._safety_gamma_min
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg

    def _on_scenario_reset(self, _msg: Int16) -> None:
        self.agent.reset()
        self._last_speed = 0.0
        self._u_ema = None

    def _step(self) -> None:
        if not self._log_step_latency:
            self._step_impl()
            return
        start = time.perf_counter()
        self._step_impl()
        self._record_step_latency(time.perf_counter() - start)

    def _record_step_latency(self, elapsed: float) -> None:
        """Log a p50/p95 summary over the last 100 steps and warn on overrun.

        Enabled via the ``log_step_latency`` param; the timing/deque bookkeeping
        is skipped entirely when off so real-time ticks pay no overhead by default.
        """
        if elapsed > self._period:
            self.logger.warn(
                f"[rosnav_rl] step took {elapsed * 1000:.1f} ms, "
                f"overrunning control period {self._period * 1000:.1f} ms"
            )
        self._step_latencies.append(elapsed)
        if len(self._step_latencies) == self._step_latencies.maxlen:
            samples = sorted(self._step_latencies)
            p50 = samples[len(samples) // 2]
            p95 = samples[int(len(samples) * 0.95)]
            self.logger.info(
                f"[rosnav_rl] step latency over last {len(samples)} steps: "
                f"p50={p50 * 1000:.1f}ms p95={p95 * 1000:.1f}ms"
            )

    def _step_impl(self) -> None:
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
            # RL_Agent.get_action already returns a decoded action (both SB3
            # and DreamerV3 backends decode internally) — decoding again here
            # previously double-decoded the action, silently zeroing angular z.
            cmd = self.agent.get_action(observations)
        except Exception as exc:
            self.logger.warn(f"[rosnav_rl] get_action failed: {type(exc).__name__}: {exc}")
            return

        gamma = self._safety_gamma()
        self._publish_cmd_vel(float(cmd[0]) * gamma, float(cmd[1]) * gamma, float(cmd[2]))

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
