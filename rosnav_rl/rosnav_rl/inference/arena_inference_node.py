"""Standalone in-process rosnav_rl inference node.

Runs the policy at a fixed rate, publishes `subgoal` (lookahead along the
straight robot-to-goal line, the formula of StraightLineSubgoalGenerator) for display, and emits the
decoded command as `Twist` on `~/cmd_vel`. A chunked policy is queried once per
chunk, its commands play out one per tick while the observations keep encoding.
Under lockstep the sim waits for every tick: the node holds a hard `rosnav/<robot>`
channel at the control period and beats once per tick, after its command is out.

When `train_mode=true`, only the subgoal display runs; the agent, ObservationManager,
and cmd_vel publisher are skipped — the training loop owns the policy in that case.
"""

from __future__ import annotations

import math
import posixpath
from typing import TYPE_CHECKING

import numpy as np
import rclpy
import tf2_ros
import yaml
from arena_robots.lockstep_beat import LockstepBeat
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int16

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.factory.factory import (
    create_observation_manager_from_config,
    set_pedestrians_topic,
    set_robot_pose_frames,
)
from rosnav_rl.observations.utils.semantic import straight_line_subgoal
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.agent_paths import resolve_agent_dir, resolve_observations_config_path
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.utils import load_yaml

if TYPE_CHECKING:
    from rosnav_rl.observations.core.manager import ObservationManager


_NODE_NAME = "rosnav_rl_inference"
_RELEASE_TIMEOUT_S = 3.0


class ArenaInferenceNode:
    """Owns the per-robot RL agent and drives cmd_vel from observations."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.logger = node.get_logger()

        node.declare_parameter("agent", "")
        node.declare_parameter("namespace", "")
        node.declare_parameter("task_generator_node", "")
        node.declare_parameter("frame", "")
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
        self.task_generator_node = node.get_parameter("task_generator_node").get_parameter_value().string_value
        if not self.task_generator_node and not self.train_mode:
            raise RuntimeError("rosnav_rl_inference: parameter 'task_generator_node' is required")
        self.namespace = Namespace(node.get_parameter("namespace").get_parameter_value().string_value)
        self.frame = node.get_parameter("frame").get_parameter_value().string_value
        self.base_frame = node.get_parameter("base_frame").get_parameter_value().string_value or "base_link"
        self.control_rate = float(node.get_parameter("control_rate").get_parameter_value().double_value or 10.0)
        self.min_lookahead = float(node.get_parameter("min_lookahead_dist").get_parameter_value().double_value or 0.5)
        self.max_lookahead = float(node.get_parameter("max_lookahead_dist").get_parameter_value().double_value or 2.5)
        self.lookahead_time = float(node.get_parameter("lookahead_time").get_parameter_value().double_value or 1.5)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        self._goal: PoseStamped | None = None
        self._last_speed: float = 0.0
        self._queued_commands: list[np.ndarray] = []

        self._period = 1.0 / max(self.control_rate, 1e-3)
        self._beat: LockstepBeat | None = None

        self._subgoal_pub = node.create_publisher(PoseStamped, "subgoal", 1)
        self._goal_sub = node.create_subscription(
            PoseStamped,
            "goal_pose",
            self._on_goal,
            1,
        )

        if not self.train_mode:
            self._cmd_vel_pub = node.create_publisher(Twist, "cmd_vel", 1)
            self._reset_sub = node.create_subscription(
                Int16,
                f"{self.task_generator_node}/task_reset",
                self._on_task_reset,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
            )
            self.logger.info(f"[rosnav_rl] loading agent '{self.agent_name}'")
            self.agent = self._load_agent()
            self.logger.info("[rosnav_rl] agent loaded")
            self.observation_manager = self._load_observation_manager()
            self.logger.info("[rosnav_rl] observation manager ready")
            self._beat = LockstepBeat(node, "rosnav", grace_s=None)
        else:
            self._cmd_vel_sub = node.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 1)
            self.logger.info("[rosnav_rl] subgoal-only mode (train_mode=true); agent and cmd_vel publisher skipped")

        self._timer = node.create_timer(self._period, self._tick)
        self.logger.info(f"[rosnav_rl] running at {self.control_rate:.1f} Hz")

    async def start(self) -> None:
        """Hold the lockstep channel the ticks beat on."""
        if self._beat is not None:
            await self._beat.acquire(self._period)

    async def stop(self) -> None:
        if self._beat is not None:
            await self._beat.release()

    def _tick(self) -> None:
        stamp = self.node.get_clock().now().to_msg()
        try:
            self._step()
        finally:
            if self._beat is not None:
                self._beat.pulse(stamp=stamp)

    def _load_agent(self) -> RL_Agent:
        from arena_training.arena_rosnav_rl.cfg.train import TrainingCfg

        model_dir = resolve_agent_dir(self.agent_name)
        training_cfg = TrainingCfg.model_validate(load_yaml(model_dir / "training_config.yaml"))
        spec = training_cfg.agent_config
        self._agent_spec = spec
        self._agent_parameters: AgentParameters = spec.parameters
        agent = RL_Agent(spec)
        agent.load_model(path=model_dir / "best_model.zip")
        return agent

    def _load_observation_manager(self) -> ObservationManager:
        obs_config_path = resolve_observations_config_path(self._agent_spec)
        with open(obs_config_path) as f:
            config = yaml.safe_load(f)
        set_robot_pose_frames(config, self.frame, self.base_frame)
        set_pedestrians_topic(config, str(self.namespace))
        return create_observation_manager_from_config(
            config=config,
            node=self.node,
            ns=str(self.namespace),
            simulation_state_container=self._agent_parameters,
            wait_for_obs=False,
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._last_speed = math.hypot(msg.linear.x, msg.linear.y)

    def _on_task_reset(self, _msg: Int16) -> None:
        self.agent.reset()
        self._last_speed = 0.0
        self._queued_commands = []

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
            self.logger.warn(f"[rosnav_rl] get_observations failed: {type(exc).__name__}: {exc}")
            return
        observations["is_terminal"] = False

        if self._queued_commands:
            self.agent.space_manager.encode_observation(observations)
            cmd = self._queued_commands.pop(0)
        else:
            try:
                cmd = self.agent.get_action(observations, deterministic=self._agent_spec.deterministic_inference)
            except Exception as exc:
                self.logger.warn(f"[rosnav_rl] get_action failed: {type(exc).__name__}: {exc}")
                return
            if cmd.ndim == 2:
                self._queued_commands = list(cmd[1:])
                cmd = cmd[0]

        self._publish_cmd_vel(float(cmd[0]), float(cmd[1]), float(cmd[2]))

    def _lookup_robot_xy(self) -> tuple[float, float] | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._goal.header.frame_id,
                posixpath.join(self.frame, self.base_frame),
                rclpy.time.Time(),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def _publish_subgoal(self, robot_xy: tuple[float, float], goal_xy: tuple[float, float]) -> None:
        sx, sy = straight_line_subgoal(
            robot_xy, goal_xy, self._last_speed, self.min_lookahead, self.max_lookahead, self.lookahead_time
        )

        msg = PoseStamped()
        msg.header.frame_id = self._goal.header.frame_id
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = sx
        msg.pose.position.y = sy
        msg.pose.orientation.w = 1.0
        self._subgoal_pub.publish(msg)

    def _publish_cmd_vel(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self._cmd_vel_pub.publish(msg)
        self._last_speed = math.hypot(vx, vy)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = rclpy.create_node(_NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    inference: ArenaInferenceNode | None = None
    try:
        inference = ArenaInferenceNode(node)
        executor.spin_until_future_complete(executor.create_task(inference.start()))
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if inference is not None and rclpy.ok():
            executor.spin_until_future_complete(executor.create_task(inference.stop()), timeout_sec=_RELEASE_TIMEOUT_S)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
