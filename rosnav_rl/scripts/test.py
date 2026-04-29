#!/usr/bin/env python3
import rosnav_rl

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg
from arena_training.arena_rosnav_rl.cfg.arena_cfg.task import TaskCfg
from arena_training.arena_rosnav_rl.cfg.train import TrainingCfg
from arena_training.arena_rosnav_rl.envs import GazeboEnv
from arena_training.arena_rosnav_rl.envs.wrappers import TimeSyncWrapper

# For other ROS 2 messages:
# from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from rosnav_rl.observations import (
    GoalCollector,
    LaserCollector,
    PeopleDataCollector,
    RobotPoseCollector,
)
from rosnav_rl.observations.collectors import *
from rosnav_rl.observations.observation_manager import (
    GenericObservation,
    ObservationManager,
)
from rosnav_rl.reward import RewardFunction


class SimpleSubscriber(Node):
    def __init__(self):
        super().__init__("simple_subscriber")

        self.observation = GenericObservation(
            initial_msg=LaserCollector.msg_data_class(),
            process_fnc=LaserCollector().safe_preprocess,
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.subscription = self.create_subscription(
            LaserScan,  # Change to your message type
            "/task_generator_node/jackal/lidar",  # Change to your topic name
            self.listener_callback,
            qos_profile=qos,
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info("Simple subscriber node started")

    def listener_callback(self, msg):
        # Process the received message
        self.observation.update(msg)
        self.get_logger().info(f"I heard: {msg.ranges}")


def main(args=None):
    rclpy.init(args=args)

    # node = SimpleSubscriber()

    node = Node("observation_test")

    om = ObservationManager(
        node=node,
        ns="/task_generator_node/jackal",
        obs_structure=[
            LaserCollector,
            RobotPoseCollector,
            GoalCollector,
            # PeopleDataCollector,
        ],
        wait_for_obs=True,
    )

    rclpy.spin_once(node, timeout_sec=0.1)
    rclpy.spin_once(node, timeout_sec=0.1)

    for i in range(30):
        obs = om.get_observations()
        # reward_function.get_reward(
        #     obs_dict=obs,
        #     simulation_state_container=om._simulation_state_container,
        # )
        rclpy.spin_once(node, timeout_sec=0.1)

    from arena_rclpy_mixins.spin import spin_node
    spin_node(node)


def test_env():
    rclpy.init()

    config_path = "/home/le/arena5_ws/src/Arena/arena_bringup/configs/training/sb_training_config.yaml"
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    training_cfg = TrainingCfg.model_validate(raw)

    # Build simulation state & populate agent_spec from robot description
    from scripts.create_test_agent import (  # noqa: PLC0415
        _build_simulation_state,
        _populate_agent_spec,
    )

    simulation_state_container = _build_simulation_state(training_cfg)
    _populate_agent_spec(training_cfg, simulation_state_container)

    rl_agent = rosnav_rl.RL_Agent(training_cfg.agent_config)

    env = GazeboEnv(
        ns="/task_generator_node/jackal",
        node=Node("gazebo_env_test"),
        space_manager=rl_agent.space_manager,
        reward_function=rl_agent.reward_function,
        simulation_state_container=simulation_state_container,
        max_steps_per_episode=3000,
    )
    # env = TimeSyncWrapper(env, 1.0)

    for _ in range(10):
        obs = env.reset()
        # print("Initial Observation:", obs)

        for i in range(3000):
            print(f"Step {i + 1} in the environment")
            action = rl_agent.space_manager.action_space.sample()
            # print("Action Sampled:", action)
            obs, reward, done, _, info = env.step(action=action)
            # print("Observation:", obs, "Reward:", reward, "Done:", done, "Info:", info)

            if done:
                break

        # Reset the environment for the next iteration


if __name__ == "__main__":
    test_env()
