#!/usr/bin/env python3
import asyncio
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
from rl_utils.cfg import RobotCfg
from rl_utils.cfg.arena_cfg.task import TaskCfg
from rl_utils.envs import GazeboEnv
from rl_utils.envs.wrappers import TimeSyncWrapper
from rl_utils.tools.states import get_arena_states

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

    reward_function = RewardFunction(
        function_dict=reward_dict,
    )

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
        obs = asyncio.run(om.get_observations())
        # reward_function.get_reward(
        #     obs_dict=obs,
        #     simulation_state_container=om._simulation_state_container,
        # )
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_env():
    rclpy.init()
    simulation_state_container = get_arena_states(
        goal_radius=0.33,
        max_steps=350,
        is_discrete=False,
        safety_distance=1.0,
        robot_cfg=RobotCfg(),
        task_modules_cfg=TaskCfg(),
    )
    agent_state_cont = simulation_state_container.to_agent_state_container()
    agent_state_cont.action_space.actions = {
        "linear_range": [-2.0, 2.0],
        "angular_range": [-4.0, 4.0],
    }
    rl_agent = rosnav_rl.RL_Agent(
        agent_cfg=rosnav_rl.AgentCfg.model_validate(
            yaml.safe_load(
                open(
                    "/home/le/arena4_ws_exp/src/arena/arena-rosnav/arena_bringup/configs/training/sb_training_config.yaml",
                    "r",
                )
            )["agent_cfg"]
        ),
        agent_state_container=agent_state_cont,
    )

    env = GazeboEnv(
        node=Node("gazebo_env_test"),
        ns="/task_generator_node/jackal",
        space_manager=rl_agent.space_manager,
        reward_function=rl_agent.reward_function,
        simulation_state_container=simulation_state_container,
        max_steps_per_episode=3000,
    )
    env = TimeSyncWrapper(env)

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
