"""Tests for feeding the pedestrian observations from arena_people_msgs."""

from __future__ import annotations

import importlib.resources

import numpy as np
import yaml
from arena_people_msgs.msg import Pedestrian, Pedestrians

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg
from rosnav_rl.observations.data_sources.generators import (
    ArenaPedestrianTypeDistanceGenerator,
    ArenaPedestrianTypeGenerator,
)
from rosnav_rl.observations.factory.factory import set_pedestrians_topic
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.spaces.observation_space.spaces import environment


def _pedestrians(count: int) -> Pedestrians:
    return Pedestrians(pedestrians=[Pedestrian(name=f"ped_{i}", id=i) for i in range(count)])


def test_every_arena_pedestrian_is_type_zero():
    types = ArenaPedestrianTypeGenerator(name="pedestrian_types")._generate(_pedestrians(3), AgentParameters())

    assert types.tolist() == [0, 0, 0]


def test_types_are_empty_before_the_first_pedestrians_message():
    types = ArenaPedestrianTypeGenerator(name="pedestrian_types")._generate(None, AgentParameters())

    assert types.shape == (0,)


def test_type_distance_is_the_nearest_pedestrian_under_type_zero():
    locations = np.array([[3.0, 4.0], [0.0, 2.0], [-6.0, 0.0]], dtype=np.float32)

    distances = ArenaPedestrianTypeDistanceGenerator(name="pedestrian_distances")._generate(
        locations, AgentParameters()
    )

    assert distances == {0: 2.0}


def test_type_distance_is_empty_without_pedestrians():
    distances = ArenaPedestrianTypeDistanceGenerator(name="pedestrian_distances")._generate(
        np.array([]), AgentParameters()
    )

    assert distances == {}


def test_pedestrians_topic_is_rebased_onto_the_env_namespace():
    config = {
        "datasources": {
            "arena_pedestrian_detections": {"type": "arena_people_msgs/Pedestrians", "params": {"topic": "arena_peds"}},
            "front_laser": {"type": "sensor_msgs/LaserScan", "params": {"topic": "lidar"}},
        }
    }

    set_pedestrians_topic(config, "/arena/env_0/task_generator_node/jackal")

    assert config["datasources"]["arena_pedestrian_detections"]["params"]["topic"] == "/arena/env_0/arena_peds"
    assert config["datasources"]["front_laser"]["params"] == {"topic": "lidar"}


def test_bundled_observations_provide_what_the_dreamer_environment_spaces_require(tmp_path):
    framework = DreamerV3Cfg()
    framework.general.logdir = tmp_path
    agent = RL_Agent(
        AgentConfig(
            name="dreamer_sources",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=360, laser_max_range=30.0),
            framework=framework,
        )
    )
    agent.initialize_model()
    config = yaml.safe_load((importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml").read_text())
    aliases = config["aliases"]
    provided = (set(config["datasources"]) - set(aliases.values())) | set(aliases)

    required = {
        key
        for space in agent.model.observation_space_list
        if space.__module__.startswith(environment.__name__)
        for key in space.requires
    }

    assert required <= provided
