"""Pedestrian generators on real people messages and the TF robot pose generator on a real buffer."""

import numpy as np
import pytest
import rclpy
from arena_people_msgs.msg import Pedestrian, Pedestrians
from geometry_msgs.msg import Point, Pose, TransformStamped, Twist, Vector3
from people_msgs.msg import People, Person
from rclpy.node import Node
from tf_transformations import quaternion_from_euler

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.data_sources.generators import (
    ArenaPedestrianDistanceGenerator,
    ArenaPedestrianLocationGenerator,
    ArenaPedestrianRelativeLocationGenerator,
    ArenaPedestrianRelativeVelGenerator,
    ArenaPedestrianStateGenerator,
    EndEffectorPoseTFGenerator,
    PedestrianDistanceGenerator,
    PedestrianLocationGenerator,
    PedestrianRelativeVelGenerator,
    PedestrianSocialStateGenerator,
    PedestrianTypeGenerator,
    RobotPoseTFGenerator,
)
from rosnav_rl.observations.utils.pose import Pose2DType

PARAMS = AgentParameters()


def robot(x, y, yaw):
    return np.array((x, y, yaw), dtype=Pose2DType)


def person(name, x=0.0, y=0.0, vx=0.0, vy=0.0, **tags):
    return Person(
        name=name,
        position=Point(x=x, y=y),
        velocity=Point(x=vx, y=vy),
        tagnames=list(tags),
        tags=list(tags.values()),
    )


def arena_ped(ped_id, x=0.0, y=0.0, vx=0.0, vy=0.0, state=0):
    return Pedestrian(
        name=f"ped_{ped_id}",
        id=ped_id,
        pose=Pose(position=Point(x=x, y=y)),
        twist=Twist(linear=Vector3(x=vx, y=vy)),
        animation_state=state,
    )


def test_world_locations_follow_a_changing_crowd_size():
    gen = PedestrianLocationGenerator(name="pedestrian_world_locations")

    two = gen._generate(People(people=[person("a", 1.0, 2.0), person("b", -3.0, 4.0)]), PARAMS).copy()
    moved = gen._generate(People(people=[person("a", 5.0, 6.0), person("b", 7.0, 8.0)]), PARAMS).copy()
    one = gen._generate(People(people=[person("a", 0.5, 0.25)]), PARAMS)

    np.testing.assert_allclose(two, [[1.0, 2.0], [-3.0, 4.0]])
    np.testing.assert_allclose(moved, [[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(one, [[0.5, 0.25]])


def test_relative_velocities_are_rotated_into_the_robot_frame_across_steps():
    gen = PedestrianRelativeVelGenerator(name="pedestrian_relative_velocities")
    people = People(people=[person("a", vx=1.0), person("b", vy=2.0)])

    first = gen._generate(robot(0.0, 0.0, np.pi / 2), people, PARAMS)
    second = gen._generate(robot(5.0, 5.0, 0.0), people, PARAMS)

    np.testing.assert_allclose(first, [[0.0, -1.0], [2.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(second, [[1.0, 0.0], [0.0, 2.0]], atol=1e-6)


def test_pedestrian_distance_keeps_the_nearest_member_of_each_group():
    gen = PedestrianDistanceGenerator(name="pedestrian_distances")
    people = People(people=[person("a", group_id="1"), person("b", group_id="1"), person("c", group_id="2")])

    result = gen._generate(np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 8.0]]), people, PARAMS)

    assert result == {"1": pytest.approx(1.0), "2": pytest.approx(10.0)}


def test_pedestrian_distance_rejects_mismatched_inputs():
    gen = PedestrianDistanceGenerator(name="pedestrian_distances")

    with pytest.warns(UserWarning, match="do not match"):
        result = gen._generate(np.array([[0.0, 1.0], [2.0, 0.0]]), People(people=[person("a", group_id="1")]), PARAMS)

    assert result == {}


def test_pedestrian_types_are_parsed_from_group_id_tags():
    gen = PedestrianTypeGenerator(name="pedestrian_types")

    types = gen._generate(People(people=[person("a", group_id="2"), person("b", group_id="0")]), PARAMS)

    assert types.tolist() == [2, 0]
    assert types.dtype == int


def test_pedestrian_types_reuse_the_cache_until_ids_or_count_change():
    gen = PedestrianTypeGenerator(name="pedestrian_types")
    people = People(people=[person("a", group_id="1"), person("b", group_id="3")])

    gen._generate(people, PARAMS)
    cached = gen._ped_types_buffer
    gen._generate(people, PARAMS)
    reused = gen._ped_types_buffer
    changed = gen._generate(People(people=[person("a", group_id="1"), person("b", group_id="4")]), PARAMS)

    assert reused is cached
    assert changed.tolist() == [1, 4]


def test_pedestrian_types_reset_on_an_empty_crowd():
    gen = PedestrianTypeGenerator(name="pedestrian_types")
    gen._generate(People(people=[person("a", group_id="1")]), PARAMS)

    result = gen._generate(People(people=[]), PARAMS)

    assert result.shape == (0,)
    assert (gen._ped_ids, gen._ped_types_buffer, gen._last_num_peds) == ([], None, 0)
    assert gen._generate(None, PARAMS).shape == (0,)


def test_pedestrian_types_without_group_tag_warn_and_are_empty():
    gen = PedestrianTypeGenerator(name="pedestrian_types")

    with pytest.warns(UserWarning, match="group ID not found"):
        result = gen._generate(People(people=[person("a", kind="adult")]), PARAMS)

    assert result.shape == (0,)
    assert gen._last_num_peds == 0


def test_pedestrian_types_with_non_numeric_group_warn_and_are_empty():
    gen = PedestrianTypeGenerator(name="pedestrian_types")

    with pytest.warns(UserWarning, match="group ID not found"):
        result = gen._generate(People(people=[person("a", group_id="adult")]), PARAMS)

    assert result.shape == (0,)
    assert gen._ped_ids == ["adult"]


def test_social_states_are_parsed_from_behavior_tags():
    gen = PedestrianSocialStateGenerator(name="pedestrian_social_states")

    states = gen._generate(People(people=[person("a", behavior="3"), person("b", behavior="1")]), PARAMS)

    assert states.tolist() == [3, 1]


def test_social_states_are_recomputed_when_the_crowd_changes():
    gen = PedestrianSocialStateGenerator(name="pedestrian_social_states")
    gen._generate(People(people=[person("a", behavior="3")]), PARAMS)

    states = gen._generate(People(people=[person("b", behavior="2"), person("c", behavior="0")]), PARAMS)

    assert states.tolist() == [2, 0]
    assert gen._ped_ids == ["b", "c"]


def test_social_states_follow_a_behavior_change_of_the_same_pedestrian():
    gen = PedestrianSocialStateGenerator(name="pedestrian_social_states")
    gen._generate(People(people=[person("a", behavior="1")]), PARAMS)

    assert gen._generate(People(people=[person("a", behavior="3")]), PARAMS).tolist() == [3]


def test_social_states_without_behavior_tag_warn_and_are_empty():
    gen = PedestrianSocialStateGenerator(name="pedestrian_social_states")

    with pytest.warns(UserWarning, match="social state not found"):
        result = gen._generate(People(people=[person("a", group_id="1")]), PARAMS)

    assert result.shape == (0,)


def test_social_states_reset_on_an_empty_crowd():
    gen = PedestrianSocialStateGenerator(name="pedestrian_social_states")
    gen._generate(People(people=[person("a", behavior="1")]), PARAMS)

    assert gen._generate(None, PARAMS).shape == (0,)
    assert (gen._ped_ids, gen._ped_social_states_buffer, gen._last_num_peds) == ([], None, 0)


def test_arena_relative_locations_rotate_into_the_robot_frame():
    gen = ArenaPedestrianRelativeLocationGenerator(name="arena_pedestrian_relative_locations")

    first = gen._generate(robot(1.0, 0.0, np.pi / 2), Pedestrians(pedestrians=[arena_ped(0, 1.0, 2.0)]), PARAMS).copy()
    crowd = Pedestrians(pedestrians=[arena_ped(0, 3.0, 0.0), arena_ped(1, 0.0, -2.0)])
    second = gen._generate(robot(0.0, 0.0, 0.0), crowd, PARAMS).copy()
    buffer = gen._pose_buffer
    turned = gen._generate(robot(0.0, 0.0, np.pi), crowd, PARAMS)

    np.testing.assert_allclose(first, [[2.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(second, [[3.0, 0.0], [0.0, -2.0]], atol=1e-6)
    np.testing.assert_allclose(turned, [[-3.0, 0.0], [0.0, 2.0]], atol=1e-6)
    assert gen._pose_buffer is buffer


def test_arena_world_locations_follow_the_crowd():
    gen = ArenaPedestrianLocationGenerator(name="arena_pedestrian_locations")

    first = gen._generate(Pedestrians(pedestrians=[arena_ped(0, 1.5, -2.5)]), PARAMS).copy()
    second = gen._generate(Pedestrians(pedestrians=[arena_ped(0, 1.0, 1.0), arena_ped(1, 2.0, 3.0)]), PARAMS)
    third = gen._generate(Pedestrians(pedestrians=[arena_ped(0, 4.0, 4.0), arena_ped(1, 5.0, 6.0)]), PARAMS)

    np.testing.assert_allclose(first, [[1.5, -2.5]])
    assert second is third
    np.testing.assert_allclose(third, [[4.0, 4.0], [5.0, 6.0]])


def test_arena_relative_velocities_rotate_with_robot_heading():
    gen = ArenaPedestrianRelativeVelGenerator(name="arena_pedestrian_relative_velocities")
    crowd = Pedestrians(pedestrians=[arena_ped(0, vx=1.0), arena_ped(1, vy=-0.5)])

    np.testing.assert_allclose(
        gen._generate(robot(0.0, 0.0, np.pi / 2), crowd, PARAMS), [[0.0, -1.0], [-0.5, 0.0]], atol=1e-6
    )
    np.testing.assert_allclose(
        gen._generate(robot(0.0, 0.0, np.pi), crowd, PARAMS), [[-1.0, 0.0], [0.0, 0.5]], atol=1e-6
    )


def test_arena_distances_keep_the_nearest_entry_per_id():
    gen = ArenaPedestrianDistanceGenerator(name="arena_pedestrian_distances")
    crowd = Pedestrians(pedestrians=[arena_ped(4), arena_ped(4), arena_ped(9), arena_ped(4)])

    result = gen._generate(np.array([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0], [0.0, 3.0]]), crowd, PARAMS)

    assert result == {4: 2.0, 9: 1.0}


def test_arena_distances_are_empty_without_locations():
    gen = ArenaPedestrianDistanceGenerator(name="arena_pedestrian_distances")

    assert gen._generate(np.array([]), Pedestrians(pedestrians=[arena_ped(1)]), PARAMS) == {}


def test_arena_animation_states_follow_each_message():
    gen = ArenaPedestrianStateGenerator(name="arena_pedestrian_states")

    first = gen._generate(Pedestrians(pedestrians=[arena_ped(0, state=Pedestrian.WALKING)]), PARAMS).copy()
    second = gen._generate(
        Pedestrians(pedestrians=[arena_ped(0, state=Pedestrian.PANIC), arena_ped(1, state=Pedestrian.IDLE)]), PARAMS
    ).copy()
    third = gen._generate(
        Pedestrians(pedestrians=[arena_ped(0, state=Pedestrian.RUNNING), arena_ped(1, state=Pedestrian.CURIOUS)]),
        PARAMS,
    )

    assert first.tolist() == [1]
    assert second.tolist() == [3, 0]
    assert third.tolist() == [2, 5]
    assert third.dtype == np.uint8


@pytest.mark.parametrize(
    ("generator", "args"),
    [
        (ArenaPedestrianRelativeLocationGenerator, (robot(0.0, 0.0, 0.0),)),
        (ArenaPedestrianLocationGenerator, ()),
        (ArenaPedestrianRelativeVelGenerator, (robot(0.0, 0.0, 0.0),)),
        (ArenaPedestrianStateGenerator, ()),
    ],
)
@pytest.mark.parametrize("crowd", [None, Pedestrians(pedestrians=[])])
def test_arena_generators_are_empty_without_pedestrians(generator, args, crowd):
    assert generator(name="arena")._generate(*args, crowd, PARAMS).shape == (0,)


@pytest.fixture
def node():
    if not rclpy.ok():
        rclpy.init()
    node = Node("robot_pose_tf_generator_test")
    yield node
    node.destroy_node()


def transform(parent, child, x, y, yaw):
    stamped = TransformStamped()
    stamped.header.frame_id = parent
    stamped.header.stamp.sec = 1
    stamped.child_frame_id = child
    stamped.transform.translation.x = x
    stamped.transform.translation.y = y
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    stamped.transform.rotation.x = qx
    stamped.transform.rotation.y = qy
    stamped.transform.rotation.z = qz
    stamped.transform.rotation.w = qw
    return stamped


def test_tf_generator_requires_a_node():
    with pytest.raises(ValueError, match="requires a ROS 2 node"):
        RobotPoseTFGenerator(name="robot_pose", source_frame="base_link")


def test_tf_generator_requires_a_source_frame(node):
    with pytest.raises(ValueError, match="non-empty source_frame"):
        RobotPoseTFGenerator(name="robot_pose", node=node)


def test_tf_generator_holds_the_origin_until_the_transform_exists(node):
    gen = RobotPoseTFGenerator(name="robot_pose", node=node, source_frame="base_link", target_frame="odom")

    pose = gen._generate(PARAMS)

    assert pose.tolist() == (0.0, 0.0, 0.0)
    assert gen._is_initialized is False


def test_tf_generator_reads_planar_pose_from_the_buffer(node):
    gen = RobotPoseTFGenerator(name="robot_pose", node=node, source_frame="base_link", target_frame="odom")
    gen._tf_buffer.set_transform(transform("odom", "base_link", 1.5, -2.0, 0.75), "test")

    pose = gen._generate(PARAMS)

    assert gen._is_initialized is True
    np.testing.assert_allclose([pose["x"], pose["y"], pose["yaw"]], [1.5, -2.0, 0.75], atol=1e-6)


def test_tf_generator_keeps_the_last_pose_when_lookup_fails(node):
    gen = RobotPoseTFGenerator(name="robot_pose", node=node, source_frame="base_link", target_frame="odom")
    gen._tf_buffer.set_transform(transform("odom", "base_link", 3.0, 1.0, -1.0), "test")
    gen._generate(PARAMS)

    gen._tf_buffer.clear()
    pose = gen._generate(PARAMS)

    np.testing.assert_allclose([pose["x"], pose["y"], pose["yaw"]], [3.0, 1.0, -1.0], atol=1e-6)


def pose3d_transform(parent, child, x, y, z):
    stamped = transform(parent, child, x, y, 0.0)
    stamped.transform.translation.z = z
    return stamped


def test_end_effector_generator_requires_a_source_frame(node):
    with pytest.raises(ValueError, match="non-empty source_frame"):
        EndEffectorPoseTFGenerator(name="end_effector_pose", node=node)


def test_end_effector_generator_reads_its_own_robots_tip_frame(node):
    gen = EndEffectorPoseTFGenerator(
        name="end_effector_pose", node=node, source_frame="env_1/ur5/tool0", target_frame="env_1/ur5/base_link"
    )
    gen._tf_buffer.set_transform(pose3d_transform("env_0/ur5/base_link", "env_0/ur5/tool0", 9.0, 9.0, 9.0), "test")
    gen._tf_buffer.set_transform(pose3d_transform("env_1/ur5/base_link", "env_1/ur5/tool0", 0.4, -0.1, 0.6), "test")

    pose = gen._generate(simulation_state_container=PARAMS)

    np.testing.assert_allclose(pose, [0.4, -0.1, 0.6, 0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_end_effector_generator_holds_its_pose_and_warns_once_while_the_frame_is_missing(node, capfd):
    gen = EndEffectorPoseTFGenerator(
        name="end_effector_pose", node=node, source_frame="env_1/ur5/tool0", target_frame="env_1/ur5/base_link"
    )
    gen._tf_buffer.set_transform(pose3d_transform("env_0/ur5/base_link", "env_0/ur5/tool0", 9.0, 9.0, 9.0), "test")

    first = gen._generate(simulation_state_container=PARAMS).copy()
    second = gen._generate(simulation_state_container=PARAMS)

    np.testing.assert_allclose(first, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(second, first)
    assert capfd.readouterr().err.count("Waiting for transform from 'env_1/ur5/tool0'") == 1
