"""Collector preprocessing of real ROS messages into observation arrays."""

import numpy as np
import pytest
from arena_people_msgs.msg import Pedestrian, Pedestrians
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3
from nav2_msgs.msg import CollisionMonitorState
from nav_msgs.msg import Odometry, Path
from people_msgs.msg import People, Person
from sensor_msgs.msg import Image, LaserScan
from tf_transformations import quaternion_from_euler
from trajectory_msgs.msg import JointTrajectoryPoint

from rosnav_rl.observations.data_sources.collectors import (
    ArenaPedestrianCollector,
    CollisionMonitorStateCollector,
    ImageColorCollector,
    JointStateCollector,
    LaserScanCollector,
    OdometryCollector,
    PathCollector,
    PeopleCollector,
    PoseStampedCollector,
    TwistCollector,
)


def yawed_pose(x, y, yaw):
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    return Pose(position=Point(x=x, y=y), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))


def test_laser_starts_empty_and_replaces_nan_and_inf_with_range_max():
    collector = LaserScanCollector("front_laser", topic="scan")
    initial = collector.get_observation()

    collector.update(LaserScan(ranges=[1.0, float("nan"), float("inf"), float("-inf"), 2.5], range_max=5.0))

    assert initial.shape == (0,)
    assert collector.get_observation().dtype == np.float32
    np.testing.assert_allclose(collector.get_observation(), [1.0, 5.0, 5.0, 5.0, 2.5])
    assert (collector.stale, collector.update_count) == (False, 1)


def test_odometry_becomes_a_planar_pose():
    collector = OdometryCollector("odom", topic="odom")
    odom = Odometry()
    odom.pose.pose = yawed_pose(1.0, -2.0, 0.5)

    collector.update(odom)

    pose = collector.get_observation()
    np.testing.assert_allclose([pose["x"], pose["y"], pose["yaw"]], [1.0, -2.0, 0.5], atol=1e-6)


def test_goal_pose_becomes_a_planar_pose():
    collector = PoseStampedCollector("goal", topic="goal")

    collector.update(PoseStamped(pose=yawed_pose(4.0, 3.0, -2.5)))

    pose = collector.get_observation()
    np.testing.assert_allclose([pose["x"], pose["y"], pose["yaw"]], [4.0, 3.0, -2.5], atol=1e-6)


def test_twist_keeps_linear_x_linear_y_and_angular_z():
    collector = TwistCollector("cmd_vel", topic="cmd_vel")

    collector.update(Twist(linear=Vector3(x=0.4, y=-0.1, z=9.0), angular=Vector3(x=7.0, z=0.8)))

    np.testing.assert_allclose(collector.get_observation(), [0.4, -0.1, 0.8])


def test_collision_monitor_matches_the_configured_polygon():
    collector = CollisionMonitorStateCollector("collision", topic="collision_state", state_key="stop")
    idle = collector.get_observation()

    collector.update(CollisionMonitorState(action_type=CollisionMonitorState.STOP, polygon_name="stop"))
    triggered = collector.get_observation()
    collector.update(CollisionMonitorState(action_type=CollisionMonitorState.SLOWDOWN, polygon_name="slowdown"))

    assert (idle, triggered, collector.get_observation()) == (False, True, False)
    assert collector.state_key == "stop"


def test_collision_monitor_ignores_polygons_that_only_contain_the_key():
    collector = CollisionMonitorStateCollector("collision", topic="collision_state", state_key="stop")

    collector.update(CollisionMonitorState(action_type=CollisionMonitorState.SLOWDOWN, polygon_name="nonstop_slowdown"))

    assert collector.get_observation() is False


def test_path_becomes_an_xy_array():
    collector = PathCollector("global_plan", topic="plan")
    initial = collector.get_observation()

    collector.update(
        Path(poses=[PoseStamped(pose=yawed_pose(1.0, 2.0, 0.0)), PoseStamped(pose=yawed_pose(-3.0, 0.5, 1.0))])
    )

    assert initial.shape == (0, 2)
    assert collector.get_observation().dtype == np.float32
    np.testing.assert_allclose(collector.get_observation(), [[1.0, 2.0], [-3.0, 0.5]])


def image(encoding, height, width, pixels, dtype=np.uint8):
    return Image(encoding=encoding, height=height, width=width, data=np.asarray(pixels, dtype=dtype).tobytes())


@pytest.mark.parametrize(
    ("encoding", "channels"),
    [("rgb8", 3), ("bgr8", 3), ("rgba8", 4), ("bgra8", 4), ("yuv422", 3)],
)
def test_multichannel_uint8_images_become_channels_first(encoding, channels):
    collector = ImageColorCollector("camera", topic="image")
    pixels = np.arange(2 * 3 * channels).reshape(2, 3, channels)

    collector.update(image(encoding, 2, 3, pixels))

    result = collector.get_observation()
    assert result.shape == (channels, 2, 3)
    np.testing.assert_array_equal(result, pixels.transpose(2, 0, 1))


def test_mono8_image_stays_height_by_width():
    collector = ImageColorCollector("camera", topic="image")

    collector.update(image("mono8", 2, 2, [[1, 2], [3, 4]]))

    np.testing.assert_array_equal(collector.get_observation(), [[1, 2], [3, 4]])


@pytest.mark.parametrize("encoding", ["32FC1", "mono32"])
def test_single_channel_float_images_keep_their_values(encoding):
    collector = ImageColorCollector("depth", topic="depth")

    collector.update(image(encoding, 1, 3, [[0.5, 1.25, 7.0]], dtype=np.float32))

    result = collector.get_observation()
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[0.5, 1.25, 7.0]])


def test_three_channel_float_image_becomes_channels_first():
    collector = ImageColorCollector("normals", topic="normals")
    pixels = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32)

    collector.update(image("32FC3", 1, 2, pixels, dtype=np.float32))

    np.testing.assert_array_equal(collector.get_observation(), pixels.transpose(2, 0, 1))


def test_empty_image_is_an_empty_array():
    collector = ImageColorCollector("camera", topic="image")

    collector.update(Image(encoding="rgb8", height=0, width=0))

    assert collector.get_observation().shape == (0,)


def test_people_and_arena_pedestrians_pass_through_unchanged():
    people = People(people=[Person(name="a", position=Point(x=1.0))])
    crowd = Pedestrians(pedestrians=[Pedestrian(name="b", id=3)])
    people_collector = PeopleCollector("people", topic="people")
    arena_collector = ArenaPedestrianCollector("pedestrians", topic="pedestrians")

    people_collector.update(people)
    arena_collector.update(crowd)

    assert people_collector.get_observation() is people
    assert arena_collector.get_observation() is crowd


def test_joint_states_map_each_joint_to_its_feedback():
    collector = JointStateCollector("joint_states", topic="arm_controller/controller_state")

    collector.update(
        JointTrajectoryControllerState(
            joint_names=["shoulder", "elbow"],
            feedback=JointTrajectoryPoint(positions=[0.5, -1.0], velocities=[0.1, 0.2], effort=[3.0, 4.0]),
        )
    )

    assert collector.get_observation() == {
        "shoulder": {"position": 0.5, "velocity": 0.1, "effort": 3.0},
        "elbow": {"position": -1.0, "velocity": 0.2, "effort": 4.0},
    }


def test_joint_states_without_velocity_or_effort_feedback_read_zero():
    collector = JointStateCollector("joint_states", topic="arm_controller/controller_state")

    collector.update(
        JointTrajectoryControllerState(joint_names=["shoulder"], feedback=JointTrajectoryPoint(positions=[0.25]))
    )

    assert collector.get_observation() == {"shoulder": {"position": 0.25, "velocity": 0.0, "effort": 0.0}}
