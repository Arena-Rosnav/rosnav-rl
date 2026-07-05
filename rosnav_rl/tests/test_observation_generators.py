"""Comprehensive tests for observation data sources (generators).

Tests cover:
- GoalLocationInRobotFrameGenerator
- SubgoalLocationInRobotFrameGenerator
- DistAngleToGoalGenerator
- DistAngleToSubgoalGenerator
- LaserSafeDistanceGenerator
- PedestrianRelativeLocationGenerator
- PedestrianLocationGenerator
- PedestrianRelativeVelGenerator
- PedestrianRelativeVelXGenerator / VelYGenerator
- PedestrianDistanceGenerator
- Coordinate transform utilities (get_relative_pos_to_robot, get_relative_vel_to_robot)
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from conftest import SimulationStateContainerStub

from geometry_msgs.msg import Point
from people_msgs.msg import Person, People

from rosnav_rl.observations.utils.semantic import (
    get_relative_pos_to_robot,
    get_relative_vel_to_robot,
)
from rosnav_rl.observations.utils.pose import Pose2DType
from rosnav_rl.observations.data_sources.generators import (
    GoalLocationInRobotFrameGenerator,
    SubgoalLocationInRobotFrameGenerator,
    DistAngleToGoalGenerator,
    DistAngleToSubgoalGenerator,
    LaserSafeDistanceGenerator,
    PedestrianRelativeLocationGenerator,
    PedestrianLocationGenerator,
    PedestrianRelativeVelGenerator,
    PedestrianRelativeVelXGenerator,
    PedestrianRelativeVelYGenerator,
    PedestrianDistanceGenerator,
    PedestrianGraphNodeGenerator,
)


# =====================================================================
#  Helper — create structured numpy pose
# =====================================================================

def make_pose(x, y, yaw):
    """Create structured numpy array with Pose2DType dtype."""
    p = np.zeros(1, dtype=Pose2DType)[0]
    p["x"] = x
    p["y"] = y
    p["yaw"] = yaw
    return p


# =====================================================================
#  get_relative_pos_to_robot  (utility function)
# =====================================================================

class TestGetRelativePosToRobot:
    def test_identity_transform(self):
        """Robot at origin facing +x => positions unchanged."""
        robot = make_pose(0.0, 0.0, 0.0)
        points = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        result = get_relative_pos_to_robot(robot, points)

        np.testing.assert_allclose(result[0], [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result[1], [0.0, 1.0], atol=1e-6)

    def test_rotation_90_degrees(self):
        """Robot at origin facing +y (yaw=π/2)."""
        robot = make_pose(0.0, 0.0, np.pi / 2)
        # Point at (1, 0) in world => should be (0, -1) in robot frame
        points = np.array([[1.0, 0.0, 1.0]])

        result = get_relative_pos_to_robot(robot, points)

        np.testing.assert_allclose(result[0], [0.0, -1.0], atol=1e-6)

    def test_translation(self):
        """Robot at (2, 3) facing +x => points shifted."""
        robot = make_pose(2.0, 3.0, 0.0)
        points = np.array([[5.0, 3.0, 1.0]])  # 3m ahead in world

        result = get_relative_pos_to_robot(robot, points)

        np.testing.assert_allclose(result[0], [3.0, 0.0], atol=1e-6)

    def test_rotation_and_translation(self):
        """Robot at (1, 1) facing +y (yaw=π/2), point at (1, 3)."""
        robot = make_pose(1.0, 1.0, np.pi / 2)
        # In robot frame: point should be (2, 0) — directly ahead
        points = np.array([[1.0, 3.0, 1.0]])

        result = get_relative_pos_to_robot(robot, points)

        np.testing.assert_allclose(result[0], [2.0, 0.0], atol=1e-6)

    def test_output_buffer(self):
        """Test that output_buffer is used when provided."""
        robot = make_pose(0.0, 0.0, 0.0)
        points = np.array([[2.0, 3.0, 1.0]])
        buffer = np.empty((1, 2), dtype=np.float64)

        result = get_relative_pos_to_robot(robot, points, output_buffer=buffer)

        np.testing.assert_allclose(result, [[2.0, 3.0]], atol=1e-6)
        # Should be writing into the same buffer
        assert result.base is buffer or np.shares_memory(result, buffer)

    def test_multiple_points(self):
        """Multiple points transformed at once."""
        robot = make_pose(0.0, 0.0, 0.0)
        points = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ])

        result = get_relative_pos_to_robot(robot, points)

        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0], [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result[1], [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(result[2], [-1.0, -1.0], atol=1e-6)

    def test_full_rotation_360(self):
        """Full rotation back to identity."""
        robot = make_pose(0.0, 0.0, 2 * np.pi)
        points = np.array([[1.0, 0.0, 1.0]])

        result = get_relative_pos_to_robot(robot, points)

        np.testing.assert_allclose(result[0], [1.0, 0.0], atol=1e-5)

    def test_behind_robot(self):
        """Point behind robot facing +x."""
        robot = make_pose(5.0, 0.0, 0.0)
        points = np.array([[3.0, 0.0, 1.0]])  # behind

        result = get_relative_pos_to_robot(robot, points)

        np.testing.assert_allclose(result[0], [-2.0, 0.0], atol=1e-6)


# =====================================================================
#  get_relative_vel_to_robot  (utility function)
# =====================================================================

class TestGetRelativeVelToRobot:
    def test_identity_rotation(self):
        """Robot facing +x, velocities unchanged."""
        robot = make_pose(0.0, 0.0, 0.0)
        vels = np.array([[1.0, 0.5]])

        result = get_relative_vel_to_robot(robot, vels)

        np.testing.assert_allclose(result, [[1.0, 0.5]], atol=1e-6)

    def test_rotated_velocity(self):
        """Robot facing +y, world +x velocity => robot -y velocity."""
        robot = make_pose(0.0, 0.0, np.pi / 2)
        vels = np.array([[1.0, 0.0]])

        result = get_relative_vel_to_robot(robot, vels)

        np.testing.assert_allclose(result[0], [0.0, -1.0], atol=1e-6)

    def test_empty_velocities(self):
        """Empty input returns empty output."""
        robot = make_pose(0.0, 0.0, 0.0)
        vels = np.empty((0, 2))

        result = get_relative_vel_to_robot(robot, vels)

        assert result.shape == (0, 2)

    def test_multiple_velocities(self):
        """Transform multiple velocity vectors."""
        robot = make_pose(0.0, 0.0, 0.0)
        vels = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])

        result = get_relative_vel_to_robot(robot, vels)

        assert result.shape == (3, 2)


# =====================================================================
#  GoalLocationInRobotFrameGenerator
# =====================================================================

class TestGoalLocationInRobotFrameGenerator:
    def test_goal_ahead(self):
        gen = GoalLocationInRobotFrameGenerator(name="test_goal")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        goal_pose = {"x": 5.0, "y": 0.0}
        sim = SimulationStateContainerStub()

        result = gen._generate(
            robot_pose=robot_pose,
            goal_pose=goal_pose,
            simulation_state_container=sim,
        )

        assert result.shape == (2,)
        np.testing.assert_allclose(result, [5.0, 0.0], atol=1e-5)

    def test_goal_behind_rotated_robot(self):
        gen = GoalLocationInRobotFrameGenerator(name="test_goal")
        robot_pose = make_pose(0.0, 0.0, np.pi)
        goal_pose = {"x": 3.0, "y": 0.0}
        sim = SimulationStateContainerStub()

        result = gen._generate(
            robot_pose=robot_pose,
            goal_pose=goal_pose,
            simulation_state_container=sim,
        )

        # Robot facing -x, point at +x => should be behind
        np.testing.assert_allclose(result[0], -3.0, atol=1e-5)

    def test_goal_to_the_left(self):
        gen = GoalLocationInRobotFrameGenerator(name="test_goal")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        goal_pose = {"x": 0.0, "y": 3.0}
        sim = SimulationStateContainerStub()

        result = gen._generate(
            robot_pose=robot_pose,
            goal_pose=goal_pose,
            simulation_state_container=sim,
        )

        np.testing.assert_allclose(result, [0.0, 3.0], atol=1e-5)


# =====================================================================
#  SubgoalLocationInRobotFrameGenerator
# =====================================================================

class TestSubgoalLocationInRobotFrameGenerator:
    def test_subgoal_transform(self):
        gen = SubgoalLocationInRobotFrameGenerator(name="test_subgoal")
        robot_pose = make_pose(1.0, 1.0, 0.0)
        subgoal_pose = {"x": 4.0, "y": 1.0}
        sim = SimulationStateContainerStub()

        result = gen._generate(
            subgoal_pose=subgoal_pose,
            robot_pose=robot_pose,
            simulation_state_container=sim,
        )

        assert result.shape == (2,)
        np.testing.assert_allclose(result, [3.0, 0.0], atol=1e-5)


# =====================================================================
#  DistAngleToGoalGenerator
# =====================================================================

class TestDistAngleToGoalGenerator:
    def test_goal_ahead(self):
        gen = DistAngleToGoalGenerator(name="test_dist_angle")
        goal_in_frame = np.array([3.0, 0.0])
        sim = SimulationStateContainerStub()

        result = gen._generate(
            goal_in_robot_frame=goal_in_frame,
            simulation_state_container=sim,
        )

        assert result.shape == (2,)
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(0.0)

    def test_goal_to_left(self):
        gen = DistAngleToGoalGenerator(name="test_dist_angle")
        goal_in_frame = np.array([0.0, 3.0])
        sim = SimulationStateContainerStub()

        result = gen._generate(
            goal_in_robot_frame=goal_in_frame,
            simulation_state_container=sim,
        )

        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(np.pi / 2)

    def test_goal_behind(self):
        gen = DistAngleToGoalGenerator(name="test_dist_angle")
        goal_in_frame = np.array([-2.0, 0.0])
        sim = SimulationStateContainerStub()

        result = gen._generate(
            goal_in_robot_frame=goal_in_frame,
            simulation_state_container=sim,
        )

        assert result[0] == pytest.approx(2.0)
        assert abs(result[1]) == pytest.approx(np.pi)

    def test_diagonal_goal(self):
        gen = DistAngleToGoalGenerator(name="test_dist_angle")
        goal_in_frame = np.array([1.0, 1.0])
        sim = SimulationStateContainerStub()

        result = gen._generate(
            goal_in_robot_frame=goal_in_frame,
            simulation_state_container=sim,
        )

        assert result[0] == pytest.approx(np.sqrt(2))
        assert result[1] == pytest.approx(np.pi / 4)

    def test_zero_distance(self):
        gen = DistAngleToGoalGenerator(name="test_dist_angle")
        goal_in_frame = np.array([0.0, 0.0])
        sim = SimulationStateContainerStub()

        result = gen._generate(
            goal_in_robot_frame=goal_in_frame,
            simulation_state_container=sim,
        )

        assert result[0] == pytest.approx(0.0)


# =====================================================================
#  DistAngleToSubgoalGenerator
# =====================================================================

class TestDistAngleToSubgoalGenerator:
    def test_subgoal_metrics(self):
        gen = DistAngleToSubgoalGenerator(name="test_dist_angle_sub")
        subgoal_in_frame = np.array([3.0, 4.0])
        sim = SimulationStateContainerStub()

        result = gen._generate(
            subgoal_in_robot_frame=subgoal_in_frame,
            simulation_state_container=sim,
        )

        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(np.arctan2(4.0, 3.0))


# =====================================================================
#  LaserSafeDistanceGenerator
# =====================================================================

class TestLaserSafeDistanceGenerator:
    def test_safe(self):
        gen = LaserSafeDistanceGenerator(name="test_laser_safe")
        laser = np.array([1.0, 2.0, 3.0, 4.0])
        sim = SimulationStateContainerStub(safety_distance=0.5)

        result = gen._generate(
            front_laser=laser,
            simulation_state_container=sim,
        )

        assert result == False  # min(laser)=1.0 > 0.5

    def test_violation(self):
        gen = LaserSafeDistanceGenerator(name="test_laser_safe")
        laser = np.array([0.3, 2.0, 3.0])
        sim = SimulationStateContainerStub(safety_distance=0.5)

        result = gen._generate(
            front_laser=laser,
            simulation_state_container=sim,
        )

        assert result == True  # min(laser)=0.3 <= 0.5

    def test_at_boundary(self):
        gen = LaserSafeDistanceGenerator(name="test_laser_safe")
        laser = np.array([0.5, 2.0, 3.0])
        sim = SimulationStateContainerStub(safety_distance=0.5)

        result = gen._generate(
            front_laser=laser,
            simulation_state_container=sim,
        )

        assert result == True  # min=0.5 <= 0.5

    def test_empty_laser(self):
        gen = LaserSafeDistanceGenerator(name="test_laser_safe")
        laser = np.array([])
        sim = SimulationStateContainerStub(safety_distance=0.5)

        result = gen._generate(
            front_laser=laser,
            simulation_state_container=sim,
        )

        assert result == False  # Empty => inf > 0.5


# =====================================================================
#  PedestrianRelativeLocationGenerator
# =====================================================================

class TestPedestrianRelativeLocationGenerator:
    def test_basic_transform(self):
        gen = PedestrianRelativeLocationGenerator(name="test_ped_rel")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        sim = SimulationStateContainerStub()

        person = Person(position=Point(x=3.0, y=0.0))
        people_data = People(people=[person])

        result = gen._generate(
            robot_pose=robot_pose,
            people_data=people_data,
            simulation_state_container=sim,
        )

        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [3.0, 0.0], atol=1e-5)

    def test_no_pedestrians(self):
        gen = PedestrianRelativeLocationGenerator(name="test_ped_rel")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        sim = SimulationStateContainerStub()

        people_data = People(people=[])

        result = gen._generate(
            robot_pose=robot_pose,
            people_data=people_data,
            simulation_state_container=sim,
        )

        assert len(result) == 0

    def test_multiple_pedestrians(self):
        gen = PedestrianRelativeLocationGenerator(name="test_ped_rel")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        sim = SimulationStateContainerStub()

        people_data = People(people=[
            Person(position=Point(x=1.0, y=2.0)),
            Person(position=Point(x=-1.0, y=3.0)),
        ])

        result = gen._generate(
            robot_pose=robot_pose,
            people_data=people_data,
            simulation_state_container=sim,
        )

        assert result.shape == (2, 2)

    def test_buffer_reuse_same_count(self):
        gen = PedestrianRelativeLocationGenerator(name="test_ped_rel")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        sim = SimulationStateContainerStub()

        people_data = People(people=[
            Person(position=Point(x=1.0, y=0.0)),
        ])

        # First call creates buffers
        gen._generate(robot_pose=robot_pose, people_data=people_data,
                      simulation_state_container=sim)
        buf1 = gen._pose_buffer

        # Second call with same number should reuse buffers
        gen._generate(robot_pose=robot_pose, people_data=people_data,
                      simulation_state_container=sim)
        buf2 = gen._pose_buffer

        assert buf1 is buf2

    def test_none_people_data(self):
        gen = PedestrianRelativeLocationGenerator(name="test_ped_rel")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        sim = SimulationStateContainerStub()

        result = gen._generate(
            robot_pose=robot_pose,
            people_data=None,
            simulation_state_container=sim,
        )

        assert len(result) == 0


# =====================================================================
#  PedestrianLocationGenerator
# =====================================================================

class TestPedestrianLocationGenerator:
    def test_world_locations(self):
        gen = PedestrianLocationGenerator(name="test_ped_world")
        sim = SimulationStateContainerStub()

        people_data = People(people=[
            Person(position=Point(x=1.0, y=2.0)),
            Person(position=Point(x=3.0, y=4.0)),
        ])

        result = gen._generate(people_data=people_data, simulation_state_container=sim)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [1.0, 2.0], atol=1e-5)
        np.testing.assert_allclose(result[1], [3.0, 4.0], atol=1e-5)

    def test_empty(self):
        gen = PedestrianLocationGenerator(name="test_ped_world")
        sim = SimulationStateContainerStub()

        result = gen._generate(people_data=People(), simulation_state_container=sim)
        assert len(result) == 0


# =====================================================================
#  PedestrianRelativeVelGenerator
# =====================================================================

class TestPedestrianRelativeVelGenerator:
    def test_velocity_transform(self):
        gen = PedestrianRelativeVelGenerator(name="test_ped_vel")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        sim = SimulationStateContainerStub()

        person = Person(velocity=Point(x=1.0, y=0.5))
        people_data = People(people=[person])

        result = gen._generate(
            robot_pose=robot_pose,
            people_data=people_data,
            simulation_state_container=sim,
        )

        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [1.0, 0.5], atol=1e-5)

    def test_empty(self):
        gen = PedestrianRelativeVelGenerator(name="test_ped_vel")
        robot_pose = make_pose(0.0, 0.0, 0.0)
        sim = SimulationStateContainerStub()

        result = gen._generate(
            robot_pose=robot_pose,
            people_data=People(),
            simulation_state_container=sim,
        )

        assert len(result) == 0


# =====================================================================
#  PedestrianRelativeVelXGenerator / VelYGenerator
# =====================================================================

class TestPedestrianVelComponentGenerators:
    def test_vel_x_extraction(self):
        gen = PedestrianRelativeVelXGenerator(name="test_vel_x")
        sim = SimulationStateContainerStub()
        vels = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = gen._generate(
            pedestrian_relative_velocities=vels,
            simulation_state_container=sim,
        )

        np.testing.assert_allclose(result, [1.0, 3.0])

    def test_vel_y_extraction(self):
        gen = PedestrianRelativeVelYGenerator(name="test_vel_y")
        sim = SimulationStateContainerStub()
        vels = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = gen._generate(
            pedestrian_relative_velocities=vels,
            simulation_state_container=sim,
        )

        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_vel_x_empty(self):
        gen = PedestrianRelativeVelXGenerator(name="test_vel_x")
        sim = SimulationStateContainerStub()

        result = gen._generate(
            pedestrian_relative_velocities=np.array([]),
            simulation_state_container=sim,
        )

        assert len(result) == 0

    def test_vel_y_empty(self):
        gen = PedestrianRelativeVelYGenerator(name="test_vel_y")
        sim = SimulationStateContainerStub()

        result = gen._generate(
            pedestrian_relative_velocities=np.array([]),
            simulation_state_container=sim,
        )

        assert len(result) == 0


# =====================================================================
#  PedestrianDistanceGenerator
# =====================================================================

class TestPedestrianDistanceGenerator:
    def test_basic_distances(self):
        gen = PedestrianDistanceGenerator(name="test_ped_dist")
        sim = SimulationStateContainerStub()

        locations = np.array([[3.0, 4.0], [1.0, 0.0]])
        people = People(people=[
            Person(name="a", tags=["1"], tagnames=["group_id"]),
            Person(name="b", tags=["2"], tagnames=["group_id"]),
        ])

        result = gen._generate(
            pedestrian_relative_locations=locations,
            people_data=people,
            simulation_state_container=sim,
        )

        assert "1" in result
        assert "2" in result
        assert result["1"] == pytest.approx(5.0)  # sqrt(9+16)
        assert result["2"] == pytest.approx(1.0)

    def test_same_group_min_distance(self):
        gen = PedestrianDistanceGenerator(name="test_ped_dist")
        sim = SimulationStateContainerStub()

        locations = np.array([[3.0, 4.0], [1.0, 0.0]])  # dist: 5.0, 1.0
        people = People(people=[
            Person(name="a", tags=["1"], tagnames=["group_id"]),
            Person(name="b", tags=["1"], tagnames=["group_id"]),
        ])

        result = gen._generate(
            pedestrian_relative_locations=locations,
            people_data=people,
            simulation_state_container=sim,
        )

        assert result["1"] == pytest.approx(1.0)  # min of group

    def test_empty(self):
        gen = PedestrianDistanceGenerator(name="test_ped_dist")
        sim = SimulationStateContainerStub()

        result = gen._generate(
            pedestrian_relative_locations=np.array([]),
            people_data=People(),
            simulation_state_container=sim,
        )

        assert result == {}

    def test_missing_group_id_tag(self):
        gen = PedestrianDistanceGenerator(name="test_ped_dist")
        sim = SimulationStateContainerStub()

        locations = np.array([[1.0, 0.0]])
        people = People(people=[
            Person(name="a", tags=["foo"], tagnames=["other_tag"]),
        ])

        result = gen._generate(
            pedestrian_relative_locations=locations,
            people_data=people,
            simulation_state_container=sim,
        )

        # Should gracefully skip persons without group_id
        assert result == {}


# =====================================================================
#  Integration: goal transform pipeline
# =====================================================================

class TestGoalPipeline:
    """Test the full pipeline: pose → relative position → dist/angle."""

    def test_full_pipeline_straight_ahead(self):
        """Goal straight ahead of robot at origin."""
        robot_pose = make_pose(0.0, 0.0, 0.0)
        goal_pose = {"x": 5.0, "y": 0.0}
        sim = SimulationStateContainerStub()

        # Step 1: Goal in robot frame
        gen1 = GoalLocationInRobotFrameGenerator(name="goal_rf")
        goal_rf = gen1._generate(
            robot_pose=robot_pose, goal_pose=goal_pose,
            simulation_state_container=sim,
        )

        # Step 2: Dist/angle
        gen2 = DistAngleToGoalGenerator(name="dist_angle")
        da = gen2._generate(
            goal_in_robot_frame=goal_rf,
            simulation_state_container=sim,
        )

        assert da[0] == pytest.approx(5.0)
        assert da[1] == pytest.approx(0.0)

    def test_full_pipeline_goal_left_45_deg(self):
        """Goal at 45 degrees to the left."""
        robot_pose = make_pose(0.0, 0.0, 0.0)
        goal_pose = {"x": 3.0, "y": 3.0}
        sim = SimulationStateContainerStub()

        gen1 = GoalLocationInRobotFrameGenerator(name="goal_rf")
        goal_rf = gen1._generate(
            robot_pose=robot_pose, goal_pose=goal_pose,
            simulation_state_container=sim,
        )

        gen2 = DistAngleToGoalGenerator(name="dist_angle")
        da = gen2._generate(
            goal_in_robot_frame=goal_rf,
            simulation_state_container=sim,
        )

        assert da[0] == pytest.approx(3.0 * np.sqrt(2))
        assert da[1] == pytest.approx(np.pi / 4)


# =====================================================================
#  PedestrianGraphNodeGenerator
# =====================================================================

def _people(*entries):
    """Build a People msg from (x, y, vx, vy, name[, tag]) tuples."""
    persons = []
    for entry in entries:
        x, y, vx, vy, name = entry[:5]
        tag = entry[5] if len(entry) > 5 else "moving"
        persons.append(
            Person(
                position=Point(x=float(x), y=float(y), z=0.0),
                velocity=Point(x=float(vx), y=float(vy), z=0.0),
                name=str(name),
                tags=[tag],
                tagnames=["behavior"],
            )
        )
    return People(people=persons)


class TestPedestrianGraphNodeGenerator:
    def _gen(self, max_peds=8, include_social_state=True):
        return PedestrianGraphNodeGenerator(
            name="test_peds",
            max_peds=max_peds,
            include_social_state=include_social_state,
        )

    def _call(self, gen, robot_pose, people_data):
        return gen._generate(
            robot_pose=robot_pose,
            people_data=people_data,
            simulation_state_container=SimulationStateContainerStub(),
        )

    # ------------------------------------------------------------------ shape
    def test_output_shape_with_social_state(self):
        gen = self._gen(max_peds=8, include_social_state=True)
        out = self._call(gen, make_pose(0, 0, 0), _people((1, 0, 0, 0, "p0")))
        assert out.shape == (8, 6)  # 4 kin + 1 social + 1 validity

    def test_output_shape_without_social_state(self):
        gen = self._gen(max_peds=4, include_social_state=False)
        out = self._call(gen, make_pose(0, 0, 0), _people((1, 0, 0, 0, "p0")))
        assert out.shape == (4, 5)  # 4 kin + 1 validity

    # ------------------------------------------------------------------ empty
    def test_empty_people_all_zeros(self):
        gen = self._gen()
        out = self._call(gen, make_pose(0, 0, 0), People(people=[]))
        assert out.shape == (8, 6)
        assert (out == 0).all()

    def test_none_people_all_zeros(self):
        gen = self._gen()
        out = self._call(gen, make_pose(0, 0, 0), None)
        assert out.shape == (8, 6)
        assert (out == 0).all()

    # ------------------------------------------------------------------ padding
    def test_fewer_peds_than_max_pads_remainder(self):
        gen = self._gen(max_peds=8)
        out = self._call(gen, make_pose(0, 0, 0), _people((1, 0, 0, 0, "p0"), (2, 0, 0, 0, "p1")))
        assert out[0, -1] == 1.0  # valid
        assert out[1, -1] == 1.0  # valid
        assert (out[2:, :] == 0).all()  # padded rows all zero

    def test_more_peds_than_max_keeps_nearest(self):
        gen = self._gen(max_peds=2)
        # 4 peds at distances 5, 1, 3, 0.5 — nearest 2 are 0.5 and 1.0
        people = _people((5, 0, 0, 0, "far"), (1, 0, 0, 0, "mid"), (3, 0, 0, 0, "mid2"), (0.5, 0, 0, 0, "near"))
        out = self._call(gen, make_pose(0, 0, 0), people)
        assert out.shape == (2, 6)
        assert (out[:, -1] == 1.0).all()  # both valid
        # nearest ped is at (0.5, 0) in robot frame → rel_x ≈ 0.5
        assert out[0, 0] == pytest.approx(0.5, abs=1e-4)
        assert out[1, 0] == pytest.approx(1.0, abs=1e-4)

    # ------------------------------------------------------------------ robot-frame transform
    def test_robot_facing_plus_x_ped_ahead(self):
        """Robot at origin facing +x, ped at (3,0) world → rel_pos = (3,0)."""
        gen = self._gen(max_peds=1, include_social_state=False)
        out = self._call(gen, make_pose(0, 0, 0), _people((3, 0, 0, 0, "p")))
        assert out[0, 0] == pytest.approx(3.0, abs=1e-4)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-4)

    def test_robot_facing_plus_y_ped_ahead(self):
        """Robot at origin facing +y (yaw=pi/2), ped at (0,3) world → rel_pos = (3,0)."""
        gen = self._gen(max_peds=1, include_social_state=False)
        out = self._call(gen, make_pose(0, 0, np.pi / 2), _people((0, 3, 0, 0, "p")))
        assert out[0, 0] == pytest.approx(3.0, abs=1e-4)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-4)

    def test_robot_offset_position(self):
        """Robot at (1,1) facing +x, ped at world (4,1) → rel_pos = (3,0)."""
        gen = self._gen(max_peds=1, include_social_state=False)
        out = self._call(gen, make_pose(1, 1, 0), _people((4, 1, 0, 0, "p")))
        assert out[0, 0] == pytest.approx(3.0, abs=1e-4)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-4)

    # ------------------------------------------------------------------ determinism
    def test_deterministic_on_repeated_call(self):
        gen = self._gen()
        people = _people((2, 0, 0, 0, "a"), (0, 2, 0.5, 0, "b"), (2, 0, 0, 0.1, "c"))
        pose = make_pose(0, 0, 0)
        out1 = self._call(gen, pose, people)
        out2 = self._call(gen, pose, people)
        assert np.array_equal(out1, out2)

    def test_tie_break_by_stable_id(self):
        """Two peds equidistant — ordering must be stable (CRC32 tie-break, not dict order)."""
        gen = self._gen(max_peds=2)
        # both at distance 2.0 from origin
        people = _people((2, 0, 0, 0, "ped_z"), (0, 2, 0, 0, "ped_a"))
        out1 = self._call(gen, make_pose(0, 0, 0), people)
        out2 = self._call(gen, make_pose(0, 0, 0), people)
        # must agree across calls — content determined by CRC32 order
        assert np.array_equal(out1, out2)

    # ------------------------------------------------------------------ validity flag
    def test_validity_flag_is_last_column(self):
        gen = self._gen(max_peds=3)
        out = self._call(gen, make_pose(0, 0, 0), _people((1, 0, 0, 0, "p")))
        assert out[0, -1] == 1.0   # real ped
        assert out[1, -1] == 0.0   # padding
        assert out[2, -1] == 0.0   # padding

    # ------------------------------------------------------------------ dtype
    def test_output_dtype_float32(self):
        gen = self._gen()
        out = self._call(gen, make_pose(0, 0, 0), _people((1, 0, 0, 0, "p")))
        assert out.dtype == np.float32


# =====================================================================
#  GeneratorManager: per-generator failure counting (P2.3, audit 2026-07-04)
# =====================================================================

class TestGeneratorManagerFailureHandling:
    """A raising generator used to be logged straight to a plain Python
    logger with no counter and no connection to the unified error-collector
    system. It must now be counted per-generator and routed through
    ``collect_error`` (which, since the systemic flush fix, actually reaches
    the log — see ``ObservationSpaceManager.reset_spaces()``).
    """

    def _manager(self, *, name: str = "broken_gen"):
        from rosnav_rl.observations.strategies.generator import GeneratorManager

        node = MagicMock()
        resolver = MagicMock()
        resolver.execution_order = [name]

        manager = GeneratorManager(
            node=node,
            dependency_resolver=resolver,
            simulation_state_container=SimulationStateContainerStub(),
            validate_generators=False,
        )
        return manager

    def test_generate_observations_counts_failures_and_nulls_output(self):
        manager = self._manager(name="broken_gen")

        broken_generator = MagicMock()
        broken_generator.get_observation.side_effect = RuntimeError("boom")

        obs_dict = {}
        manager.generate_observations(
            {"broken_gen": broken_generator}, obs_dict
        )

        assert obs_dict["broken_gen"] is None
        assert manager.generator_failure_counts == {"broken_gen": 1}

        manager.generate_observations(
            {"broken_gen": broken_generator}, obs_dict
        )
        assert manager.generator_failure_counts == {"broken_gen": 2}

    def test_execute_generator_counts_failures_and_nulls_output(self):
        manager = self._manager(name="broken_gen")

        broken_generator = MagicMock()
        broken_generator.get_observation.side_effect = RuntimeError("boom")

        obs_dict = {}
        manager._execute_generator("broken_gen", broken_generator, obs_dict)

        assert obs_dict["broken_gen"] is None
        assert manager.generator_failure_counts == {"broken_gen": 1}
