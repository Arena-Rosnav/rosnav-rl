"""Arm goal generators on real poses."""

import numpy as np

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.data_sources.generators import (
    GraspDistanceAngleGenerator
)

PARAMS = AgentParameters()


def quat_axis(axis: str, angle: float) -> list[float]:
    half = angle / 2.0
    s, c = np.sin(half), np.cos(half)
    return {
        "x": [s, 0.0, 0.0, c],
        "y": [0.0, s, 0.0, c],
        "z": [0.0, 0.0, s, c],
    }[axis]


def pose(xyz, quat=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    return np.array([*xyz, *quat], dtype=np.float32)


def metrics(gen, grasp_point, target_pose) -> np.ndarray:
    return gen.get_observation(
        {"grasp_point": grasp_point, "target_pose": target_pose},
        simulation_state_container=PARAMS,
    )


def test_distance_and_approach_angle_come_from_grasp_point_and_target():
    gen = GraspDistanceAngleGenerator(name="grasp_dist_angle")

    # tool z tilted 0.5 rad about x away from +z; target 0.5 m along +z
    grasp_point = pose((0.0, 0.0, 0.0), quat_axis("x", 0.5))
    target_pose = pose((0.0, 0.0, 0.5))

    result = metrics(gen, grasp_point, target_pose)

    assert result.shape == (2,)
    np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-6)


def test_target_orientation_is_ignored():
    gen = GraspDistanceAngleGenerator(name="grasp_dist_angle")

    grasp_point = pose((0.0, 0.0, 0.0), quat_axis("x", 0.5))
    upright = pose((0.0, 0.0, 0.5))
    flipped = pose((0.0, 0.0, 0.5), quat_axis("y", np.pi))

    np.testing.assert_allclose(
        metrics(gen, grasp_point, flipped),
        metrics(gen, grasp_point, upright),
        atol=1e-6,
    )


def test_approach_direction_points_from_grasp_point_to_target():
    gen = GraspDistanceAngleGenerator(name="grasp_dist_angle")

    # tool z is +x and the target lies at +x, so the tool is aligned (0, not pi)
    grasp_point = pose((0.0, 0.0, 0.0), quat_axis("y", np.pi / 2))
    target_pose = pose((0.5, 0.0, 0.0))

    np.testing.assert_allclose(metrics(gen, grasp_point, target_pose), [0.5, 0.0], atol=1e-6)


def test_angle_reuses_the_last_direction_inside_the_epsilon():
    gen = GraspDistanceAngleGenerator(name="grasp_dist_angle", direction_epsilon=0.05)

    far = metrics(
        gen,
        pose((0.0, 0.0, 0.0), quat_axis("y", np.pi / 2)),
        pose((1.0, 0.0, 0.0)),
    )
    np.testing.assert_allclose(far, [1.0, 0.0], atol=1e-6)

    near = metrics(
        gen,
        pose((0.0, 0.0, 0.0)),
        pose((0.01, 0.0, 0.0)),
    )
    np.testing.assert_allclose(near, [0.01, np.pi / 2], atol=1e-6)


def test_angle_is_zero_before_any_direction_was_seen():
    gen = GraspDistanceAngleGenerator(name="grasp_dist_angle", direction_epsilon=0.05)

    result = metrics(
        gen,
        pose((0.0, 0.0, 0.0)),
        pose((0.01, 0.0, 0.0)),
    )

    np.testing.assert_allclose(result, [0.01, 0.0], atol=1e-6)
