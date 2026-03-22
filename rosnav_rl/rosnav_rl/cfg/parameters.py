"""Unified agent runtime parameters.

:class:`AgentParameters` is the **single config model** for all scalar
constants consumed by the observation pipeline, reward units, and observation
generators.

There is no separate "simulation state container" — all parameters live here.
Construction-time fields are fed to observation spaces via
:meth:`observation_kwargs`; step-time fields (``robot_radius``,
``safety_distance``, ``goal_radius``, ``max_steps``) are read directly from
this object by reward units and generators.

Usage::

    from rosnav_rl.cfg.parameters import AgentParameters

    params = AgentParameters(laser_num_beams=360, robot_radius=0.215)

    # Construction-time: feed to observation spaces
    obs_kwargs = params.observation_kwargs()   # 18 keys (17 sensor/nav + normalizer)

    # Inference-time: reconstruct from a saved AgentConfig
    params = AgentParameters.from_spec(agent_config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from rosnav_rl.cfg.agent import AgentConfig


class AgentParameters(BaseModel, frozen=True):
    """All fixed runtime parameters for the agent pipeline.

    Fields split into two groups, both stored here so users have **one place**
    to review before training.

    **Observation-space fields** (passed as ``**kwargs`` to every
    ``BaseObservationSpace`` constructor via
    :meth:`~rosnav_rl.spaces.space_manager.base_space_manager.BaseSpaceManager`):

    - Laser: ``laser_num_beams``, ``laser_max_range``
    - Velocity: ``min/max_linear_vel``, ``min/max_translational_vel``, ``min/max_angular_vel``
    - Pedestrian: ``ped_num_types``, ``ped_min/max_speed_x/y``, ``ped_social_state_num``
    - Navigation: ``goal_max_dist``, ``subgoal_max_dist``
    - General: ``normalize``, ``normalizer``

    **Reward / generator fields** (read per-step by reward units and
    observation generators directly from this object):

    - ``robot_radius`` — collision and safety checks
    - ``safety_distance`` — minimum safe clearance from obstacles
    - ``goal_radius`` — goal-reached distance threshold
    - ``max_steps`` — episode length cap

    .. tip::
        Call :meth:`from_spec` to build from a saved
        :class:`~rosnav_rl.cfg.agent.AgentConfig` at inference time::

            params = AgentParameters.from_spec(agent_config)
    """

    # ------------------------------------------------------------------
    # Laser / range-sensor configuration
    # Used by: LaserObservationSpace, StackedLaserMapSpace,
    #          LaserSafeDistanceGenerator
    # ------------------------------------------------------------------

    laser_num_beams: int = Field(
        720,
        description=(
            "Number of laser scan beams. Must match the robot's LIDAR. "
            "Used by: BasicLaserObservationSpace, LaserObservationSpace, "
            "MultiScaleLaserSpace, StackedLaserMapSpace. "
            "Set to robot_description.laser.num_beams at training time."
        ),
    )
    laser_max_range: float = Field(
        30.0,
        description=(
            "Maximum range of the laser sensor (m). "
            "Used by: BasicLaserObservationSpace, LaserObservationSpace, "
            "MultiScaleLaserSpace, StackedLaserMapSpace (normalisation scale). "
            "Set to robot_description.laser.range at training time."
        ),
    )

    # ------------------------------------------------------------------
    # Linear / angular velocity bounds
    # Used by: VelocityObservationSpace
    # ------------------------------------------------------------------

    min_linear_vel: float = Field(
        0.0,
        description=(
            "Minimum linear velocity (m/s). "
            "Used by: VelocityObservationSpace (space low bound). "
            "Set to robot_description.actions.continuous.linear_range[0]."
        ),
    )
    max_linear_vel: float = Field(
        1.0,
        description=(
            "Maximum linear velocity (m/s). "
            "Used by: VelocityObservationSpace (space high bound), "
            "VelocityHistorySpace (normalisation). "
            "Set to robot_description.actions.continuous.linear_range[1]."
        ),
    )
    min_translational_vel: float = Field(
        0.0,
        description=(
            "Minimum translational (y-axis) velocity for holonomic robots (m/s). "
            "Used by: VelocityObservationSpace (holonomic y-vel low bound). "
            "Set to robot_description.actions.continuous.linear_range[0]."
        ),
    )
    max_translational_vel: float = Field(
        1.0,
        description=(
            "Maximum translational (y-axis) velocity for holonomic robots (m/s). "
            "Used by: VelocityObservationSpace (holonomic y-vel high bound). "
            "Set to robot_description.actions.continuous.linear_range[1]."
        ),
    )
    min_angular_vel: float = Field(
        -1.0,
        description=(
            "Minimum angular velocity (rad/s). "
            "Used by: VelocityObservationSpace (space low bound). "
            "Set to robot_description.actions.continuous.angular_range[0]."
        ),
    )
    max_angular_vel: float = Field(
        1.0,
        description=(
            "Maximum angular velocity (rad/s). "
            "Used by: VelocityObservationSpace (space high bound), "
            "VelocityHistorySpace (normalisation). "
            "Set to robot_description.actions.continuous.angular_range[1]."
        ),
    )

    # ------------------------------------------------------------------
    # Pedestrian / social-state configuration
    # Used by: PedestrianObservationSpace
    # ------------------------------------------------------------------

    ped_num_types: int = Field(
        5,
        description=(
            "Number of distinct pedestrian behaviour/social types. "
            "Used by: PedestrianTypeFeaturesSpace (one-hot encoding dimension)."
        ),
    )
    ped_min_speed_x: float = Field(
        0.0,
        description=(
            "Minimum pedestrian speed along x-axis (m/s). "
            "Used by: PedestrianSpeedXSpace (space low bound)."
        ),
    )
    ped_max_speed_x: float = Field(
        2.0,
        description=(
            "Maximum pedestrian speed along x-axis (m/s). "
            "Used by: PedestrianSpeedXSpace (space high bound)."
        ),
    )
    ped_min_speed_y: float = Field(
        0.0,
        description=(
            "Minimum pedestrian speed along y-axis (m/s). "
            "Used by: PedestrianSpeedYSpace (space low bound)."
        ),
    )
    ped_max_speed_y: float = Field(
        2.0,
        description=(
            "Maximum pedestrian speed along y-axis (m/s). "
            "Used by: PedestrianSpeedYSpace (space high bound)."
        ),
    )
    ped_social_state_num: int = Field(
        5,
        description=(
            "Number of social-state categories for pedestrian modelling. "
            "Used by: SocialStateFeaturesSpace."
        ),
    )

    # ------------------------------------------------------------------
    # Navigation / goal-distance bounds
    # Used by: GoalObservationSpace, SubgoalObservationSpace
    # ------------------------------------------------------------------

    goal_max_dist: float = Field(
        10.0,
        description=(
            "Maximum expected distance to the navigation goal (m). "
            "Used by: DistAngleToGoalSpace, RobustGoalSpace, MultiScaleGoalSpace "
            "(normalisation divisor)."
        ),
    )
    subgoal_max_dist: float = Field(
        10.0,
        description=(
            "Maximum expected distance to the current sub-goal (m). "
            "Used by: DistAngleToSubgoalSpace, SubgoalContextSpace "
            "(normalisation divisor)."
        ),
    )

    # ------------------------------------------------------------------
    # General observation flags
    # Used by: all observation spaces
    # ------------------------------------------------------------------

    normalize: bool = Field(
        True,
        description=(
            "Whether to normalise all observation values to [-1, 1] or [0, 1]. "
            "Used by: BaseObservationSpace.__init__ — inherited by every observation space."
        ),
    )
    normalizer: str = Field(
        "max_abs",
        description=(
            "Normalizer algorithm applied when normalize=True. "
            "One of: 'max_abs' (scale to [-1, 1]), 'min_max' (scale to [0, 1]), "
            "'standard' (zero mean, unit variance), 'identity'/'none' (no-op). "
            "Used by: BaseObservationSpace.__init__ — inherited by every observation space."
        ),
    )

    # ------------------------------------------------------------------
    # Robot physical / safety parameters
    # Used by: RewardCollision, RewardSafeDistance, LaserSafeDistanceGenerator
    # ------------------------------------------------------------------

    robot_radius: float = Field(
        0.3,
        json_schema_extra={"group": "reward"},
        description=(
            "Physical bounding radius of the robot (m). "
            "Used by: RewardCollision (collision threshold), "
            "RewardFactoredSafeDistance (safety_distance + robot_radius), "
            "RewardActiveHeadingDirection (VO cone half-angle), "
            "RewardPedTypeCollision (bumper-zone threshold). "
            "Set to robot_description.robot_radius at training time."
        ),
    )
    safety_distance: float = Field(
        1.0,
        json_schema_extra={"group": "reward"},
        description=(
            "Minimum safe clearance from obstacles (m). "
            "Used by: RewardFactoredSafeDistance, "
            "LaserSafetyViolationGenerator (triggers safety flag when min_laser <= safety_distance). "
            "Note: LaserSafetyClearanceSpace and RiskAwareNavigationSpace accept "
            "safety_distance as a direct constructor kwarg and are NOT driven by this field. "
            "Set to arena_config.general.safety_distance at training time."
        ),
    )

    # ------------------------------------------------------------------
    # Task / episode parameters
    # Used by: RewardGoalReached, RewardMaxStepsExceeded
    # ------------------------------------------------------------------

    goal_radius: float = Field(
        0.5,
        json_schema_extra={"group": "reward"},
        description=(
            "Distance threshold for considering the goal reached (m). "
            "Used by: RewardGoalReached (if target_distance < goal_radius). "
            "Set to arena_config.general.goal_radius at training time."
        ),
    )
    max_steps: int = Field(
        600,
        json_schema_extra={"group": "reward"},
        description=(
            "Maximum number of environment steps per episode. "
            "Used by: RewardMaxStepsExceeded (if steps >= max_steps). "
            "Set to arena_config.general.max_num_moves_per_eps at training time."
        ),
    )

    def observation_kwargs(self) -> dict:
        """Return the subset of fields required by observation space constructors.

        Excludes fields tagged ``json_schema_extra={"group": "reward"}`` —
        those are ``robot_radius``, ``safety_distance``, ``goal_radius``, and
        ``max_steps``, which reward units and generators read directly from this
        object and are not consumed by observation spaces.

        New observation-pipeline fields are included automatically; only reward
        fields need the explicit ``json_schema_extra={"group": "reward"}`` tag.

        Returns:
            dict: 18 keyword arguments suitable for ``BaseObservationSpace``
                  constructors (17 sensor/nav fields + ``normalizer``).

        Example::

            obs_kwargs = spec.parameters.observation_kwargs()
            ObservationSpaceManager(..., config={"MySpace": obs_kwargs})
        """
        return {
            name: getattr(self, name)
            for name, field_info in self.__class__.model_fields.items()
            if (field_info.json_schema_extra or {}).get("group") != "reward"
        }

    @classmethod
    def from_spec(cls, spec: "AgentConfig") -> "AgentParameters":
        """Return the parameters from a saved :class:`~rosnav_rl.cfg.agent.AgentConfig`.

        Convenience method for inference-time reconstruction::

            params = AgentParameters.from_spec(agent_config)
            # equivalent to: params = agent_config.parameters
        """
        return spec.parameters
