"""Pedestrian-aware safety, collision, and social-proximity reward units."""

import warnings
from typing import Any, Dict

import numpy as np

from rosnav_rl.utils.observation_types import (
    PedestrianRelativeLocations,
    PedestrianRelativeVelocities,
    PedestrianTypeMinDistances,
    RobotActionVector,
)
from rosnav_rl.cfg.parameters import AgentParameters

from ..constants import DEFAULTS
from ..reward_function import RewardFunction
from ..utils import check_params
from .base_reward_units import RewardUnit
from .reward_unit_factory import RewardUnitFactory


@RewardUnitFactory.register("ped_type_safety_distance")
class RewardPedTypeSafetyDistance(RewardUnit):
    """Reward unit for pedestrian type-specific safety distance violation detection.

    Applies negative rewards when the robot violates minimum safety distance to specific
    pedestrian types or groups. Supports flexible type-reward mapping for different
    pedestrian behaviors and prioritized safety enforcement.

    Technical Specifications:
    - Type-Specific Safety: Different safety thresholds per pedestrian type/group
    - Distance Monitoring: Continuous distance tracking to pedestrian groups
    - Flexible Configuration: Dictionary-based type-reward mapping or single type mode

    Configuration:
    - type_reward_pairs: Dictionary mapping pedestrian types to reward values
    - ped_type: Single pedestrian type to monitor (fallback if no pairs provided)
    - reward: Default reward value for single type mode
    - safety_distance: Distance threshold for safety violation detection
    - _on_safe_dist_violation: Enable/disable during general safety violations

    Output Behavior: Applies type-specific reward when distance < safety_distance

    Applications: Social navigation, type-aware safety, and pedestrian behavior adaptation.
    """

    requires = {
        "pedestrian_distances": PedestrianTypeMinDistances,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        type_reward_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE,
        reward: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.REWARD,
        safety_distance: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize pedestrian type-specific safety distance reward unit.

        Args:
            reward_function: The reward function object managing this unit
            type_reward_pairs: Dictionary mapping pedestrian types to reward values
            ped_type: Single pedestrian type to monitor (fallback mode)
            reward: Default reward value for single type mode
            safety_distance: Distance threshold for safety violation detection
            _on_safe_dist_violation: Enable/disable during general safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._reward = reward
        self._safety_distance = safety_distance
        self._type_reward_pairs = (
            type_reward_pairs
            if isinstance(type_reward_pairs, dict)
            else {ped_type: reward}
        )

    def __call__(
        self,
        pedestrian_distances: PedestrianTypeMinDistances,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply type-specific safety distance violation penalties.

        Monitors minimum distances to pedestrian groups and applies configured
        penalties when safety thresholds are violated.

        Args:
            pedestrian_distances (PedestrianTypeMinDistances): Min distances to pedestrian groups
                - Format: Dict[Union[str, int], float]
                - Units: meters
                - Source: pedestrian distance tracking system
                - Constraints: Keys are group IDs, values are minimum distances ≥ 0
                - Example: {1: 2.5, 2: 4.1, 3: 1.8} (type-specific distance monitoring)
        """
        if not pedestrian_distances:
            self._report_warning(f"No pedestrian type distances found.")
            return

        for ped_type, reward in self._type_reward_pairs.items():
            if ped_type not in pedestrian_distances:
                self._report_warning(f"Pedestrian type {ped_type} not found.")
                continue

            if pedestrian_distances[ped_type] < self._safety_distance:
                self.add_reward(reward)

    def reset(self):
        """Reset internal state for new episode."""
        pass

@RewardUnitFactory.register("ped_type_factored_safety_distance")
class RewardPedTypeFactoredSafetyDistance(RewardUnit):
    """Reward unit for pedestrian type-specific factored safety distance violations.

    Applies proportional negative rewards based on safety distance violations to specific
    pedestrian types. The penalty magnitude scales with the severity of the violation,
    providing fine-grained feedback for social navigation training.

    Technical Specifications:
    - Factored Distance Penalty: Reward = factor * (safety_distance - actual_distance)
    - Type-Specific Factors: Different scaling factors per pedestrian type/group
    - Proportional Feedback: Penalty magnitude reflects violation severity

    Configuration:
    - type_factor_pairs: Dictionary mapping pedestrian types to scaling factors
    - ped_type: Single pedestrian type to monitor (fallback if no pairs provided)
    - factor: Default scaling factor for single type mode
    - safety_distance: Distance threshold for safety violation detection
    - _on_safe_dist_violation: Enable/disable during general safety violations

    Output Behavior: reward = factor * (safety_distance - distance) when distance < safety_distance

    Applications: Smooth social navigation, fine-grained safety training, and type-aware penalty scaling.
    """

    requires = {
        "pedestrian_distances": PedestrianTypeMinDistances,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        type_factor_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.TYPE,
        factor: float = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.FACTOR,
        safety_distance: float = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize pedestrian type-specific factored safety distance reward unit.

        Args:
            reward_function: The reward function object managing this unit
            type_factor_pairs: Dictionary mapping pedestrian types to scaling factors
            ped_type: Single pedestrian type to monitor (fallback mode)
            factor: Default scaling factor for single type mode
            safety_distance: Distance threshold for safety violation detection
            _on_safe_dist_violation: Enable/disable during general safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._factor = factor
        self._safety_distance = safety_distance
        self._type_factor_pairs = (
            type_factor_pairs
            if isinstance(type_factor_pairs, dict)
            else {ped_type: factor}
        )

    def __call__(
        self,
        pedestrian_distances: PedestrianTypeMinDistances,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply type-specific factored safety distance violation penalties.

        Calculates proportional penalties based on safety distance violations,
        with penalty magnitude reflecting violation severity and type-specific scaling.

        Args:
            pedestrian_distances (PedestrianTypeMinDistances): Min distances to pedestrian groups
                - Format: Dict[Union[str, int], float]
                - Units: meters
                - Source: pedestrian distance tracking system
                - Constraints: Keys are group IDs, values are minimum distances ≥ 0
                - Example: {1: 2.5, 2: 4.1, 3: 1.8} (factored scaling per type)
        """
        if not pedestrian_distances:
            self._report_warning(
                "Won't apply reward unit. No pedestrian type distances found."
            )
            return

        for ped_type, factor in self._type_factor_pairs.items():
            if ped_type not in pedestrian_distances:
                self._report_warning(f"Pedestrian type {ped_type} not found.")
                continue

            # Apply proportional penalty based on safety distance violation
            if pedestrian_distances[ped_type] < self._safety_distance:
                violation_magnitude = (
                    self._safety_distance - pedestrian_distances[ped_type]
                )
                factored_penalty = factor * violation_magnitude
                self.add_reward(factored_penalty)

    def reset(self):
        """Reset internal state for new episode."""
        pass

@RewardUnitFactory.register("ped_type_collision")
class RewardPedTypeCollision(RewardUnit):
    """Reward unit for pedestrian type-specific collision detection and penalty application.

    Detects robot collisions with specific pedestrian types using proximity-based collision
    detection with configurable bumper zones. Applies significant negative rewards for
    type-specific collisions to promote safe social navigation behaviors.

    Technical Specifications:
    - Collision Detection: Distance-based collision with bumper zone consideration
    - Type-Specific Penalties: Different rewards per pedestrian type/group
    - Robot Radius Integration: Accounts for robot physical dimensions in collision detection

    Configuration:
    - type_reward_pairs: Dictionary mapping pedestrian types to collision penalties
    - ped_type: Single pedestrian type to monitor (fallback if no pairs provided)
    - reward: Default collision penalty for single type mode
    - bumper_zone: Additional collision buffer beyond robot radius

    Output Behavior: Applies type-specific penalty when distance ≤ (bumper_zone + robot_radius)

    Applications: Social safety enforcement, type-aware collision avoidance, and penalty differentiation.
    """

    requires = {
        "pedestrian_distances": PedestrianTypeMinDistances,
        "simulation_state_container": AgentParameters,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        type_reward_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.TYPE,
        reward: float = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.REWARD,
        bumper_zone: float = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.BUMPER_ZONE,
        *args,
        **kwargs,
    ) -> None:
        """Initialize pedestrian type-specific collision detection reward unit.

        Args:
            reward_function: The reward function object managing this unit
            type_reward_pairs: Dictionary mapping pedestrian types to collision penalties
            ped_type: Single pedestrian type to monitor (fallback mode)
            reward: Default collision penalty for single type mode
            bumper_zone: Additional collision buffer beyond robot radius
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._type_reward_pairs = (
            type_reward_pairs
            if isinstance(type_reward_pairs, dict)
            else {ped_type: reward}
        )
        self._bumper_zone = bumper_zone

    def __call__(
        self,
        pedestrian_distances: PedestrianTypeMinDistances,
        simulation_state_container: AgentParameters,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Detect and penalize type-specific pedestrian collisions.

        Monitors minimum distances to pedestrian groups and applies collision penalties
        when robots collide with specific pedestrian types within the bumper zone.

        Args:
            pedestrian_distances (PedestrianTypeMinDistances): Min distances to pedestrian groups
                - Format: Dict[Union[str, int], float]
                - Units: meters
                - Source: pedestrian distance tracking system
                - Constraints: Keys are group IDs, values are minimum distances ≥ 0
                - Example: {1: 2.5, 2: 4.1, 3: 1.8} (collision detection per type)
            simulation_state_container (AgentParameters): Robot and environment state
                - Contains: robot configuration, dimensions, and simulation parameters
                - Used for: robot radius in collision detection calculations
        """
        if not pedestrian_distances:
            self._report_warning(
                "Won't apply reward unit. No pedestrian type distances found."
            )
            return

        # Calculate collision threshold including robot dimensions
        collision_threshold = (
            self._bumper_zone + simulation_state_container.robot_radius
        )

        for ped_type, reward in self._type_reward_pairs.items():
            if ped_type not in pedestrian_distances:
                self._report_warning(f"Pedestrian type {ped_type} not found.")
                continue

            # Apply collision penalty if pedestrian within collision zone
            if pedestrian_distances[ped_type] <= collision_threshold:
                self.add_reward(reward)

    def reset(self):
        """Reset internal state for new episode."""
        pass

@RewardUnitFactory.register("ped_type_vel_constraint")
class RewardPedTypeVelocityConstraint(RewardUnit):
    """Reward unit for pedestrian type-specific velocity constraints and speed regulation.

    Monitors robot linear velocity in proximity to specific pedestrian types, applying
    velocity-proportional penalties to encourage speed reduction in social spaces.
    Promotes socially-aware navigation by constraining robot speed near pedestrians.

    Technical Specifications:
    - Velocity-Proportional Penalties: Reward scales with robot forward velocity
    - Type-Specific Activation: Only applies constraints near specified pedestrian types
    - Distance-Based Activation: Constraint activates within specified proximity distance

    Configuration:
    - ped_type: Specific pedestrian type to monitor for velocity constraints
    - penalty_factor: Velocity penalty scaling factor (negative for penalties)
    - active_distance: Distance threshold for constraint activation

    Output Behavior: Applies penalty = -penalty_factor × linear_velocity when pedestrian within active_distance

    Applications: Social speed regulation, pedestrian comfort zones, and velocity-aware navigation.
    """

    requires = {
        "pedestrian_distances": PedestrianTypeMinDistances,
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE,
        penalty_factor: float = 0.05,
        active_distance: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize pedestrian type-specific velocity constraint reward unit.

        Args:
            reward_function: The reward function object managing this unit
            ped_type: Specific pedestrian type to monitor for velocity constraints
            penalty_factor: Velocity penalty scaling factor (negative for penalties)
            active_distance: Distance threshold for constraint activation
            _on_safe_dist_violation: Whether to apply penalty on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._penalty_factor = penalty_factor
        self._active_distance = active_distance

    def __call__(
        self,
        pedestrian_distances: PedestrianTypeMinDistances,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply velocity constraints when robot is near specific pedestrian types.

        Monitors robot linear velocity and applies proportional penalties when traveling
        too fast in proximity to specific pedestrian types within the active distance.

        Args:
            pedestrian_distances (PedestrianTypeMinDistances): Min distances to pedestrian groups
                - Format: Dict[Union[str, int], float]
                - Units: meters
                - Source: pedestrian distance tracking system
                - Constraints: Keys are group IDs, values are minimum distances ≥ 0
                - Example: {1: 2.5, 2: 4.1, 3: 1.8} (velocity constraints per type)
            last_action (ActionState): Robot's most recent action command
                - Format: Tuple[float, float, float] or similar action representation
                - Units: [m/s, m/s, rad/s] for linear_x, linear_y, angular_z
                - Source: action execution system
                - Constraints: action[0] represents forward linear velocity
                - Example: (0.5, 0.0, 0.2) (forward velocity used for penalty scaling)
        """
        if not pedestrian_distances:
            self._report_warning(
                "Won't apply reward unit. No pedestrian type distances found."
            )
            return

        if last_action is None:
            self._report_warning("Won't apply reward unit. No last action found.")
            return

        if self._type not in pedestrian_distances:
            self._report_warning(f"Pedestrian type {self._type} not found.")
            return

        # Apply velocity penalty when pedestrian within active distance
        if pedestrian_distances[self._type] < self._active_distance:
            self.add_reward(-self._penalty_factor * last_action[0])

    def reset(self):
        """Reset internal state for new episode."""
        pass

@RewardUnitFactory.register("proxemic_intrusion")
class RewardProxemicIntrusion(RewardUnit):
    """Reward unit for asymmetric, heading-aware pedestrian personal-space intrusion.

    Penalizes the robot for entering a pedestrian's Gaussian comfort zone, with the zone
    stretched further ahead of the pedestrian's direction of travel than behind it (Kirby/SARL
    proxemics). This gives the policy a smooth gradient to route around a pedestrian's future
    path rather than just its current position, complementing the isotropic safety-distance
    units above.

    Technical Specifications:
    - Heading-Frame Rotation: Robot position relative to each pedestrian is rotated into that
      pedestrian's own heading frame (derived from its relative velocity).
    - Asymmetric Gaussian: Separate along-heading sigma ahead (sigma_front) vs. behind
      (sigma_back) the pedestrian; lateral spread uses sigma_side.
    - Stationary Fallback: Pedestrians with negligible speed use sigma_side for the along-axis
      too, since their heading is undefined.

    Configuration:
    - weight: Penalty scale applied to the summed per-pedestrian intrusion.
    - sigma_front / sigma_back / sigma_side: Comfort-zone spreads (meters) ahead, behind, and
      to the side of a pedestrian's heading.
    - activation_radius: Pedestrians farther than this (meters) are skipped entirely.
    - min_ped_speed: Speed (m/s) below which a pedestrian is treated as stationary.

    Output Behavior: reward = -weight * sum_i exp(-0.5 * [(d_along_i / sigma_i)^2 +
    (d_perp_i / sigma_side)^2]), sigma_i = sigma_front if d_along_i > 0 else sigma_back.

    Applications: Social navigation, proxemic comfort, path-anticipatory pedestrian avoidance.
    """

    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
        "pedestrian_relative_velocities": PedestrianRelativeVelocities,
        "simulation_state_container": AgentParameters,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        weight: float = 0.1,
        sigma_front: float = 1.2,
        sigma_back: float = 0.4,
        sigma_side: float = 0.6,
        activation_radius: float = 3.0,
        min_ped_speed: float = 0.15,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the proxemic intrusion reward unit.

        Args:
            reward_function: The reward function object holding this unit
            weight: Penalty scale applied to the summed per-pedestrian intrusion
            sigma_front: Comfort-zone spread (meters) ahead of a pedestrian's heading
            sigma_back: Comfort-zone spread (meters) behind a pedestrian's heading
            sigma_side: Comfort-zone spread (meters) lateral to a pedestrian's heading
            activation_radius: Pedestrians farther than this (meters) are skipped
            min_ped_speed: Speed (m/s) below which a pedestrian is treated as stationary
            _on_safe_dist_violation: Whether to apply penalty on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._weight = weight
        self._sigma_front = sigma_front
        self._sigma_back = sigma_back
        self._sigma_side = sigma_side
        self._activation_radius = activation_radius
        self._min_ped_speed = min_ped_speed

    def __call__(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        pedestrian_relative_velocities: PedestrianRelativeVelocities,
        simulation_state_container: AgentParameters,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Penalize the robot for intruding into pedestrians' asymmetric comfort zones.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Pedestrian positions
                in the robot frame
                - Format: N×2 array of [x, y] positions in meters
                - Source: pedestrian tracking system
                - Example: [[2.0, 1.5], [-1.0, 0.5]]
            pedestrian_relative_velocities (PedestrianRelativeVelocities): Pedestrian velocities
                in the robot frame
                - Format: N×2 array of [vx, vy] velocities in meters/second
                - Source: pedestrian tracking system
                - Constraints: N×2 array matching pedestrian_relative_locations
                - Example: [[0.5, 0.2], [-0.3, 0.8]]
            simulation_state_container (AgentParameters): Robot and environment state
                - Unused here; part of the common schema surface for future robot-frame needs
        """
        if (
            pedestrian_relative_locations is None
            or pedestrian_relative_velocities is None
            or len(pedestrian_relative_locations) == 0
        ):
            return

        intrusion = 0.0
        for ped_location, ped_velocity in zip(
            pedestrian_relative_locations, pedestrian_relative_velocities
        ):
            px, py = ped_location[0], ped_location[1]
            dist = np.hypot(px, py)
            if dist > self._activation_radius:
                continue

            speed = np.hypot(ped_velocity[0], ped_velocity[1])
            if speed >= self._min_ped_speed:
                heading = np.arctan2(ped_velocity[1], ped_velocity[0])
            else:
                heading = None

            # Vector from pedestrian to robot, in robot-frame coordinates.
            rx, ry = -px, -py

            if heading is not None:
                cos_h, sin_h = np.cos(heading), np.sin(heading)
                d_along = rx * cos_h + ry * sin_h
                d_perp = -rx * sin_h + ry * cos_h
                sigma_along = self._sigma_front if d_along > 0 else self._sigma_back
            else:
                d_along, d_perp = rx, ry
                sigma_along = self._sigma_side

            intrusion += np.exp(
                -0.5
                * (
                    (d_along / sigma_along) ** 2
                    + (d_perp / self._sigma_side) ** 2
                )
            )

        if intrusion > 0.0:
            self.add_reward(-self._weight * intrusion)

    def reset(self):
        """Reset internal state for new episode."""
        pass

@RewardUnitFactory.register("social_potential")
class RewardSocialPotential(RewardUnit):
    """Reward unit for potential-based shaping toward crowd separation.

    Provides a dense, policy-invariant gradient (PBRS, Ng et al. 1999) that rewards
    increasing separation from the nearest pedestrian, complementing the cumulative
    comfort-zone cost of `proxemic_intrusion` with a telescoping gradient that helps
    the policy reach the low-cost region faster in imagination.

    Technical Specifications:
    - Potential: Phi(s) = min(nearest_ped_distance, clip_distance). Sign convention
      is the OPPOSITE of `approach_goal` (Phi = -distance_to_goal there): here Phi
      increases with pedestrian *separation*, since the objective is to move away
      from crowds rather than toward a goal. Do not "fix" this sign later.
    - Shaping: F = factor * (gamma * Phi(s') - Phi(s)). Per Ng et al. 1999 this
      leaves the optimal policy invariant for any factor/gamma — note this holds
      for the *optimal* policy, not per-step: with gamma < 1 a constant potential
      still yields a fixed per-step drift factor*(gamma-1)*Phi rather than exactly
      zero (same property already present in `approach_goal`'s PBRS branch).
    - Discontinuity Guard: skips shaping when the potential jumps by more than
      jump_threshold in one step (nearest-pedestrian identity switch, or a
      pedestrian entering/leaving clip_distance), avoiding spurious spikes.
    - Degenerate Cases: no pedestrians, or nearest pedestrian beyond clip_distance,
      both clip Phi(s) to clip_distance — identical, distance-independent drift in
      either case, so a far/absent pedestrian never adds a live gradient signal.

    Configuration:
    - factor: shaping scale.
    - gamma: discount used in the potential difference; must match the RL
      algorithm's discount factor for the PBRS policy-invariance guarantee.
    - clip_distance: potential saturates beyond this distance (meters).
    - jump_threshold: skip shaping if |Phi(s') - Phi(s)| exceeds this (meters).

    Applications: Dense crowd-separation gradient for imagination-horizon training,
    complementing `proxemic_intrusion`'s cumulative comfort-zone cost.
    """

    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        factor: float = DEFAULTS.SOCIAL_POTENTIAL.FACTOR,
        gamma: float = DEFAULTS.SOCIAL_POTENTIAL.GAMMA,
        clip_distance: float = DEFAULTS.SOCIAL_POTENTIAL.CLIP_DISTANCE,
        jump_threshold: float = DEFAULTS.SOCIAL_POTENTIAL.JUMP_THRESHOLD,
        _on_safe_dist_violation: bool = DEFAULTS.SOCIAL_POTENTIAL._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the social potential reward unit.

        Args:
            reward_function: The reward function object holding this unit
            factor: shaping scale applied to the potential difference
            gamma: discount used in the potential difference; must match the RL
                algorithm's discount factor for the PBRS policy-invariance guarantee
            clip_distance: potential saturates beyond this distance (meters)
            jump_threshold: skip shaping this step if the potential jumps more than
                this (meters), e.g. on a nearest-pedestrian identity switch
            _on_safe_dist_violation: Whether to apply shaping on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._factor = factor
        self._gamma = gamma
        self._clip_distance = clip_distance
        self._jump_threshold = jump_threshold

        self.last_phi = None

    def __call__(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Shape reward toward increasing separation from the nearest pedestrian.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Pedestrian
                positions in the robot frame
                - Format: N×2 array of [x, y] positions in meters
                - Source: pedestrian tracking system
                - Example: [[2.0, 1.5], [-1.0, 0.5]]
        """
        if pedestrian_relative_locations is None or len(pedestrian_relative_locations) == 0:
            nearest_dist = self._clip_distance
        else:
            nearest_dist = np.min(
                np.hypot(
                    pedestrian_relative_locations[:, 0],
                    pedestrian_relative_locations[:, 1],
                )
            )

        current_phi = min(float(nearest_dist), self._clip_distance)

        if self.last_phi is not None:
            phi_jumped = abs(current_phi - self.last_phi) > self._jump_threshold
            if not phi_jumped:
                shaped = self._gamma * current_phi - self.last_phi
                self.add_reward(self._factor * shaped)

        self.last_phi = current_phi

    def reset(self):
        """Reset internal state for new episode."""
        self.last_phi = None


@RewardUnitFactory.register("tgrf_discomfort")
class RewardTGRFDiscomfort(RewardUnit):
    """Reward unit for TGRF-style (Transformable Gaussian Reward Function) discomfort.

    Peak-normalized Gaussian penalty on the nearest pedestrian's distance, following
    Kim et al. 2024 (Sensors 24(14):4540, arXiv:2402.14569). Unlike `proxemic_intrusion`'s
    anisotropic, heading-aware penalty summed over all pedestrians, TGRF is isotropic and
    keyed only to the single nearest pedestrian, active only inside a fixed danger zone.
    This is the literature-established composite term cited in the paper; kept separate
    from and simpler than `proxemic_intrusion`/`social_potential` (both remain available
    in the codebase but are dropped from the active composite to avoid double-counting
    discomfort).

    Technical Specifications:
    - Peak-Normalized Gaussian: TGRF(w, mu, sigma; x) = w * N(x; mu, sigma) / max(N) =
      w * exp(-0.5 * ((x - mu) / sigma)^2) with mu=0 — peak-normalization cancels the
      1/(sigma*sqrt(2*pi)) constant, so `weight` alone sets the maximum magnitude.
    - Danger Zone Gating: term is exactly zero once nearest-pedestrian distance
      d_min >= danger_zone_m (paper: d_disc).
    - Nearest-Pedestrian Selection: uses the minimum Euclidean distance across all
      tracked pedestrians; no heading/velocity dependence (isotropic, unlike
      `proxemic_intrusion`).

    Configuration:
        weight: Penalty scale at d_min=0 (paper: w_disc=0.25).
        sigma: Gaussian spread in meters (paper: sigma_disc=0.2).
        danger_zone_m: Distance in meters beyond which the term is exactly zero
            (paper: d_disc=0.5).

    Output Behavior: reward = -weight * exp(-0.5 * (d_min / sigma)^2) if
    d_min < danger_zone_m else 0.0.

    Applications: Literature-established discomfort composite for Social-Dreamer /
    ICRA2027 experiments (citable, TGRF-style; Kim et al. 2024).
    """

    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        weight: float = DEFAULTS.TGRF_DISCOMFORT.WEIGHT,
        sigma: float = DEFAULTS.TGRF_DISCOMFORT.SIGMA,
        danger_zone_m: float = DEFAULTS.TGRF_DISCOMFORT.DANGER_ZONE_M,
        _on_safe_dist_violation: bool = DEFAULTS.TGRF_DISCOMFORT._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the TGRF discomfort reward unit.

        Args:
            reward_function: The reward function object holding this unit
            weight: Penalty scale at zero distance (paper: w_disc)
            sigma: Gaussian spread in meters (paper: sigma_disc)
            danger_zone_m: Distance beyond which the term is exactly zero (paper: d_disc)
            _on_safe_dist_violation: Whether to apply penalty on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._weight = weight
        self._sigma = sigma
        self._danger_zone_m = danger_zone_m

    def check_parameters(self, *args: Any, **kwargs: Any) -> None:
        """Warn on non-positive scale parameters."""
        if self._weight <= 0:
            warn_msg = (
                f"'tgrf_discomfort' weight ({self._weight}) should be positive; "
                "it is negated internally to produce a penalty."
            )
            self._report_warning(warn_msg)
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
        if self._sigma <= 0:
            warn_msg = f"'tgrf_discomfort' sigma ({self._sigma}) must be positive."
            self._report_warning(warn_msg)
            warnings.warn(warn_msg, UserWarning, stacklevel=2)

    def __call__(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Penalize the robot for the nearest pedestrian entering the danger zone.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Pedestrian
                positions in the robot frame
                - Format: N×2 array of [x, y] positions in meters
                - Source: pedestrian tracking system
                - Example: [[2.0, 1.5], [-1.0, 0.5]]
        """
        if pedestrian_relative_locations is None or len(pedestrian_relative_locations) == 0:
            return

        d_min = np.min(
            np.hypot(
                pedestrian_relative_locations[:, 0],
                pedestrian_relative_locations[:, 1],
            )
        )

        if d_min >= self._danger_zone_m:
            return

        self.add_reward(-self._weight * np.exp(-0.5 * (d_min / self._sigma) ** 2))

    def reset(self) -> None:
        """Reset internal state for new episode."""
        pass
