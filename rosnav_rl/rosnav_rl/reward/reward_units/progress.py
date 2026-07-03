"""Episode-progress reward units: distance travelled and step-limit termination."""

from typing import Any

from rosnav_rl.observations.utils.types import RobotActionVector
from rosnav_rl.cfg.parameters import AgentParameters

from ..constants import DEFAULTS, DONE_REASONS
from ..reward_function import RewardFunction
from ..utils import check_params
from .base_reward_units import RewardUnit
from .reward_unit_factory import RewardUnitFactory


@RewardUnitFactory.register("distance_travelled")
class RewardDistanceTravelled(RewardUnit):
    """Reward unit for distance traveled calculation with velocity-based energy consumption.

    Provides reward proportional to the robot's movement, encouraging efficient navigation
    while penalizing excessive energy consumption. Supports separate scaling for linear
    and angular velocities to balance speed and rotational behavior.

    Technical Specifications:
    - Energy Consumption Model: Velocity-scaled negative reward
    - Velocity Separation: Independent linear and angular scaling factors
    - Consumption Factor: Overall energy efficiency scaling

    Configuration:
    - consumption_factor: Overall energy consumption penalty scaling
    - lin_vel_scalar: Linear velocity importance weight
    - ang_vel_scalar: Angular velocity importance weight

    Output Behavior: reward = -factor * (linear_scaled + angular_scaled)

    Applications: Energy-efficient navigation, velocity regulation, and movement encouragement.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        consumption_factor: float = DEFAULTS.DISTANCE_TRAVELLED.CONSUMPTION_FACTOR,
        lin_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.LIN_VEL_SCALAR,
        ang_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.ANG_VEL_SCALAR,
        _on_safe_dist_violation: bool = DEFAULTS.DISTANCE_TRAVELLED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Initialize distance traveled reward unit with velocity scaling parameters.

        Args:
            reward_function: The reward function object managing this unit
            consumption_factor: Overall energy consumption penalty factor (default: 0.01)
            lin_vel_scalar: Linear velocity scaling weight (default: 1.0)
            ang_vel_scalar: Angular velocity scaling weight (default: 0.1)
            _on_safe_dist_violation: Enable reward during safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._factor = consumption_factor
        self._lin_vel_scalar = lin_vel_scalar
        self._ang_vel_scalar = ang_vel_scalar

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculate reward based on robot movement with energy consumption modeling.

        Computes energy-based reward proportional to robot velocity, encouraging efficient
        movement while penalizing excessive energy consumption through configurable scaling.

        Args:
            last_action (RobotActionVector): Robot velocity command in base frame
                - Shape: (3,)
                - Units: [meters/second, meters/second, radians/second]
                - Source: robot controller or action
                - Constraints: linear.x, linear.y, angular.z
                - Example: [0.5, 0.0, 0.2] (forward motion with rotation)
        """
        # Extract velocity components
        linear_velocity = last_action[0]  # Forward/backward velocity
        angular_velocity = last_action[-1]  # Rotational velocity

        # Calculate scaled energy consumption (use abs to avoid rewarding reverse/CW spin)
        linear_energy = abs(linear_velocity) * self._lin_vel_scalar
        angular_energy = abs(angular_velocity) * self._ang_vel_scalar

        # Apply negative consumption factor (encouraging efficient movement)
        total_energy_reward = -(linear_energy + angular_energy) * self._factor

        self.add_reward(total_energy_reward)

@RewardUnitFactory.register("max_steps_exceeded")
class RewardMaxStepsExceeded(RewardUnit):
    """
    A reward unit that penalizes the agent when the maximum number of steps is exceeded.

    Args:
        reward_function (RewardFunction): The reward function to which this unit belongs.
        penalty (float, optional): The penalty value to be applied when the maximum steps are exceeded. Defaults to 10.
        _on_safe_dist_violation (bool, optional): Whether to apply the penalty on safe distance violation. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _penalty (float): The penalty value to be applied when the maximum steps are exceeded.
        _steps (int): The current number of steps taken.

    Methods:
        __call__(*args, **kwargs): Updates the step count and applies the penalty if the maximum steps are exceeded.
        reset(): Resets the step count to zero.
    """

    requires = {
        "simulation_state_container": AgentParameters,
    }

    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.STEP_LIMIT,
        "is_success": 0,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        penalty: float = 10,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._penalty = penalty
        self._steps = 0

    def check_parameters(self, *args, **kwargs):
        if self._penalty < 0.0:
            warn_msg = (
                f"Reconsider this reward. "
                f"The penalty should be a positive value as it is going to be subtracted from the total reward."
                f"Current value: {self._penalty}"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        simulation_state_container: AgentParameters,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates the step count and applies the penalty if the maximum steps are exceeded.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._steps += 1
        if self._steps >= simulation_state_container.max_steps:
            self.add_reward(-self._penalty)
            self.add_info(self.DONE_INFO)

    def reset(self):
        """
        Resets the step count to zero.
        """
        self._steps = 0
