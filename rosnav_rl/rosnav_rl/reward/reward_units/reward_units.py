"""Backward-compatible re-export shim — the reward units now live in per-category modules."""
from .goal import RewardGoalReached, RewardApproachGoal
from .collision_safety import RewardSafeDistance, RewardFactoredSafeDistance, RewardCollision
from .pedestrian import (
    RewardPedTypeSafetyDistance,
    RewardPedTypeFactoredSafetyDistance,
    RewardPedTypeCollision,
    RewardPedTypeVelocityConstraint,
    RewardProxemicIntrusion,
    RewardSocialPotential,
    RewardTGRFDiscomfort,
)
from .velocity import (
    RewardNoMovement,
    RewardReverseDrive,
    RewardFactoredReverseDrive,
    RewardAbruptVelocityChange,
    RewardRootVelocityDifference,
    RewardTwoFactorVelocityDifference,
    RewardActiveHeadingDirection,
    RewardAngularVelocityConstraint,
    RewardLinearVelBoost,
)
from .progress import RewardDistanceTravelled, RewardMaxStepsExceeded

# UPDATE WHEN ADDING A NEW UNIT
__all__ = [
    "RewardGoalReached",
    "RewardSafeDistance",
    "RewardNoMovement",
    "RewardApproachGoal",
    "RewardCollision",
    "RewardDistanceTravelled",
    "RewardReverseDrive",
    "RewardAbruptVelocityChange",
    "RewardRootVelocityDifference",
    "RewardTwoFactorVelocityDifference",
    "RewardActiveHeadingDirection",
]
