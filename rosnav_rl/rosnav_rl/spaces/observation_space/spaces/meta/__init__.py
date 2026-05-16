"""Meta Spaces - Production Ready & Legacy

This module contains both production-ready and legacy meta observation spaces.

Production Spaces:
    - MissionContextSpace: Mission context with progress
    - PerformanceContextSpace: Performance metrics
    - SafetyContextSpace: Safety risk assessment

Legacy Spaces:
    - IsFirstStepSpace: Episode first step indicator
    - IsTerminalStepSpace: Episode terminal step indicator
    - EpisodeStepSpace: Episode step counter
"""

# Basic spaces
from .basic_meta_spaces import (
    IsFirstStepSpace,
    IsTerminalStepSpace,
    EpisodeStepSpace,
)

# Advanced spaces
from .advanced_meta_spaces import (
    MissionContextSpace,
    PerformanceContextSpace,
    SafetyContextSpace,
)

__all__ = [
    "IsFirstStepSpace",
    "IsTerminalStepSpace",
    "EpisodeStepSpace",
    "MissionContextSpace",
    "PerformanceContextSpace",
    "SafetyContextSpace",
]
