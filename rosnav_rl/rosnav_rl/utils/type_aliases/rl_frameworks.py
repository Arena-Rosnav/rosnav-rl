from enum import Enum


class SupportedRLFrameworks(str, Enum):
    STABLE_BASELINES3 = "stable_baselines3"
    DREAMER_V3 = "dreamerv3"
