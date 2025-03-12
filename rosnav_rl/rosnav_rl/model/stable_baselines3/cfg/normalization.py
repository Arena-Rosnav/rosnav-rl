from pydantic import BaseModel
from typing import Optional


class NormalizationCfg(BaseModel):
    """Configuration class for observation and reward normalization parameters in Stable Baselines 3.

    This class defines parameters used for the VecNormalize wrapper, which normalizes
    observations and rewards for RL training stability.

    Attributes:
        load_from (Optional[str]): Path to load saved VecNormalize statistics.
                                  Default: None (don't load stats).
        norm_obs (Optional[bool]): Whether to normalize observations. Default: False.
        norm_reward (Optional[bool]): Whether to normalize rewards. Default: False.
        clip_obs (Optional[float]): Maximum absolute value for observation clipping.
                                   Default: 30.0.
        clip_reward (Optional[float]): Maximum absolute value for reward clipping.
                                      Default: 30.0.
        gamma (Optional[float]): Discount factor for reward normalization.
                                Default: 0.99.
        epsilon (Optional[float]): Small constant for numerical stability in normalization.
                                  Default: 1e-8.
    """

    load_from: Optional[str] = None
    norm_obs: Optional[bool] = False
    norm_reward: Optional[bool] = False
    clip_obs: Optional[float] = 30.0
    clip_reward: Optional[float] = 30.0
    gamma: Optional[float] = 0.99
    epsilon: Optional[float] = 1e-8
