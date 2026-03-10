from typing import Literal, Union

from pydantic import Field
from typing_extensions import Annotated

from rosnav_rl.cfg.framework import FrameworkCfg
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks

from .a2c import A2C_Cfg
from .base import SBAlgorithmCfg
from .crossq import CrossQ_Cfg
from .ppo import PPO_Cfg
from .sac import SAC_Cfg
from .td3 import TD3_Cfg
from .tqc import TQC_Cfg
from .trpo import TRPO_Cfg


class StableBaselinesCfg(FrameworkCfg):
    """Configuration for the Stable Baselines 3 framework.

    ``algorithm`` accepts any of the supported algorithm-specific config
    classes.  When parsing from a plain dict the correct sub-class is
    selected automatically because each ``parameters`` model carries an
    ``algorithm_name`` class variable that acts as a discriminator.

    Attributes:
        name: Framework identifier (always ``stable_baselines3``).
        algorithm: Algorithm envelope containing architecture name,
            checkpoint info, and algorithm-specific hyper-parameters.
    """

    name: Literal[SupportedRLFrameworks.STABLE_BASELINES3] = (
        SupportedRLFrameworks.STABLE_BASELINES3
    )
    algorithm: Annotated[
        Union[
            PPO_Cfg,
            A2C_Cfg,
            TRPO_Cfg,
            SAC_Cfg,
            TD3_Cfg,
            TQC_Cfg,
            CrossQ_Cfg,
            SBAlgorithmCfg,
        ],
        Field(discriminator="type"),
    ]
