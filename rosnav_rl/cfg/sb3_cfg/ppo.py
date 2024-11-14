from typing import Optional, Union

from .base import BaseAlgorithmCfg, BaseAlgorithmParameters


class PPO_Algorithm_Cfg(BaseAlgorithmParameters):
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Union[float, callable] = 0.2
    clip_range_vf: Union[None, float, callable] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None


class PPO_Cfg(BaseAlgorithmCfg):
    parameters: PPO_Algorithm_Cfg
