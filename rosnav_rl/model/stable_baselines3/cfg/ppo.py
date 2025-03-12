from typing import Optional, Union

from stable_baselines3.ppo import PPO

from .base import SBAlgorithmCfg, SBAlgorithmParameters


class PPO_Algorithm_Cfg(SBAlgorithmParameters):
    """
    Configuration parameters for the Proximal Policy Optimization (PPO) algorithm.

    This class extends SBAlgorithmParameters to specify the hyperparameters used in the
    PPO algorithm from Stable Baselines 3. Each parameter controls different aspects of
    the training process.

    Attributes:
        gamma (float): Discount factor for future rewards (between 0 and 1).
        gae_lambda (float): Factor for Generalized Advantage Estimation.
        clip_range (Union[float, callable]): Clipping parameter for the policy loss.
        clip_range_vf (Union[None, float, callable]): Clipping parameter for the value function.
            If None, no clipping is performed.
        normalize_advantage (bool): Whether to normalize advantages or not.
        ent_coef (float): Entropy coefficient for the loss calculation.
        vf_coef (float): Value function coefficient for the loss calculation.
        max_grad_norm (float): Maximum value for the gradient norm clipping.
        use_sde (bool): Whether to use State Dependent Exploration (SDE).
        sde_sample_freq (int): Sample a new noise matrix every n steps when using SDE.
            If negative, noise is sampled only at the beginning of the rollout.
        target_kl (Optional[float]): Limit the KL divergence between updates, for early stopping.
            If None, no early stopping is performed.
    """
    algorithm_name = PPO.__name__
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


class PPO_Cfg(SBAlgorithmCfg):
    """
    Configuration class for Proximal Policy Optimization (PPO) algorithm.

    This class inherits from SBAlgorithmCfg and provides a structure for configuring PPO
    algorithm parameters in the ROS-Nav reinforcement learning framework.

    Attributes:
        parameters (PPO_Algorithm_Cfg): Configuration parameters specific to the PPO algorithm,
            including learning rate, batch size, n_steps, gamma, etc.
    """
    parameters: PPO_Algorithm_Cfg
