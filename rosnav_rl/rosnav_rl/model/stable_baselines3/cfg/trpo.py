from typing import Optional

from sb3_contrib.trpo import TRPO

from .base import OnPolicyParameters, SBAlgorithmCfg


class TRPO_Algorithm_Cfg(OnPolicyParameters):
    """Configuration parameters for Trust Region Policy Optimization (TRPO).

    TRPO constrains policy updates to a trust region defined by the KL
    divergence, yielding monotonic improvement guarantees.

    Attributes:
        cg_max_steps: Maximum conjugate-gradient iterations for the natural
            gradient computation.
        cg_damping: Damping coefficient for the Fisher vector product.
        line_search_shrinking_factor: Step-size shrinking factor for the
            backtracking line search.
        line_search_max_iter: Maximum line-search iterations.
        n_critic_updates: Value-function updates per policy update.
        target_kl: Target KL-divergence threshold.
        sub_sampling_factor: Fraction of the batch used for the Fisher
            matrix–vector product (1.0 = full batch).
    """

    algorithm_name = TRPO.__name__

    # TRPO does a single large step → no minibatch epochs
    n_epochs: int = 1

    cg_max_steps: int = 15
    cg_damping: float = 0.1
    line_search_shrinking_factor: float = 0.8
    line_search_max_iter: int = 10
    n_critic_updates: int = 10
    target_kl: float = 0.01
    sub_sampling_factor: float = 1.0


class TRPO_Cfg(SBAlgorithmCfg):
    """Top-level TRPO configuration (architecture + hyper-parameters).

    Attributes:
        parameters: TRPO-specific algorithm parameters.
    """

    parameters: TRPO_Algorithm_Cfg
