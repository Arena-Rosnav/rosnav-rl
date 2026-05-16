from typing import Dict, List, Literal, Type, Union

from sb3_contrib import CrossQ, RecurrentPPO, TQC, TRPO
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.module import Module

POLICY_TYPE: Dict[
    Type[BaseAlgorithm], Literal["MultiInputPolicy", "MultiInputLstmPolicy"]
] = {
    # On-policy
    PPO: "MultiInputPolicy",
    A2C: "MultiInputPolicy",
    TRPO: "MultiInputPolicy",
    RecurrentPPO: "MultiInputLstmPolicy",
    # Off-policy
    SAC: "MultiInputPolicy",
    TD3: "MultiInputPolicy",
    DDPG: "MultiInputPolicy",
    TQC: "MultiInputPolicy",
    CrossQ: "MultiInputPolicy",
}

# Parsed as policy_kwargs to the SB3 algorithm class.
# Keys present in this dict are extracted from StableBaselinesPolicyDescription
# instances and forwarded to the SB3 Policy constructor.
BASE_AGENT_ATTR = {
    # Common policy kwargs
    "features_extractor_class": Union[Type[BaseFeaturesExtractor], None],
    "features_extractor_kwargs": Union[dict, None],
    "net_arch": Union[List[Union[int, dict]], None],
    "activation_fn": Union[Type[Module], None],
    # LSTM-specific (RecurrentPPO only — silently ignored for non-recurrent algos)
    "n_lstm_layers": int,
    "lstm_hidden_size": int,
    "shared_lstm": bool,
    "enable_critic_lstm": bool,
    "lstm_kwargs": Union[dict, None],
}
