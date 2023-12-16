from enum import Enum
from typing import List, Type, Union

from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.module import Module


class PolicyType(Enum):
    CNN = "CnnPolicy"
    MLP = "MlpPolicy"
    MLP_LSTM = "MlpLstmPolicy"


# Parsed as ppo_kwargs to sb3 ppo class
BASE_AGENT_ATTR = {
    "observation_space_manager": ObservationSpaceManager,
    "features_extractor_class": Union[Type[BaseFeaturesExtractor], None],
    "features_extractor_kwargs": Union[dict, None],
    "net_arch": Union[List[Union[int, dict]], None],
    "activation_fn": Union[Type[Module], None],
    # LSTM
    # Number of LSTM layers
    "n_lstm_layers": int,
    # Number of hidden units for each LSTM layer
    "lstm_hidden_size": int,
    #  Whether the LSTM is shared between the actor and the critic
    # (in that case, only the actor gradient is used)
    # By default, the actor and the critic have two separate LSTM.
    "shared_lstm": bool,
    # Use a seperate LSTM for the critic
    "enable_critic_lstm": bool,
    "lstm_kwargs": Union[dict, None],
}

"""
n_lstm_layers: 
This parameter determines the number of LSTM layers in the model. Each LSTM layer processes the input sequence and generates a hidden state 
that is passed to the next layer. Increasing the number of LSTM layers can help the model learn more complex patterns in the input sequence, 
but it can also increase the risk of overfitting.

lstm_hidden_size: 
This parameter determines the number of hidden units for each LSTM layer. Increasing this parameter can help the model learn more complex patterns 
in the input sequence, but it can also increase the risk of overfitting.

shared_lstm: 
This parameter determines whether the LSTM is shared between the actor and the critic. If this parameter is set to True, 
then only the actor gradient is used. By default, the actor and the critic have two separate LSTMs.

enable_critic_lstm: 
This parameter determines whether a separate LSTM is used for the critic. 
If this parameter is set to True, then a separate LSTM is used for the critic. This can help improve performance by allowing the critic to learn a 
separate representation of the input sequence.
"""
