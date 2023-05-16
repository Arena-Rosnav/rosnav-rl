"""Custom policies built by SB3 during runtime through parsing 'policy_kwargs'"""
from torch import nn

from .agent_factory import AgentFactory
from .base_agent import BaseAgent, PolicyType
from .feature_extractors import *


@AgentFactory.register("AGENT_6")
class AGENT_6(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [128, 64, 64, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_7")
class AGENT_7(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [128, 128, 128, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_8")
class AGENT_8(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [64, 64, 64, 64, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_9")
class AGENT_9(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [64, 64, 64, 64, dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_10")
class AGENT_10(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [128, 128, 128, 128, dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_11")
class AGENT_11(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [512, 512, 512, 512, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_17")
class AGENT_17(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_4
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_18")
class AGENT_18(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_4
    features_extractor_kwargs = dict(features_dim=128)
    net_arch = [128, dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_19")
class AGENT_19(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_20")
class AGENT_20(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [dict(pi=[128], vf=[128])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_21")
class AGENT_21(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_22")
class AGENT_22(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_23")
class AGENT_23(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=128)
    net_arch = [128, 64, 64, 64]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_24")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_25")
class AGENT_25(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [512, 256, 128]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


@AgentFactory.register("AGENT_26")
class AGENT_26(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [256, 128]
    activation_fn = nn.ReLU
    n_lstm_layers = 16

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# framestacking
@AgentFactory.register("AGENT_27")
class AGENT_27(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = [256, 128]
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# lstm# lstm
@AgentFactory.register("AGENT_28")
class AGENT_28(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = [256, 128]
    activation_fn = nn.ReLU
    n_lstm_layers = 6
    lstm_hidden_size = 128
    shared_lstm = False
    enable_critic_lstm = True

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# lstm + framestacking
@AgentFactory.register("AGENT_29")
class AGENT_29(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [256, 128]
    activation_fn = nn.ReLU
    n_lstm_layers = 12
    lstm_hidden_size = 512
    shared_lstm = False
    enable_critic_lstm = True

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# lstm
@AgentFactory.register("AGENT_30")
class AGENT_30(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[256, 256, 64], vf=[256, 256])
    activation_fn = nn.ReLU
    n_lstm_layers = 6
    lstm_hidden_size = 256
    shared_lstm = False
    enable_critic_lstm = True

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# framestacking
@AgentFactory.register("AGENT_31")
class AGENT_31(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[512, 256, 128, 64], vf=[256, 256, 64])
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# lstm
@AgentFactory.register("AGENT_32")
class AGENT_32(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[512, 256], vf=[256, 256])
    activation_fn = nn.ReLU
    n_lstm_layers = 6
    lstm_hidden_size = 64
    shared_lstm = False
    enable_critic_lstm = True

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# framestacking
@AgentFactory.register("AGENT_33")
class AGENT_33(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=258)
    net_arch = dict(pi=[256, 256, 256], vf=[256, 256, 256])
    activation_fn = nn.ReLU

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# lstm + framestacking
@AgentFactory.register("AGENT_34")
class AGENT_34(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[512, 256, 256], vf=[256, 256, 128])
    activation_fn = nn.ReLU
    n_lstm_layers = 12
    lstm_hidden_size = 128
    shared_lstm = False
    enable_critic_lstm = True

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# lstm + framestacking
@AgentFactory.register("AGENT_35")
class AGENT_35(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = [256, 256]
    activation_fn = nn.ReLU
    n_lstm_layers = 4
    lstm_hidden_size = 128
    shared_lstm = False
    enable_critic_lstm = True

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model


# lstm
@AgentFactory.register("AGENT_36")
class AGENT_36(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[256, 64, 64], vf=[256, 256])
    activation_fn = nn.ReLU
    n_lstm_layers = 6
    lstm_hidden_size = 512
    shared_lstm = False
    enable_critic_lstm = True

    def __init__(self, robot_model: str = None):
        self.robot_model = robot_model
