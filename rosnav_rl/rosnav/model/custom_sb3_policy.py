"""Custom policies built by SB3 during runtime through parsing 'policy_kwargs'"""

from rosnav.model.feature_extractors.resnet.resnet import (
    RESNET_MID_FUSION_EXTRACTOR_1,
    RESNET_MID_FUSION_EXTRACTOR_2,
    RESNET_MID_FUSION_EXTRACTOR_3,
    RESNET_MID_FUSION_EXTRACTOR_4,
    RESNET_MID_FUSION_EXTRACTOR_5,
    RESNET_MID_FUSION_EXTRACTOR_6,
    RESNET_MID_FUSION_EXTRACTOR_7,
)
from rosnav.rosnav_space_manager.default_encoder import DefaultEncoder
from torch import nn

from .agent_factory import AgentFactory
from .base_agent import BaseAgent, PolicyType
from .feature_extractors import *


@AgentFactory.register("AGENT_19")
class AGENT_19(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_20")
class AGENT_20(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [dict(pi=[128], vf=[128])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_21")
class AGENT_21(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_22")
class AGENT_22(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_23")
class AGENT_23(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=128)
    net_arch = [128, 64, 64, 64]
    activation_fn = nn.ReLU


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


# lstm + framestacking
@AgentFactory.register("AGENT_38")
class AGENT_38(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [256, 256, 256]
    activation_fn = nn.ReLU
    n_lstm_layers = 6
    lstm_hidden_size = 64
    shared_lstm = False
    enable_critic_lstm = True


# lstm
@AgentFactory.register("AGENT_39")
class AGENT_39(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[256, 256, 64], vf=[256, 256])
    activation_fn = nn.ReLU
    n_lstm_layers = 6
    lstm_hidden_size = 128
    shared_lstm = False
    enable_critic_lstm = True


# lstm + framestacking
@AgentFactory.register("AGENT_41")
class AGENT_41(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU
    n_lstm_layers = 4
    lstm_hidden_size = 128
    shared_lstm = True
    enable_critic_lstm = False


# lstm + framestacking
@AgentFactory.register("AGENT_52")
class AGENT_52(BaseAgent):
    type = PolicyType.MLP_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = [64, 64, 64, 64]
    activation_fn = nn.ReLU
    n_lstm_layers = 8
    lstm_hidden_size = 128
    shared_lstm = False
    enable_critic_lstm = True


# framestacking
@AgentFactory.register("AGENT_55")
class AGENT_55(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [512, 256, 64]
    activation_fn = nn.ReLU


# framestacking
@AgentFactory.register("AGENT_57")
class AGENT_57(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_8
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [512, 256, 64, 64]
    activation_fn = nn.ReLU


# framestacking
@AgentFactory.register("AGENT_58")
class AGENT_58(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_8
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[256, 256, 64], vf=[256, 64])
    activation_fn = nn.ReLU


# framestacking
@AgentFactory.register("AGENT_59")
class AGENT_59(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_9
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[256, 256, 64], vf=[256, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("BarnResNet")
class BarnResNet(BaseAgent):
    """
    Custom policy class for BarnResNet.

    This policy uses a ResNet-based feature extractor and a CNN-based space encoder.
    It defines the observation spaces and their corresponding kwargs.
    The network architecture consists of two hidden layers with ReLU activation.

    Reference:
        https://ieeexplore.ieee.org/document/10089196

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for the space encoder.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The kwargs for the observation spaces.
        features_extractor_class (class): The class for the feature extractor.
        features_extractor_kwargs (dict): The kwargs for the feature extractor.
        net_arch (list): The architecture of the network.
        activation_fn (class): The activation function for the hidden layers.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_LOCATION,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.GOAL,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_1
    features_extractor_kwargs = {"features_dim": 256}
    net_arch = dict(pi=[256], vf=[128])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_1")
class RosnavResNet_1(BaseAgent):
    """
    Custom policy class for ROS navigation using ResNet-based CNN.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for encoding the observation space.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The keyword arguments for the observation space.
        features_extractor_class (class): The class for extracting features.
        features_extractor_kwargs (dict): The keyword arguments for the features extractor.
        net_arch (list): The architecture of the neural network.
        activation_fn (function): The activation function used in the neural network.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_2
    features_extractor_kwargs = {
        "features_dim": 256,
        "batch_mode": False,
        "batch_size": 32,
    }
    net_arch = [256, 64]
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_2")
class RosnavResNet_2(BaseAgent):
    """
    A custom policy class for the RosnavResNet_2 agent.

    This policy uses a ResNet-based feature extractor and a CNN-based observation space encoder.
    It defines the observation spaces and their corresponding kwargs, as well as the network architecture.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for the observation space encoder.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The kwargs for the observation spaces.
        features_extractor_class (class): The class for the feature extractor.
        features_extractor_kwargs (dict): The kwargs for the feature extractor.
        net_arch (list): The network architecture.
        activation_fn (class): The activation function for the network.

    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_3
    features_extractor_kwargs = {"features_dim": 512}
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_3")
class RosnavResNet_3(BaseAgent):
    """
    Custom policy class for ROS navigation using ResNet-based CNN.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for encoding the observation space.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The keyword arguments for the observation space.
        features_extractor_class (class): The class for extracting features.
        features_extractor_kwargs (dict): The keyword arguments for the features extractor.
        net_arch (list): The architecture of the neural network.
        activation_fn (function): The activation function used in the neural network.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_4
    features_extractor_kwargs = {
        "features_dim": 256,
    }
    net_arch = dict(pi=[256, 256, 64], vf=[256, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_4")
class RosnavResNet_4(BaseAgent):
    """
    A custom policy class for the RosnavResNet_2 agent.

    This policy uses a ResNet-based feature extractor and a CNN-based observation space encoder.
    It defines the observation spaces and their corresponding kwargs, as well as the network architecture.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for the observation space encoder.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The kwargs for the observation spaces.
        features_extractor_class (class): The class for the feature extractor.
        features_extractor_kwargs (dict): The kwargs for the feature extractor.
        net_arch (list): The network architecture.
        activation_fn (class): The activation function for the network.

    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {"features_dim": 256}
    net_arch = dict(pi=[256], vf=[128])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_5")
class RosnavResNet_5(BaseAgent):
    """
    Custom policy class for ROS navigation using ResNet-based CNN.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for encoding the observation space.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The keyword arguments for the observation space.
        features_extractor_class (class): The class for extracting features.
        features_extractor_kwargs (dict): The keyword arguments for the features extractor.
        net_arch (list): The architecture of the neural network.
        activation_fn (function): The activation function used in the neural network.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_6")
class RosnavResNet_6(BaseAgent):
    """
    Custom policy class for ROS navigation using ResNet-based CNN.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for encoding the observation space.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The keyword arguments for the observation space.
        features_extractor_class (class): The class for extracting features.
        features_extractor_kwargs (dict): The keyword arguments for the features extractor.
        net_arch (list): The architecture of the neural network.
        activation_fn (function): The activation function used in the neural network.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_6
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 128,
    }
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_7")
class RosnavResNet_7(BaseAgent):
    """
    Custom policy class for ROS navigation using ResNet-based CNN.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for encoding the observation space.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The keyword arguments for the observation space.
        features_extractor_class (class): The class for extracting features.
        features_extractor_kwargs (dict): The keyword arguments for the features extractor.
        net_arch (list): The architecture of the neural network.
        activation_fn (function): The activation function used in the neural network.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_6
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 128], vf=[256])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_5_norm")
class RosnavResNet_5_norm(BaseAgent):
    """
    Custom policy class for ROS navigation using ResNet-based CNN.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for encoding the observation space.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The keyword arguments for the observation space.
        features_extractor_class (class): The class for extracting features.
        features_extractor_kwargs (dict): The keyword arguments for the features extractor.
        net_arch (list): The architecture of the neural network.
        activation_fn (function): The activation function used in the neural network.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_6_norm")
class RosnavResNet_6_norm(BaseAgent):
    """
    Custom policy class for ROS navigation using ResNet-based CNN.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for encoding the observation space.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The keyword arguments for the observation space.
        features_extractor_class (class): The class for extracting features.
        features_extractor_kwargs (dict): The keyword arguments for the features extractor.
        net_arch (list): The architecture of the neural network.
        activation_fn (function): The activation function used in the neural network.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 128, 64], vf=[256, 64])
    activation_fn = nn.ReLU


# lstm
@AgentFactory.register("LSTM_ResNet_5_norm")
class LSTM_ResNet_5_norm(BaseAgent):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = []
    activation_fn = nn.ReLU
    n_lstm_layers = 8
    lstm_hidden_size = 256
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("LSTM_ResNet_norm_1")
class LSTM_ResNet_norm_1(BaseAgent):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = []
    activation_fn = nn.ReLU
    n_lstm_layers = 2
    lstm_hidden_size = 512
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("LSTM_ResNet_norm_2")
class LSTM_ResNet_norm_2(BaseAgent):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_7
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = []
    activation_fn = nn.ReLU
    n_lstm_layers = 2
    lstm_hidden_size = 512
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("LSTM_ResNet_norm_3")
class LSTM_ResNet_norm_1(BaseAgent):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 1024,
        "width_per_group": 64,
    }
    net_arch = []
    activation_fn = nn.ReLU
    n_lstm_layers = 2
    lstm_hidden_size = 1024
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("LSTM_ResNet_norm_4")
class LSTM_ResNet_norm_4(BaseAgent):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 256], vf=[256, 64])
    activation_fn = nn.ReLU
    n_lstm_layers = 2
    lstm_hidden_size = 512
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("LSTM_ResNet_norm_5")
class LSTM_ResNet_norm_5(BaseAgent):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 30,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[64])
    activation_fn = nn.ReLU
    n_lstm_layers = 4
    lstm_hidden_size = 512
    shared_lstm = True
    enable_critic_lstm = False

@AgentFactory.register("RosnavResNet_8_norm")
class RosnavResNet_8_norm(BaseAgent):
    """
    Custom policy class for ROS navigation using ResNet-based CNN.

    Attributes:
        type (PolicyType): The type of the policy.
        space_encoder_class (class): The class for encoding the observation space.
        observation_spaces (list): The list of observation spaces.
        observation_space_kwargs (dict): The keyword arguments for the observation space.
        features_extractor_class (class): The class for extracting features.
        features_extractor_kwargs (dict): The keyword arguments for the features extractor.
        net_arch (list): The architecture of the neural network.
        activation_fn (function): The activation function used in the neural network.
    """

    type = PolicyType.CNN
    space_encoder_class = DefaultEncoder
    observation_spaces = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_VEL_X,
        SPACE_INDEX.PEDESTRIAN_VEL_Y,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.PEDESTRIAN_SOCIAL_STATE,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 256, 64], vf=[256, 64])
    activation_fn = nn.ReLU
