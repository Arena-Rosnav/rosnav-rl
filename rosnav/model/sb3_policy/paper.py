import rosnav.utils.observation_space as spaces
from rosnav.rosnav_space_manager.base_space_encoder import BaseSpaceEncoder
from torch import nn

from ..agent_factory import AgentFactory
from ..base_agent import BaseAgent, PolicyType
from ..feature_extractors.classic import *
from ..feature_extractors.resnet.resnet import DRL_VO_ROSNAV_EXTRACTOR


@AgentFactory.register("AGENT_1")
class AGENT_1(BaseAgent):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 5,
    }
    observation_spaces = [
        spaces.LaserScanSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_2")
class AGENT_2(BaseAgent):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
    }
    observation_spaces = [
        spaces.LaserScanSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT_LSTM
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU
    n_lstm_layers = 2
    lstm_hidden_size = 128
    shared_lstm = False
    enable_critic_lstm = True
    log_std_init = -2.0


@AgentFactory.register("AGENT_3")
class AGENT_3(BaseAgent):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
    }
    observation_spaces = [
        spaces.LaserScanSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_5_extended
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU
    log_std_init = -2.0


@AgentFactory.register("AGENT_4")
class AGENT_4(BaseAgent):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
    }
    observation_spaces = [
        spaces.LaserScanSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT_LSTM
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[128, 64], vf=[128, 64])
    activation_fn = nn.GELU
    n_lstm_layers = 2
    lstm_hidden_size = 128
    shared_lstm = False
    enable_critic_lstm = True
    log_std_init = -2.0


@AgentFactory.register("AGENT_5")
class AGENT_5(BaseAgent):
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

    type = PolicyType.MULTI_INPUT
    space_encoder_class = BaseSpaceEncoder
    observation_spaces = [
        spaces.StackedLaserMapSpace,
        spaces.PedestrianVelXSpace,
        spaces.PedestrianVelYSpace,
        spaces.PedestrianTypeSpace,
        spaces.PedestrianSocialStateSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
        "goal_max_dist": 10,
    }
    features_extractor_class = DRL_VO_ROSNAV_EXTRACTOR
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 128], vf=[256, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_6")
class AGENT_6(BaseAgent):
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

    type = PolicyType.MULTI_INPUT_LSTM
    space_encoder_class = BaseSpaceEncoder
    observation_spaces = [
        spaces.StackedLaserMapSpace,
        spaces.PedestrianVelXSpace,
        spaces.PedestrianVelYSpace,
        spaces.PedestrianTypeSpace,
        spaces.PedestrianSocialStateSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
        "goal_max_dist": 10,
    }
    features_extractor_class = DRL_VO_ROSNAV_EXTRACTOR
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 128], vf=[256, 64])
    activation_fn = nn.GELU
    log_std_init = -2
    ortho_init = False
    n_lstm_layers = 2
    lstm_hidden_size = 128
    shared_lstm = False
    enable_critic_lstm = True
