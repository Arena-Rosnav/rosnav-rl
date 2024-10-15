"""Custom policies built by SB3 during runtime through parsing 'policy_kwargs'"""

from ..feature_extractors.resnet.resnet import (
    RESNET_MID_FUSION_EXTRACTOR_1,
    RESNET_MID_FUSION_EXTRACTOR_2,
    RESNET_MID_FUSION_EXTRACTOR_3,
    RESNET_MID_FUSION_EXTRACTOR_4,
    RESNET_MID_FUSION_EXTRACTOR_5,
    RESNET_MID_FUSION_EXTRACTOR_6,
    DRL_VO_NAV_EXTRACTOR,
    DRL_VO_NAV_EXTRACTOR_TEST,
    _LaserTest,
    _LaserTest_deep,
    DRL_VO_DEEP,
    DRL_VO_ROSNAV_EXTRACTOR,
)
import rosnav_rl.spaces.observation_space as SPACE

from ..feature_extractors.rgbd.rgbd_feature_nets import (
    RESNET_RGBD_FUSION_EXTRACTOR_1,
)
from torch import nn

from ..agent_factory import AgentFactory
from ..base_policy import StableBaselinesPolicy, PolicyType
from ..feature_extractors import *


@AgentFactory.register("AGENT_19")
class AGENT_19(StableBaselinesPolicy):
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_20")
class AGENT_20(StableBaselinesPolicy):
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [dict(pi=[128], vf=[128])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_21")
class AGENT_21(StableBaselinesPolicy):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 5,
    }
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24")
class AGENT_24(StableBaselinesPolicy):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 5,
    }
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24_stacked")
class AGENT_24_stacked(StableBaselinesPolicy):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
    }
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_5_extended
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU
    log_std_init = -2.0


@AgentFactory.register("AGENT_24_stacked_lstm")
class AGENT_24_stacked_lstm(StableBaselinesPolicy):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
    }
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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


@AgentFactory.register("AGENT_24_lstm")
class AGENT_24_lstm(StableBaselinesPolicy):
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
    }
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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


@AgentFactory.register("AGENT_22")
class AGENT_22(StableBaselinesPolicy):
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_23")
class AGENT_23(StableBaselinesPolicy):
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=128)
    net_arch = [128, 64, 64, 64]
    activation_fn = nn.ReLU


# lstm
@AgentFactory.register("AGENT_32")
class AGENT_32(StableBaselinesPolicy):
    observation_spaces = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    type = PolicyType.MULTI_INPUT_LSTM
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU
    n_lstm_layers = 2
    lstm_hidden_size = 256
    shared_lstm = True
    enable_critic_lstm = False


# lstm + framestacking
@AgentFactory.register("AGENT_35")
class AGENT_35(StableBaselinesPolicy):
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
class AGENT_36(StableBaselinesPolicy):
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
class AGENT_38(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT_LSTM
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
class AGENT_39(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT_LSTM
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
class AGENT_41(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT_LSTM
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
class AGENT_52(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT_LSTM
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
class AGENT_55(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_7
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [512, 256, 64]
    activation_fn = nn.ReLU


# framestacking
@AgentFactory.register("AGENT_57")
class AGENT_57(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_8
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [512, 256, 64, 64]
    activation_fn = nn.ReLU


# framestacking
@AgentFactory.register("AGENT_58")
class AGENT_58(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_8
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[256, 256, 64], vf=[256, 64])
    activation_fn = nn.ReLU


# framestacking
@AgentFactory.register("AGENT_59")
class AGENT_59(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT
    features_extractor_class = EXTRACTOR_9
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = dict(pi=[256, 256, 64], vf=[256, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("BarnResNet")
class BarnResNet(StableBaselinesPolicy):
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

    type = PolicyType.MULTI_INPUT

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
        "goal_max_dist": 10,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_1
    features_extractor_kwargs = {"features_dim": 256}
    net_arch = dict(pi=[256], vf=[128])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_1")
class RosnavResNet_1(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
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
class RosnavResNet_2(StableBaselinesPolicy):
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

    type = PolicyType.MULTI_INPUT

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class RosnavResNet_3(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
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
class RosnavResNet_4(StableBaselinesPolicy):
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

    type = PolicyType.MULTI_INPUT

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class RosnavResNet_5(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class RosnavResNet_6(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class RosnavResNet_7(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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


@AgentFactory.register("ArenaUnityResNet_1")
class ArenaUnityResNet_1(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT

    observation_spaces = [
        SPACE.RGBDSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {"image_height": 128, "image_width": 128}
    features_extractor_class = RESNET_RGBD_FUSION_EXTRACTOR_1
    features_extractor_kwargs = {
        "features_dim": 512,
        "num_groups": 4,
        "image_height": 128,
        "image_width": 128,
    }
    net_arch = dict(pi=[512, 128], vf=[512])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_5_norm")
class RosnavResNet_5_norm(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class RosnavResNet_6_norm(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class LSTM_ResNet_5_norm(StableBaselinesPolicy):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class LSTM_ResNet_norm_1(StableBaselinesPolicy):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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


@AgentFactory.register("LSTM_ResNet_norm_3")
class LSTM_ResNet_norm_3(StableBaselinesPolicy):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class LSTM_ResNet_norm_4(StableBaselinesPolicy):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
class LSTM_ResNet_norm_5(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
    n_lstm_layers = 3
    lstm_hidden_size = 512
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("RosnavResNet_8_norm")
class RosnavResNet_8_norm(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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


@AgentFactory.register("LSTM_ResNet_norm_6")
class LSTM_ResNet_norm_6(StableBaselinesPolicy):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
    net_arch = dict(pi=[256, 256], vf=[256, 256])
    activation_fn = nn.ReLU
    log_std_init = -2
    ortho_init = False
    n_lstm_layers = 2
    lstm_hidden_size = 512
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("LSTM_ResNet_norm_7")
class LSTM_ResNet_norm_7(StableBaselinesPolicy):
    type = PolicyType.MLP_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
    net_arch = dict(pi=[256, 256], vf=[256, 256])
    activation_fn = nn.ReLU
    log_std_init = -2
    ortho_init = False
    n_lstm_layers = 4
    lstm_hidden_size = 256
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("LSTM_ResNet_norm_8")
class LSTM_ResNet_norm_8(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
    log_std_init = -2
    ortho_init = False
    n_lstm_layers = 4
    lstm_hidden_size = 256
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("RosnavResNet_simple")
class RosnavResNet_simple(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = DRL_VO_NAV_EXTRACTOR
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_deeper")
class RosnavResNet_deeper(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = DRL_VO_NAV_EXTRACTOR_TEST
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU


@AgentFactory.register("LSTM_ResNet_simple")
class LSTM_ResNet_simple(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = DRL_VO_NAV_EXTRACTOR
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU
    log_std_init = -2
    ortho_init = False
    n_lstm_layers = 2
    lstm_hidden_size = 256
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("LaserTest")
class LaserTest(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.DistAngleToSubgoalSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = _LaserTest
    features_extractor_kwargs = {
        "features_dim": 128,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[128], vf=[64])
    activation_fn = nn.ReLU


@AgentFactory.register("LSTM_ResNet_simple_3")
class LSTM_ResNet_simple_3(StableBaselinesPolicy):
    type = PolicyType.MULTI_INPUT_LSTM
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU
    log_std_init = -2
    ortho_init = False
    n_lstm_layers = 2
    lstm_hidden_size = 256
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("RosnavResNet_mid")
class RosnavResNet_mid(StableBaselinesPolicy):
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
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = RESNET_MID_FUSION_EXTRACTOR_5
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_mid_2")
class RosnavResNet_mid_2(StableBaselinesPolicy):
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
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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
    net_arch = dict(pi=[256, 128], vf=[256, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet")
class RosnavResNet(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
        "goal_max_dist": 5,
    }
    features_extractor_class = DRL_VO_ROSNAV_EXTRACTOR
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU


@AgentFactory.register("DeepLaserTest")
class DeepLaserTest(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.DistAngleToSubgoalSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = _LaserTest_deep
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[128], vf=[64])
    activation_fn = nn.ReLU


@AgentFactory.register("DeepDRLVOTest")
class DeepDRLVOTest(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = DRL_VO_DEEP
    features_extractor_kwargs = {
        "features_dim": 256,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256])
    activation_fn = nn.ReLU


@AgentFactory.register("RosnavResNet_LSTM")
class RosnavResNet_LSTM(StableBaselinesPolicy):
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

    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
    }
    features_extractor_class = DRL_VO_NAV_EXTRACTOR
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 128], vf=[256, 64])
    activation_fn = nn.ReLU
    ortho_init = False
    n_lstm_layers = 2
    lstm_hidden_size = 256
    shared_lstm = True
    enable_critic_lstm = False


@AgentFactory.register("RosnavResNet__1")
class RosnavResNet__1(StableBaselinesPolicy):
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
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
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


@AgentFactory.register("RosnavResNet__2")
class RosnavResNet__2(StableBaselinesPolicy):
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
    observation_spaces = [
        SPACE.StackedLaserMapSpace,
        SPACE.PedestrianVelXSpace,
        SPACE.PedestrianVelYSpace,
        SPACE.PedestrianTypeSpace,
        SPACE.PedestrianSocialStateSpace,
        SPACE.SubgoalInRobotFrameSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
        "goal_max_dist": 5,
    }
    features_extractor_class = DRL_VO_ROSNAV_EXTRACTOR
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 64], vf=[256, 64])
    activation_fn = nn.ReLU
