from typing import Type

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from torch import nn

import rosnav_rl.spaces.observation_space as spaces

from ..agent_factory import AgentFactory
from ..base_policy import StableBaselinesPolicyDescription
from ..feature_extractors.classic import *
from ..feature_extractors.resnet.resnet import DRL_VO_ROSNAV_EXTRACTOR

from sb3_contrib import RecurrentPPO

from rosnav_rl.utils.type_aliases import ObservationSpaceList, ObservationSpaceKwargs


@AgentFactory.register("AGENT_1")
class AGENT_1(StableBaselinesPolicyDescription):
    """
    AGENT_1 is a custom policy class for the Stable Baselines3 library using the PPO algorithm.

    Attributes:
        algorithm_class (Type[BaseAlgorithm]): The algorithm class to be used, which is PPO in this case.
        observation_space_kwargs (ObservationSpaceKwargs): A dictionary containing keyword arguments for the observation space configuration.
            - normalize (bool): Whether to normalize the observations.
            - goal_max_dist (int): Maximum distance to the goal.
            - subgoal_max_dist (int): Maximum distance to the subgoal.
            - reduced_num_beams (int): Number of beams in the reduced laser scan.
        observation_spaces (ObservationSpaceList): A list of observation spaces used by the policy.
            - spaces.ReducedLaserScanSpace: Observation space for reduced laser scan.
            - spaces.DistAngleToSubgoalSpace: Observation space for distance and angle to subgoal.
            - spaces.LastActionSpace: Observation space for the last action taken.
        features_extractor_class: The class used for feature extraction, which is EXTRACTOR_5.
        features_extractor_kwargs (dict): A dictionary containing keyword arguments for the feature extractor.
            - features_dim (int): Dimension of the extracted features.
        net_arch (dict): A dictionary defining the architecture of the neural network.
            - pi (list): Architecture of the policy network.
            - vf (list): Architecture of the value function network.
        activation_fn: The activation function used in the neural network, which is ReLU.
    """

    algorithm_class: Type[BaseAlgorithm] = PPO
    observation_space_kwargs: ObservationSpaceKwargs = {
        "normalize": True,
        "goal_max_dist": 10,
        "subgoal_max_dist": 10,
        "reduced_num_beams": 360,
    }
    observation_spaces: ObservationSpaceList = [
        spaces.ReducedLaserScanSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_2")
class AGENT_2(StableBaselinesPolicyDescription):
    """
    AGENT_2 is a custom policy class for a reinforcement learning agent using the RecurrentPPO algorithm.

    Attributes:
        algorithm_class (Type[BaseAlgorithm]): The algorithm class to be used, which is RecurrentPPO.
        observation_space_kwargs (dict): Keyword arguments for configuring the observation space.
            - normalize (bool): Whether to normalize the observations.
            - goal_max_dist (int): Maximum distance to the goal.
            - subgoal_max_dist (int): Maximum distance to the subgoal.
            - reduced_num_beams (int): Number of beams in the reduced laser scan.
        observation_spaces (list): List of observation space classes used by the agent.
            - spaces.ReducedLaserScanSpace: Observation space for reduced laser scan.
            - spaces.DistAngleToSubgoalSpace: Observation space for distance and angle to subgoal.
            - spaces.LastActionSpace: Observation space for the last action taken.
        features_extractor_class: The class used for feature extraction, which is EXTRACTOR_5.
        features_extractor_kwargs (dict): Keyword arguments for the feature extractor.
            - features_dim (int): Dimension of the extracted features.
        net_arch (dict): Network architecture for the policy and value function.
            - pi (list): Architecture for the policy network.
            - vf (list): Architecture for the value function network.
        activation_fn: Activation function used in the neural network, which is nn.ReLU.
        n_lstm_layers (int): Number of LSTM layers.
        lstm_hidden_size (int): Size of the hidden state in the LSTM.
        shared_lstm (bool): Whether the LSTM is shared between the policy and value function.
        enable_critic_lstm (bool): Whether to enable LSTM for the critic.
        log_std_init (float): Initial value for the log standard deviation of the policy.
    """

    algorithm_class: Type[BaseAlgorithm] = RecurrentPPO
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
        "subgoal_max_dist": 10,
        "reduced_num_beams": 360,
    }
    observation_spaces = [
        spaces.ReducedLaserScanSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
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
class AGENT_3(StableBaselinesPolicyDescription):
    """
    AGENT_3 class implementing a specific Stable Baselines policy.

    Attributes:
        algorithm_class (Type[BaseAlgorithm]): The algorithm class to be used, defaults to PPO.
        stack_size (int): The size of the stack, defaults to 10.
        observation_space_kwargs (dict): Keyword arguments for the observation space configuration.
            - normalize (bool): Whether to normalize the observations, defaults to True.
            - goal_max_dist (int): Maximum distance to the goal, defaults to 10.
            - subgoal_max_dist (int): Maximum distance to the subgoal, defaults to 10.
            - reduced_num_beams (int): Number of beams in the reduced laser scan, defaults to 360.
        observation_spaces (list): List of observation spaces used by the agent.
            - spaces.ReducedLaserScanSpace: Reduced laser scan space.
            - spaces.DistAngleToSubgoalSpace: Distance and angle to subgoal space.
            - spaces.LastActionSpace: Last action space.
        features_extractor_class: The class used for feature extraction, defaults to EXTRACTOR_5_extended.
        features_extractor_kwargs (dict): Keyword arguments for the feature extractor.
            - features_dim (int): Dimension of the features, defaults to 256.
        net_arch (dict): Network architecture for the policy and value function.
            - pi (list): Architecture for the policy network, defaults to [64, 64].
            - vf (list): Architecture for the value function network, defaults to [64, 64].
        activation_fn: The activation function used in the network, defaults to nn.ReLU.
        log_std_init (float): Initial value for the log standard deviation, defaults to -2.0.
    """

    algorithm_class: Type[BaseAlgorithm] = PPO
    stack_size = 10
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
        "subgoal_max_dist": 10,
        "reduced_num_beams": 360,
    }
    observation_spaces = [
        spaces.ReducedLaserScanSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    features_extractor_class = EXTRACTOR_5_extended
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU
    log_std_init = -2.0


@AgentFactory.register("AGENT_4")
class AGENT_4(StableBaselinesPolicyDescription):
    """
    AGENT_4 is a custom policy class for a reinforcement learning agent using the RecurrentPPO algorithm from Stable Baselines3.

    Attributes:
        algorithm_class (Type[BaseAlgorithm]): The algorithm class to be used, which is RecurrentPPO.
        stack_size (int): The size of the observation stack.
        observation_space_kwargs (dict): Keyword arguments for configuring the observation space.
            - normalize (bool): Whether to normalize the observations.
            - goal_max_dist (int): Maximum distance to the goal.
            - subgoal_max_dist (int): Maximum distance to the subgoal.
            - reduced_num_beams (int): Number of beams in the reduced laser scan.
        observation_spaces (list): List of observation space classes used by the agent.
            - spaces.ReducedLaserScanSpace: Observation space for reduced laser scan.
            - spaces.DistAngleToSubgoalSpace: Observation space for distance and angle to subgoal.
            - spaces.LastActionSpace: Observation space for the last action taken.
        features_extractor_class: The class used for feature extraction, which is EXTRACTOR_5.
        features_extractor_kwargs (dict): Keyword arguments for the feature extractor.
            - features_dim (int): Dimension of the extracted features.
        net_arch (dict): Network architecture for the policy and value function.
            - pi (list): Architecture for the policy network.
            - vf (list): Architecture for the value function network.
        activation_fn: The activation function used in the neural network, which is nn.GELU.
        n_lstm_layers (int): Number of LSTM layers.
        lstm_hidden_size (int): Size of the hidden state in the LSTM.
        shared_lstm (bool): Whether the LSTM is shared between the policy and value function.
        enable_critic_lstm (bool): Whether to enable LSTM for the critic.
        log_std_init (float): Initial value for the log standard deviation of the policy.
    """

    algorithm_class: Type[BaseAlgorithm] = RecurrentPPO
    stack_size = 10
    observation_space_kwargs = {
        "normalize": True,
        "goal_max_dist": 10,
        "subgoal_max_dist": 10,
        "reduced_num_beams": 360,
    }
    observation_spaces = [
        spaces.ReducedLaserScanSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
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
class AGENT_5(StableBaselinesPolicyDescription):
    """
    AGENT_5 is a custom policy class for the Stable Baselines3 reinforcement learning library.

    Attributes:
        algorithm_class (Type[BaseAlgorithm]): The algorithm class to be used, which is PPO in this case.
        observation_spaces (list): A list of observation space classes used by the agent.
        observation_space_kwargs (dict): A dictionary of keyword arguments for configuring the observation spaces.
            - roi_in_m (int): Region of interest in meters.
            - feature_map_size (int): Size of the feature map.
            - laser_stack_size (int): Number of laser stacks.
            - normalize (bool): Whether to normalize the observations.
            - goal_max_dist (int): Maximum distance to the goal.
            - subgoal_max_dist (int): Maximum distance to the subgoal.
        features_extractor_class (Type): The class used for feature extraction, which is DRL_VO_ROSNAV_EXTRACTOR.
        features_extractor_kwargs (dict): A dictionary of keyword arguments for configuring the feature extractor.
            - features_dim (int): Dimension of the extracted features.
            - width_per_group (int): Width per group for the feature extractor.
        net_arch (dict): A dictionary defining the architecture of the neural network.
            - pi (list): List of layer sizes for the policy network.
            - vf (list): List of layer sizes for the value function network.
        activation_fn (Type[nn.Module]): The activation function used in the neural network, which is ReLU.
    """

    algorithm_class: Type[BaseAlgorithm] = PPO
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
        "subgoal_max_dist": 10,
    }
    features_extractor_class = DRL_VO_ROSNAV_EXTRACTOR
    features_extractor_kwargs = {
        "features_dim": 512,
        "width_per_group": 64,
    }
    net_arch = dict(pi=[256, 128], vf=[256, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_6")
class AGENT_6(StableBaselinesPolicyDescription):
    """
    AGENT_6 is a custom policy class for a reinforcement learning agent using the RecurrentPPO algorithm from Stable Baselines3.

    Attributes:
        algorithm_class (Type[BaseAlgorithm]): The algorithm class to be used, which is RecurrentPPO.
        observation_spaces (list): A list of observation space classes used by the agent.
        observation_space_kwargs (dict): Keyword arguments for configuring the observation spaces.
            - roi_in_m (int): Region of interest in meters.
            - feature_map_size (int): Size of the feature map.
            - laser_stack_size (int): Number of laser stacks.
            - normalize (bool): Whether to normalize the observations.
            - goal_max_dist (int): Maximum distance to the goal.
        features_extractor_class: The class used for feature extraction, which is DRL_VO_ROSNAV_EXTRACTOR.
        features_extractor_kwargs (dict): Keyword arguments for configuring the feature extractor.
            - features_dim (int): Dimension of the extracted features.
            - width_per_group (int): Width per group for the feature extractor.
        net_arch (dict): Network architecture for the policy and value function.
            - pi (list): Architecture for the policy network.
            - vf (list): Architecture for the value function network.
        activation_fn: The activation function used in the neural network, which is nn.GELU.
        log_std_init (float): Initial value for the log standard deviation.
        ortho_init (bool): Whether to use orthogonal initialization.
        n_lstm_layers (int): Number of LSTM layers.
        lstm_hidden_size (int): Hidden size of the LSTM layers.
        shared_lstm (bool): Whether the LSTM is shared between the policy and value function.
        enable_critic_lstm (bool): Whether to enable LSTM for the critic.
    """

    algorithm_class: Type[BaseAlgorithm] = RecurrentPPO
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
