from copy import deepcopy
from typing import List, Callable

import gymnasium as gym
from gymnasium.spaces.box import Box
from torch.nn.modules import BatchNorm2d, Module
import rospy
import torch
import torch.nn as nn
from torch import Tensor
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX

from ..base_extractor import RosnavBaseExtractor
from .resnet import resnet50_groupnorm, RgbdPerceptionNet, ResNet



class RESNET_RGBD_FUSION_EXTRACTOR_1(RosnavBaseExtractor):
    """Feature extractor for RGBD enabled robots. Currently works
    with the Arena unity training environment. The architecture is 
    a combination of our previous Rosnav RL feature extractors and
    the feature extractor proposed in DD-PPO paper 
    (https://arxiv.org/pdf/1911.00357.pdf).
    The network relies only on the RGBD camera for detecting static
    or dynamic obstacles, i.e. no laser.

    Args:
        observation_space (gym.spaces.Box): The observation space of the environment.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int, optional): The dimensionality of the output features. Defaults to 512.
        rgbd_backbone (Callable[..., ResNet], optional): The factory method to initialize a ResNet 
            object as the backbone for the RGBD perception network. Defaults to ResNet50 using GroupNorm.
        rgbd_out_dim (int, optional): Output dimension of the RGBD perception network. Defaults to 512.
        goal_out_dim (int, optional): Output dimension of the goal extractor network. Defaults to 32.
            As proposed in the DD-PPO paper.
        last_action_out_dim (int, optional): Output dimension of the last action extractor network.
            Defaults to 32 like in the DD-PPO paper.
        args: Currenlty not used.
        kwargs: Keyword arguments which are passed to the backbone factory method.
    """
    
    REQUIRED_OBSERVATIONS = [
        SPACE_INDEX.RGBD,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION
    ]
    
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 observation_space_manager: ObservationSpaceManager,
                 image_height: int,
                 image_width: int,
                 features_dim: int = 512,
                 rgbd_backbone: Callable[..., ResNet] = resnet50_groupnorm,
                 rgbd_out_dim: int = 512,
                 goal_out_dim: int = 32,
                 last_action_out_dim: int = 32,
                 **kwargs
                 ):
        self._features_dim = features_dim
        self._rgbd_backbone = rgbd_backbone
        self._rgbd_out_dim = rgbd_out_dim
        self._image_height = image_height
        self._image_width = image_width
        self._goal_out_dim = goal_out_dim
        self._last_action_out_dim = last_action_out_dim
        
        self._observation_space_manager = observation_space_manager
        
        self._get_input_sizes()
        
        super(RESNET_RGBD_FUSION_EXTRACTOR_1, self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            **kwargs
        )
        
    def _get_input_sizes(self):
        self._goal_size = self._observation_space_manager[SPACE_INDEX.GOAL].shape[-1]
        self._last_action_size = self._observation_space_manager[SPACE_INDEX.LAST_ACTION].shape[-1]
        self._image_size = 4 * self._image_height * self._image_width
    
    def _setup_network(self, *args, **kwargs):
        # RGBD part
        self.visual_net = RgbdPerceptionNet(self._rgbd_out_dim, 4, self._rgbd_backbone, **kwargs)
        
        # goal part
        self.goal_net = nn.Sequential(
            nn.Linear(
                in_features=self._goal_size,
                out_features=self._goal_out_dim
            ),
            nn.ReLU(inplace=True)
        )
        
        # last action part
        self.last_action_net = nn.Sequential(
            nn.Linear(
                in_features=self._last_action_size, 
                out_features=self._last_action_out_dim
            ),
            nn.ReLU(inplace=True)
        )
        
        # fusion
        self.fc = nn.Linear(
            in_features=self._rgbd_out_dim + self._goal_out_dim + self._last_action_out_dim,
            out_features=self._features_dim
        )
    
    def _forward_impl(self, image: Tensor, goal: Tensor, last_action: Tensor) -> Tensor:
        # normalize image
        image[:, :3, :, :] /= 255.0  # normalize to [0, 1]
        image[:, 3, :, :] = torch.clamp(image[:, 3, :, :], min=0, max=10)  # clip to [0, 10]
        image[:, 3, :, :] /= 10.0
        
        # seperate nets
        visual_out = self.visual_net(image)
        goal_out = self.goal_net(goal)
        last_action_out = self.last_action_net(last_action)
        
        # concatenation
        cat_out = torch.cat([visual_out, goal_out, last_action_out], dim=1)
        
        # fusion fully connected
        out = self.fc(cat_out)
        
        return out
    
    def _forward(self, observations: Tensor) -> Tensor:
        # extract seperate inputs
        image_vector = observations[:, :self._image_size]
        image = image_vector.reshape(-1, 4, self._image_height, self._image_width)
        
        goal_vector = observations[:, self._image_size:self._image_size + self._goal_size]
        goal = goal_vector.reshape(-1, self._goal_size)
        
        last_action_vector = observations[:, -self._last_action_size:]
        last_action = last_action_vector.reshape(-1, self._last_action_size)
        
        return self._forward_impl(image, goal, last_action)
    
    def forward(self, observations: Tensor) -> Tensor:
        return self._forward(observations)
