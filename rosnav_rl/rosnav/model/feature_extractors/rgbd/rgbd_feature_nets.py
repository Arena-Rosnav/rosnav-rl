from copy import deepcopy
from typing import List, Callable

import gymnasium as gym
from gymnasium.spaces.box import Box
from torch.nn.modules import BatchNorm2d, Module
import rospy
import torch
import torch.nn as nn
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX

from ..base_extractor import RosnavBaseExtractor
from .resent import resnet50_groupnorm, RgbdPerceptionNet, ResNet



class RESNET_RGBD_FUSION_EXTRACTOR_1(RosnavBaseExtractor):
    REQUIRED_OBSERVATIONS = [
        SPACE_INDEX.RGBD,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION
    ]
    
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 observation_space_manager: ObservationSpaceManager,
                 features_dim: int = 512 + 32 + 32,
                 rgbd_backbone: Callable[..., ResNet] = resnet50_groupnorm,
                 rgbd_out_dim: int = 512,
                 *args,
                 **kwargs
                 ):
        self._features_dim = features_dim
        self._rgbd_backbone = rgbd_backbone
        self._rgbd_out_dim = rgbd_out_dim
        
        self._observation_space_manager = observation_space_manager
        
        super(RESNET_RGBD_FUSION_EXTRACTOR_1, self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            args=args,
            kwargs=kwargs
        )
        
    def _get_input_sizes(self):
        self._goal_size = self._observation_space_manager[SPACE_INDEX.GOAL].shape[-1]
        self._last_action_size = self._observation_space_manager[SPACE_INDEX.LAST_ACTION].shape[-1]
        self.
    
    def _setup_network(self, *args, **kwargs):
        # RGBD part
        self.perception_net = RgbdPerceptionNet(self._rgbd_out_dim, self._rgbd_backbone, kwargs)
        
        # goal part
        self.goal_fc = nn.Linear(in_features=)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        pass
