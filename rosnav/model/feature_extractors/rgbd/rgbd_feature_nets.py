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
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 observation_space_manager: ObservationSpaceManager,
                 features_dim: int = 256,
                 rgbd_backbone: Callable[..., ResNet] = resnet50_groupnorm,
                 rgbd_out_dim: int = -1,
                 img_channels: int = 4
                 ):
        pass
