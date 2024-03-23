from torch import nn, Tensor
import torch
from typing import List, Union, Optional, Callable
import numpy as np
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


class ResNet(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        in_planes: int,
        base_planes: int,
        block_type: Union[BasicBlock, Bottleneck],
        cardinality: int = 1,
        norm_layer: Optional[Callable] = None
    ):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.block_type = block_type
        self.cardinality = cardinality
        self.base_planes = base_planes
        
        self.conv1 = nn.Conv2d(in_planes, base_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_planes = base_planes
        self.norm1 = self.norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_layer1 = self._make_layer(
            block_count=layer_sizes[0], planes=self.base_planes
        )
        self.res_layer2 = self._make_layer(
            block_count=layer_sizes[1], planes=self.base_planes * 2, stride=2
        )
        self.res_layer3 = self._make_layer(
            block_count=layer_sizes[2], planes=self.base_planes * 2 * 2, stride=2
        )
        self.res_layer4 = self._make_layer(
            block_count=layer_sizes[3], planes=self.base_planes * 2 * 2 * 2, stride=2
        )
        
        
        
    def _make_layer(
        self,
        block_count: int,
        planes: int,
        stride: int = 1
    ) -> nn.Sequential:
        # first block
        # downsample
        if stride != 1 or self.in_planes != planes * self.block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * self.block_type.expansion, stride),
                self.norm_layer(planes * self.block_type.expansion)
            )
        else:
            downsample = None
        
        layers = [
            self.block_type(
                self.in_planes,
                planes,
                stride,
                downsample,
                groups=self.cardinality,
                base_width=self.base_planes,
                norm_layer=self.norm_layer
            )
        ]
        self.in_planes = planes * self.block_type.expansion
        
        # Rest blocks
        for _ in range(1, block_count):
            layers.append(
                self.block_type(
                    self.in_planes,
                    planes,
                    groups=self.cardinality,
                    base_width=self.base_planes,
                    norm_layer=self.norm_layer
                )
            )
            
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        
        return x

    
def resnet50_groupnorm(input_channels: int, num_groups: int):
    return ResNet(
        layer_sizes=[3, 4, 6, 3],
        in_planes=input_channels,
        base_planes=64,
        block_type=Bottleneck,
        cardinality=1,
        norm_layer=lambda channels: nn.GroupNorm(num_groups, channels)
    )
