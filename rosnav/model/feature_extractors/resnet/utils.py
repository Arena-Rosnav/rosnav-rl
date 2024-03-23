from torch import nn
from typing import List, Union, Optional, Callable
import numpy as np
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv2x2


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """
    3x3 convolution with padding

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride for the convolution. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        dilation (int, optional): Dilation rate for dilated convolution. Defaults to 1.

    Returns:
        nn.Conv2d: 3x3 convolutional layer with specified parameters
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """
    1x1 convolution

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride for the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: 1x1 convolutional layer with specified parameters
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
        
        self.conv1 = nn.Conv2d(in_planes, base_planes, kernal_size=7, stride=2, padding=3, bias=False)
        self.in_planes = in_planes
        self.norm1 = norm_layer(self.in_planes)
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
        if stride != 1 or self.inplanes != planes * self.block_type.expansion:
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
        
        