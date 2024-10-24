from torch import nn, Tensor
import torch
from typing import List, Union, Optional, Callable, Any
from .resblocks import BasicBlock, Bottleneck, conv1x1


class ResNet(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        in_planes: int,
        base_planes: int,
        block_type: Union[BasicBlock, Bottleneck],
        cardinality: int = 1,
        norm_layer: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.block_type = block_type
        self.cardinality = cardinality
        self.base_planes = base_planes

        self.conv1 = nn.Conv2d(
            in_planes, base_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
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
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_planes = self.in_planes
        self.out_shape = torch.Size(
            [
                self.in_planes,
            ]
        )

    def _make_layer(
        self, block_count: int, planes: int, stride: int = 1
    ) -> nn.Sequential:
        # first block
        # downsample
        if stride != 1 or self.in_planes != planes * self.block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * self.block_type.expansion, stride),
                self.norm_layer(planes * self.block_type.expansion),
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
                norm_layer=self.norm_layer,
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
                    norm_layer=self.norm_layer,
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

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        return x


def resnet50_groupnorm(input_channels: int, num_groups, **kwargs):
    """Instantiates a ResNet50 with GroupNorm as normalization layer.

    Args:
        input_channels (int): Number of channels of the input image, e.g. RGB: 3, RGBD: 4.
        num_groups (int): Number of groups for the GroupNorm layers.

    Returns:
        ResNet50 nn.Module object from ResNet class.
    """
    return ResNet(
        layer_sizes=[3, 4, 6, 3],
        in_planes=input_channels,
        base_planes=64,
        block_type=Bottleneck,
        cardinality=1,
        norm_layer=lambda channels: nn.GroupNorm(num_groups, channels),
    )


class RgbdPerceptionNet(nn.Module):
    """Encapsulates the chosen backbone for the RGBD perception part.
    NOTICE: The backbone is passed via a factory method. The backbone should be a
    ResNet object which results in a (out_planes, 1, 1) tensor where out_planes
    is variable.

    Args:
        out_dim (int): The final output dimension of the RGBD perception.
        network_factory (Callable[..., ResNet]): Method which instantiates the backbone.
            This method will be given the given kwargs.
    """

    def __init__(
        self,
        out_dim: int,
        input_channels: int,
        network_factory: Callable[..., ResNet],
        **kwargs: Any
    ):
        super(RgbdPerceptionNet, self).__init__()

        self.input_channels = input_channels
        self.output = out_dim

        self.net = network_factory(self.input_channels, **kwargs)
        self.fc = nn.Linear(in_features=self.net.out_planes, out_features=out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        x = self.net(x)
        x = self.fc(x)
        x = self.relu(x)

        return x
