"""
This file contains the definition of a custom CNN model `MID_FUSION_BOTTLENECK_EXTRACTOR_1` 
for feature extraction in a reinforcement learning environment. 

Source:
    https://ieeexplore.ieee.org/document/10089196

Details:
    - The model is implemented using PyTorch and inherits from the BaseFeaturesExtractor class.
    - It includes a Bottleneck class that implements a variant of the ResNet architecture known as 
        ResNet V1.5, designed to improve accuracy for image recognition tasks.
    - The MID_FUSION_BOTTLENECK_EXTRACTOR_1 class defines a custom feature extractor that is part of 
        a middle-fusion-network.
    - The feature extractor takes input observations and performs a series of convolutional and 
        batch normalization operations, followed by fusion and goal networks to extract features.
"""
import rospy
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

__all__ = ["MID_FUSION_BOTTLENECK_EXTRACTOR_1"]


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


class Bottleneck(nn.Module):
    """
    The `Bottleneck` class is a module in the torchvision library that implements a variant of the
    ResNet architecture known as ResNet V1.5. It is designed to improve accuracy for image recognition tasks.
    The class consists of an initialization method and a forward method.

    Note:
        Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
        while original implementation places the stride at the first 1x1 convolution(self.conv1)
        according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
        This variant is also known as ResNet V1.5 and improves accuracy according to
        https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    expansion = 2  # 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: callable = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: nn.Module = None,
    ):
        """
        The method initializes various layers and parameters including:
            - convolutional layers (`conv1`, `conv2`, `conv3`),
            - batch normalization layers (`bn1`, `bn2`, `bn3`),
            - ReLU activation function (`relu`),
            - downsampling function (`downsample`),
            - and stride (`stride`).

        Args:
            inplanes (int): Number of input channels
            planes (int): Number of output channels
            stride (int, optional): Stride for downsampling. Defaults to 1.
            downsample (callable, optional): Optional downsampling function. Defaults to None.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
            base_width (int, optional): Base width for the bottleneck. Defaults to 64.
            dilation (int, optional): Dilation rate for dilated convolution. Defaults to 1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass takes an input tensor `x` and performs the following operations:
            1. Assigns the input tensor to the variable `identity`
            2. Applies a 1x1 convolution, batch normalization, and ReLU activation (conv1, bn1, relu)
            3. Applies a 3x3 convolution, batch normalization, and ReLU activation (conv2, bn2, relu)
            4. Applies another 1x1 convolution and batch normalization (conv3, bn3)
            5. If downsampling is specified, applies the downsampling function to the input tensor
            6. Adds the input tensor to the output tensor (Skip-Connection)
            7. Applies ReLU activation to the output tensor
            8. Returns the output tensor

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MID_FUSION_BOTTLENECK_EXTRACTOR_1(BaseFeaturesExtractor):
    """
    This class defines a custom feature extractor `SEM_EXTRACTOR_1` that inherits from `BaseFeaturesExtractor`.
    It is supposed to be part of an middle-fusion-network.

    The feature extractor consists of a fusion net and a goal net.
    The fusion net processes the input data (pedestrian position and scan) and the goal net processes the goal tensor.
    The fusion net includes convolutional layers, batch normalization, and residual connections using bottleneck blocks.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Initializes the SEM_EXTRACTOR_1 feature extractor.

        Args:
            observation_space (gym.spaces.Box): Observation space of the environment
            features_dim (int, optional): Number of features extracted. Defaults to 256.
        """
        rospy.set_param("rl_agent/resnet", True)
        # network parameters:
        block = Bottleneck
        layers = [2, 1, 1]
        zero_init_residual = True
        groups = 1
        width_per_group = 64
        replace_stride_with_dilation = None
        norm_layer = None

        # superclass properties/methods
        super(MID_FUSION_BOTTLENECK_EXTRACTOR_1, self).__init__(
            observation_space, features_dim
        )

        ################## ped_pos net model: ###################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(256),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(1, 1),
                stride=(2, 2),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(256),
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(512),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=512,
                kernel_size=(1, 1),
                stride=(4, 4),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(512),
        )
        self.relu3 = nn.ReLU(inplace=True)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_fc = nn.Sequential(
            nn.Linear(256 * block.expansion + 2, features_dim),
            # nn.BatchNorm1d(features_dim),
            nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):  # add by xzt
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        """
        Constructs a layer using the specified block type and parameters.

        Args:
            block (Bottleneck): Type of block to use
            planes (int): Number of output channels
            blocks (int): The number of block layers
            stride (int, optional): Stride for the layer. Defaults to 1.
            dilate (bool, optional): Whether to apply dilation. Defaults to False.

        Returns:
            nn.Sequential: A sequential layer constructed using the specified parameters
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(
        self, ped_pos: torch.Tensor, scan: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Implements the forward pass for the feature extractor.

        Args:
            ped_pos (torch.Tensor): Pedestrian position tensor
            scan (torch.Tensor): Scan tensor
            goal (torch.Tensor): Goal tensor

        Returns:
            torch.Tensor: Output tensor after forward pass
        """
        ###### Start of fusion net ######
        ped_in = ped_pos.reshape(-1, 2, 80, 80)
        scan_in = scan.reshape(-1, 1, 80, 80)
        fusion_in = torch.cat((scan_in, ped_in), dim=1)

        # See note [TorchScript super()]
        x = self.conv1(fusion_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity3 = self.downsample3(x)

        x = self.layer1(x)

        identity2 = self.downsample2(x)

        x = self.layer2(x)

        x = self.conv2_2(x)
        x += identity2
        x = self.relu2(x)

        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.conv3_2(x)
        x += identity3
        x = self.relu3(x)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)
        ###### End of fusion net ######

        ###### Start of goal net #######
        goal_in = goal.reshape(-1, 2)
        goal_out = torch.flatten(goal_in, 1)
        ###### End of goal net #######
        # Combine
        fc_in = torch.cat((fusion_out, goal_out), dim=1)
        x = self.linear_fc(fc_in)

        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the feature extractor.

        Args:
            observations (torch.Tensor): Input observations

        Returns:
            torch.Tensor: Output tensor after forward pass
        """
        # preprocessing:
        # scan_map = observations[:, :6400]
        # ped_map = observations[:, 6400:32000]
        # goal = observations[:, 32000:32002]
        # last_action = observations[:, 32002:32005]

        scan = observations[:, :6400]
        ped_pos = observations[:, 6400:19200]
        goal = observations[:, 19200:]
        return self._forward_impl(ped_pos, scan, goal)
