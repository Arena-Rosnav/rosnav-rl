import torch
from torch import nn

from .utils import conv1x1, conv3x3


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
