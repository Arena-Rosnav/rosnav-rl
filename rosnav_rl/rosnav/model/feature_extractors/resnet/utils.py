from torch import nn


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
