"""Helper functions to create modules."""

import torch
from torch import nn

from ..constants import LOGGER
from ..exceptions import ShapeMismatchError
from .partialconv import PartialConv2d, PartialConv3d
from .pixelshuffle import (
    PixelShuffle2d,
    PixelShuffle3d,
    PixelUnshuffle2d,
    PixelUnshuffle3d,
)
from .separableconv import SeparableConv3d


def conv111(
    in_channels: int,
    out_channels: int,
) -> nn.Conv3d:
    """Helper function to create 1x1x1 convolutions.

    :param in_channels: number of input channels
    :param out_channels: number of output channels

    :return: 1x1x1 convolution
    """
    return nn.Conv3d(in_channels, out_channels, 1)


def convnnn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    z_conv: bool,  # True = 3D convolution & False = 2D convolution
    partial: bool = False,
) -> nn.Conv3d:
    """Helper function to create nxnxn convolutions with padding.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of the convolving kernel
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param partial: if True, use partial convolutions, defaults to False

    :return: nxnxn convolution
    """
    padding_size = kernel_size // 2
    padding = padding_size if z_conv else (0, padding_size, padding_size)
    kernel = kernel_size if z_conv else (1, kernel_size, kernel_size)
    return (
        nn.Conv3d(in_channels, out_channels, kernel, padding=padding)
        if not partial
        else PartialConv3d(in_channels, out_channels, kernel, padding=padding)
    )


def conv333(
    in_channels: int,
    out_channels: int,
    z_conv: bool,  # True = 3D convolution & False = 2D convolution
) -> nn.Conv3d:
    """Helper function to create 3x3x3 convolutions with padding.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution

    :return: 3x3x3 convolution
    """
    return convnnn(in_channels, out_channels, 3, z_conv)


def conv555(
    in_channels: int,
    out_channels: int,
    z_conv: bool,  # True = 3D convolution & False = 2D convolution
    separable: bool = False,
) -> nn.Conv3d | nn.Module:
    """Helper function to create 5x5x5 convolutions with padding.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param separable: if True, use separable convolutions, defaults to False

    :return: 5x5x5 convolution
    """
    if separable:
        return SeparableConv3d(in_channels, out_channels, 5, z_conv)
    else:
        return convnnn(in_channels, out_channels, 5, z_conv)


def conv777(
    in_channels: int,
    out_channels: int,
    z_conv: bool,  # True = 3D convolution & False = 2D convolution
    separable: bool = False,
) -> nn.Conv3d | nn.Module:
    """Helper function to create 7x7x7 convolutions with padding.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param separable: if True, use separable convolutions, defaults to False

    :return: 7x7x7 convolution
    """
    if separable:
        return SeparableConv3d(in_channels, out_channels, 7, z_conv)
    else:
        return convnnn(in_channels, out_channels, 7, z_conv)


def maxpool_downsample(
    z_conv: bool,  # True = 3D convolution & False = 2D convolution
) -> nn.MaxPool3d:
    """Helper function to create maxpooling with padding.

    :param z_conv: True = 3D convolution & False = 2D convolution

    :return: maxpooling layer
    """
    kernel = 2 if z_conv else (1, 2, 2)
    stride = 2 if z_conv else (1, 2, 2)
    return nn.MaxPool3d(kernel, stride=stride)


def avgpool_downsample(
    z_conv: bool,  # True = 3D convolution & False = 2D convolution
) -> nn.AvgPool3d:
    """Helper function to create avgpooling with padding.

    :param z_conv: True = 3D convolution & False = 2D convolution

    :return: avgpooling layer
    """
    kernel = 2 if z_conv else (1, 2, 2)
    stride = 2 if z_conv else (1, 2, 2)
    return nn.AvgPool3d(kernel, stride=stride)


def conv_downsample(
    in_channels: int,
    out_channels: int,
    z_conv: bool,  # True = 3D convolution & False = 2D convolution
) -> nn.Conv3d:
    """Helper function to create 3x3x3 convolutions with padding.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution

    :return: 3x3x3 convolution
    """
    kernel = 3 if z_conv else (1, 3, 3)
    padding = 1 if z_conv else (0, 1, 1)
    stride = 2 if z_conv else (1, 2, 2)
    return nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding)


def pixelunshuffle(in_channels: int, out_channels: int, z_conv: bool, scale: int = 2) -> nn.Module:
    """Helper function to create pixelunshuffle layers.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param scale: scale of pixelunshuffle in each dim, defaults to 2

    :return: pixelunshuffle layer
    """
    if in_channels == out_channels:
        return PixelUnshuffle3d(scale) if z_conv else PixelUnshuffle2d(scale)
    else:
        return (
            nn.Sequential(
                PixelUnshuffle3d(scale),
                conv111(in_channels * (scale**3), out_channels),
            )
            if z_conv
            else nn.Sequential(
                PixelUnshuffle2d(scale),
                conv111(in_channels * (scale**2), out_channels),
            )
        )


def pool(
    in_channels: int,
    out_channels: int,
    down_mode: str,
    z_conv: bool,
    last: bool = False,
) -> nn.Module:
    """Helper function to create pooling layers.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param down_mode: type of downsample ("maxpool" | "avgpool" | "conv" | "unshuffle")
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param last: no pooling at the last layer, defaults to False

    :return: pooling layer
    """
    if last:
        return nn.Identity()
    match down_mode:
        case "maxpool":
            return maxpool_downsample(z_conv)
        case "avgpool":
            return avgpool_downsample(z_conv)
        case "conv":
            return conv_downsample(in_channels, out_channels, z_conv)
        case "unshuffle":
            return pixelunshuffle(in_channels, out_channels, z_conv)
        case _:
            LOGGER.warning(f"Unknown downsample mode: {down_mode}. Using maxpool.")
            return maxpool_downsample(z_conv)


def pixelshuffle(in_channels: int, out_channels: int, z_conv: bool, scale: int = 2) -> nn.Module:
    """Helper function to create pixelshuffle layers.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param scale: scale of pixelshuffle in each dim, defaults to 2

    :return: pixelshuffle layer
    """
    if in_channels // (scale**3) == out_channels:
        return PixelShuffle3d(scale) if z_conv else PixelShuffle2d(scale)
    else:
        return (
            nn.Sequential(
                PixelShuffle3d(scale),
                conv333(in_channels // (scale**3), out_channels, z_conv),
            )
            if z_conv
            else nn.Sequential(
                PixelShuffle2d(scale),
                conv333(in_channels // (scale**2), out_channels, z_conv),
            )
        )


def upconv222(
    in_channels: int,
    out_channels: int,
    z_conv: bool,
    up_mode: str = "transpose",  # type of upconvolution ("transpose" | "upsample" | "pixelshuffle")
) -> nn.Module:
    """Helper function to create 2x2x2 upconvolutions.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param up_mode: type of upconvolution ("transpose" | "upsample" | "pixelshuffle")

    :return: 2x2x2 upconvolution
    """
    kernel = 2 if z_conv else (1, 2, 2)
    stride = 2 if z_conv else (1, 2, 2)
    scale_factor = (2, 2, 2) if z_conv else (1, 2, 2)

    match up_mode:
        case "transpose":
            return nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride)
        case "pixelshuffle":
            return pixelshuffle(in_channels, out_channels, z_conv, scale=2)
        case "bilinear":
            return nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=scale_factor),
                conv333(in_channels, out_channels, z_conv),
            )
        case "nearest":
            return nn.Sequential(
                nn.Upsample(mode="nearest", scale_factor=scale_factor),
                conv333(in_channels, out_channels, z_conv),
            )
        case "trilinear":
            return nn.Sequential(
                nn.Upsample(mode="trilinear", scale_factor=scale_factor),
                conv333(in_channels, out_channels, z_conv),
            )
        case _:
            LOGGER.warning(f"Unknown up_mode: {up_mode}. Using transpose instead.")
            return nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride)


def partial333(
    in_channels: int,
    out_channels: int,
    z_conv: bool,
    multi_channel: bool = True,  # Changed default to True
) -> PartialConv3d:
    """Helper function to create 3x3x3 partial convolutions with padding.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param multi_channel: if True, the mask will be applied to all channels, defaults to True

    :return: 3x3x3 partial convolution
    """
    padding = 1 if z_conv else (0, 1, 1)
    kernel = (3, 3, 3) if z_conv else (1, 3, 3)
    return PartialConv3d(
        in_channels,
        out_channels,
        kernel,
        padding=padding,
        multi_channel=multi_channel,
    )


def partial33(
    in_channels: int,
    out_channels: int,
    multi_channel: bool = True,  # New helper for 2D with multi_channel=True
) -> PartialConv2d:
    """Helper function to create 3x3 partial convolutions with padding.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param multi_channel: if True, the mask will be applied to all channels, defaults to True

    :return: 3x3 partial convolution
    """
    padding = (1, 1)
    kernel = (3, 3)
    return PartialConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        padding=padding,
        multi_channel=multi_channel,
    )


def merge(
    input_a: torch.Tensor,
    input_b: torch.Tensor | None,
    merge_mode: str = "concat",
) -> torch.Tensor:
    """Helper function to merge two tensors.

    :param input_a: input tensor A
    :param input_b: input tensor B
    :param merge_mode: merge mode ("concat" | "add")

    :return: merged tensor
    """
    if input_b is None:
        return input_a
    match merge_mode:
        case "concat":
            if input_a.shape[1:] != input_b.shape[1:]:
                raise ShapeMismatchError()
            return torch.cat((input_a, input_b), dim=1)
        case "add":
            if input_a.shape != input_b.shape:
                raise ShapeMismatchError()
            return input_a + input_b
        case _:
            LOGGER.warning(f"Unknown merge_mode: {merge_mode}. Using concat instead.")
            return torch.cat((input_a, input_b), dim=1)


def merge_conv(
    in_channels: int,
    out_channels: int,
    z_conv: bool,
    mode: str = "concat",
) -> nn.Module:
    """Helper function to merge two tensors.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param z_conv: True = 3D convolution & False = 2D convolution
    :param mode: merge mode ("concat" | "add")

    :return: 3x3x3 convolution
    """
    match mode:
        case "concat":
            return conv333(in_channels * 2, out_channels, z_conv)
        case "add":
            return conv333(in_channels, out_channels, z_conv)
        case _:
            LOGGER.warning(f"Unknown mode: {mode}. Using concat instead.")
            return conv333(in_channels * 2, out_channels, z_conv)


def activation_function(activation: str, **kwargs) -> nn.Module:
    """Helper function to create activation layers.

    :param activation: activation function

    :return: activation layer
    """
    match activation:
        case "relu":
            return nn.ReLU(inplace=kwargs.get("inplace", True))
        case "leakyrelu":
            return nn.LeakyReLU(
                kwargs.get("negative_slope", 0.01),
                inplace=kwargs.get("inplace", True),
            )
        case "prelu":
            return nn.PReLU(
                num_parameters=kwargs.get("num_parameters", 1),
                init=kwargs.get("init", 0.25),
            )
        case "gelu":
            return nn.GELU(approximate=kwargs.get("approximate", "none"))
        case "silu":
            return nn.SiLU(inplace=kwargs.get("inplace", True))
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "softmax":
            return nn.Softmax(dim=kwargs.get("dim", None))
        case "logsoftmax":
            return nn.LogSoftmax(dim=kwargs.get("dim", None))
        case _:
            LOGGER.warning(f"Unknown activation: {activation}. Using ReLU instead.")
            return nn.ReLU(inplace=kwargs.get("inplace", True))
