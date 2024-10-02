"""Separable Convolution Layers."""

import torch
from torch import nn


class SeparableConv3d(nn.Module):
    """This class is a 3d version of separable convolution.

    https://arxiv.org/abs/1610.02357
    If z_conv is False, it will perform 2D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        z_conv: bool,
        mid_channels: int | None = None,
    ) -> None:
        """Initialize SeparableConv3d.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolving kernel
        :param z_conv: Ture = 3D convolution & False = 2D convolution
        :param intermed_channels: number of intermediate channels
        """
        super().__init__()
        padding_size = kernel_size // 2
        padding = padding_size if z_conv else (0, padding_size, padding_size)
        kernel = kernel_size if z_conv else (1, kernel_size, kernel_size)
        mid_channels = mid_channels or in_channels
        self.depthwise = nn.Conv3d(
            in_channels, mid_channels, kernel, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.pointwise(self.depthwise(input))
