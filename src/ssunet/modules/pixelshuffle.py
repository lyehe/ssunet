"""Pixel Shuffle Layers."""

import torch
from torch import nn

from ..exceptions import (
    ChannelDivisibilityError,
    InputDimensionError,
    SizeDivisibilityError,
)


class PixelShuffle3d(nn.Module):
    """This class is a 3d version of pixelshuffle."""

    def __init__(self, scale: int = 2) -> None:
        """Initialize PixelShuffle3d.

        :param scale: upsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        if input.dim() != 5:
            raise InputDimensionError(5, input.dim())
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        if in_channels % (scale**3) != 0:
            raise ChannelDivisibilityError(in_channels, 3)
        out_channels = in_channels // (scale**3)
        out_z, out_x, out_y = z * scale, x * scale, y * scale
        view_shape = (batch, out_channels, scale, scale, scale, z, x, y)
        input_view = input.contiguous().view(*view_shape)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch, out_channels, out_z, out_x, out_y)


class PixelUnshuffle3d(nn.Module):
    """This class is a 3d version of pixelunshuffle."""

    def __init__(self, scale: int = 2) -> None:
        """Initialize PixelUnshuffle3d.

        :param scale: downsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        if input.dim() != 5:
            raise InputDimensionError(5, input.dim())
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        if z % self.scale != 0 or x % self.scale != 0 or y % self.scale != 0:
            raise SizeDivisibilityError((z, x, y))
        out_channels = in_channels * (self.scale**3)
        out_z, out_x, out_y = z // scale, x // scale, y // scale
        view_shape = (batch, in_channels, out_z, scale, out_x, scale, out_y, scale)
        input_view = input.contiguous().view(*view_shape)
        output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return output.view(batch, out_channels, out_z, out_x, out_y)


class PixelShuffle2d(nn.Module):
    """This class is a 2d version of pixelshuffle on BCZXY data on XY."""

    def __init__(self, scale: int = 2) -> None:
        """Initialize PixelShuffle2d.

        :param scale: upsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        if input.dim() != 5:
            raise InputDimensionError(5, input.dim())
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        if in_channels % (scale**2) != 0:
            raise ChannelDivisibilityError(in_channels, 2)
        out_channels = in_channels // (scale**2)
        out_x, out_y = x * scale, y * scale
        input_view = input.contiguous().view(batch, out_channels, scale, scale, z, x, y)
        output = input_view.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        return output.view(batch, out_channels, z, out_x, out_y)


class PixelUnshuffle2d(nn.Module):
    """This class is a 2d version of pixelunshuffle on BCZXY data on XY."""

    def __init__(self, scale: int = 2) -> None:
        """Initialize PixelUnshuffle2d.

        :param scale: downsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        if input.dim() != 5:
            raise InputDimensionError(5, input.dim())
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        if x % self.scale != 0 or y % self.scale != 0:
            raise SizeDivisibilityError((x, y))
        out_channels = in_channels * (self.scale**2)
        out_x, out_y = x // scale, y // scale
        view_shape = (batch, in_channels, z, out_x, scale, out_y, scale)
        input_view = input.contiguous().view(*view_shape)
        output = input_view.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        return output.view(batch, out_channels, z, out_x, out_y)
