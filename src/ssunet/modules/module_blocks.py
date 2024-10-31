"""Modules for the SSUnet."""

from abc import abstractmethod
from functools import partial

import torch
from torch import nn

from .base_modules import (
    activation_function,
    conv111,
    conv333,
    conv777,
    merge,
    merge_conv,
    partial333,
    pool,
    upconv222,
)

_EncoderOut = tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]


class UnetBlockConv3D(nn.Module):
    """A base class for the Unet block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        z_conv: bool = True,
        skip_out: bool = True,
        batch_norm: bool = False,
        group_norm: int = 0,
        dropout_p: float = 0.0,
        partial_conv: bool = False,
        last: bool = False,
        down_mode: str = "maxpool",
        up_mode: str = "transpose",
        merge_mode: str = "concat",
        activation: str = "relu",
        **kwargs,
    ) -> None:
        """Initializes the UnetBlockConv3D class.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param z_conv: if True, the convolution will be 3D, defaults to True
        :param skip_out: if True, the output will include the skip connection, defaults to True
        :param batch_norm: determines whether to use batch normalization, defaults to False
        :param group_norm: determines whether to use group normalization, defaults to 0
        :param dropout_p: dropout probability, defaults to 0
        :param last: if True, the block is the last in the network, defaults to False
        :param down_mode: mode of downsampling, defaults to "maxpool"
        :param up_mode: mode of upsampling, defaults to "transpose"
        :param merge_mode: mode of merging, defaults to "concat"
        :param activation: activation function, defaults to "relu"
        :param kwargs: additional keyword arguments
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_conv = z_conv
        self.skip_out = skip_out
        self.dropout_p = dropout_p
        self.partial_conv = partial_conv
        self.last = last
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.kwargs = kwargs

        self.conv3d = partial333 if partial_conv else conv333

        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        n = group_norm > 0 and out_channels % group_norm == 0
        self.group_norm = nn.GroupNorm(group_norm, out_channels) if n else nn.Identity()
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0.01 else nn.Identity()
        self.merge = partial(merge, merge_mode=merge_mode)
        self.merge_conv = partial(merge_conv, z_conv=z_conv, mode=merge_mode)
        self.conv333 = partial(self.conv3d, z_conv=z_conv)
        self.down_sample = partial(pool, down_mode=down_mode, z_conv=z_conv, last=last)
        self.up_sample = upconv222(in_channels, out_channels, z_conv, up_mode=up_mode)
        self.activation = activation_function(activation)
        self.__other__()

    @abstractmethod
    def __other__(self):
        """Define other modules."""


class DownConvDual3D(UnetBlockConv3D):
    """Simplified DownConv block with residual connection.

    Performs 2 convolutions and 1 MaxPool. A ReLU activation follows each convolution.
    """

    def __other__(self):
        """Define other modules."""
        self.residual = conv111(self.in_channels, self.out_channels)
        self.conv1 = self.conv333(self.in_channels, self.out_channels)
        self.conv2 = self.conv333(self.out_channels, self.out_channels)
        self.pool = self.down_sample(self.out_channels, self.out_channels)

    def forward(self, input):
        """Forward pass."""
        residual = self.residual(input)
        input = self.activation(self.conv1(input))
        input = self.activation(self.group_norm(self.conv2(input)))
        input = self.dropout(input)
        before_pool = input + residual
        output = self.pool(before_pool)
        return (output, before_pool) if self.skip_out else (output, None)


class UpConvDual3D(UnetBlockConv3D):
    """A helper Module that performs 2 convolutions and 1 UpConvolution.

    A ReLU activation follows each convolution.
    """

    def __other__(self):
        """Define other modules."""
        merge_channels = self.in_channels if self.merge_mode == "concat" else self.out_channels
        self.resconv = self.merge_conv(self.out_channels, self.out_channels)
        self.conv1 = self.conv333(merge_channels, self.out_channels)
        self.conv2 = self.conv333(self.out_channels, self.out_channels)

    def forward(self, input: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass."""
        input = self.up_sample(input)
        input = self.merge(input, skip)
        residual = self.group_norm(self.resconv(input)) if skip is not None else input
        input = self.activation(self.group_norm(self.conv1(input)))
        input = self.activation(self.group_norm(self.conv2(input)))
        input = self.dropout(input)
        output = input + residual
        return output


class DownConvTri3D(UnetBlockConv3D):
    """Helper Module that performs 2 convolutions and 1 MaxPool.

    A ReLU activation follows each convolution.
    """

    def __other__(self):
        """Define other modules."""
        self.resconv = self.conv333(self.in_channels, self.out_channels)
        self.conv2 = self.conv333(self.out_channels, self.out_channels)
        self.conv3 = self.conv333(self.out_channels, self.out_channels)
        self.pool = self.down_sample(self.out_channels, self.out_channels)

    def forward(self, input: torch.Tensor) -> _EncoderOut:
        """Forward pass."""
        residual = self.group_norm(self.resconv(input))
        input = self.activation(self.group_norm(self.conv2(residual)))
        input = self.activation(self.group_norm(self.conv3(input) + residual))
        before_pool = self.dropout(input)
        output = self.pool(before_pool)
        return (output, before_pool) if self.skip_out else (output, None)


class UpConvTri3D(UnetBlockConv3D):
    """A helper Module that performs 2 convolutions and 1 UpConvolution.

    A ReLU activation follows each convolution.
    """

    def __other__(self):
        """Define other modules."""
        self.resconv = self.merge_conv(self.out_channels, self.out_channels)
        self.conv2 = self.conv333(self.out_channels, self.out_channels)
        self.conv3 = self.conv333(self.out_channels, self.out_channels)

    def forward(self, input: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass."""
        input = self.up_sample(input)
        input = self.merge(input, skip)
        residual = self.group_norm(self.resconv(input)) if skip is not None else input
        input = self.activation(self.group_norm(self.conv2(residual)))
        input = self.activation(self.group_norm(self.conv3(input) + residual))
        output = self.dropout(input)
        return output


class LKDownConv3D(UnetBlockConv3D):
    """A helper Module that performs 2 convolutions and 1 MaxPool.

    A ReLU activation follows each convolution.
    """

    def __other__(self):
        """Define other modules."""
        in_channels = self.in_channels
        out_channels = self.out_channels
        z_conv = self.z_conv
        self.conv333_1 = self.conv333(in_channels, out_channels)
        self.conv333_2 = self.conv333(out_channels, out_channels)
        self.conv111 = conv111(out_channels, out_channels)
        self.conv777 = conv777(
            out_channels,
            out_channels,
            z_conv,
            separable=self.kwargs.get("separable", True),
        )
        self.pool = self.down_sample(self.out_channels, self.out_channels)

    def forward(self, input: torch.Tensor) -> _EncoderOut:
        """Forward pass."""
        input = self.activation(self.group_norm(self.conv333_1(input)))
        input = self.activation(
            input + self.conv111(input) + self.conv333_2(input) + self.conv777(input)
        )
        before_pool = self.dropout(input)
        output = self.pool(before_pool)
        return (output, before_pool) if self.skip_out else (output, None)
