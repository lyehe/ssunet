import torch
import torch.nn as nn
from .modulets import (
    conv111,
    conv333,
    conv777,
    pool,
    upconv222,
    partial333,
    merge,
    merge_conv,
    activation_function,
)

from abc import abstractmethod
from functools import partial
from typing import TypeAlias

_EncoderOut: TypeAlias = tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]


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
        dropout_p: float = 0,
        last: bool = False,
        down_mode: str = "maxpool",
        up_mode: str = "transpose",
        merge_mode: str = "concat",
        activation: str = "relu",
        **kwargs,
    ):
        """Initializes the UnetBlockConv3D class.

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param z_conv: if True, the convolution will be 3D, defaults to True
        :type z_conv: bool, optional
        :param skip_out: if True, the output will include the skip connection, defaults to True
        :type skip_out: bool, optional
        :param batch_norm: determines whether to use batch normalization, defaults to False
        :type batch_norm: bool, optional
        :param group_norm: determines whether to use group normalization, defaults to 0
        :type group_norm: int, optional
        :param dropout_p: dropout probability, defaults to 0
        :type dropout_p: float, optional
        :param last: if True, the block is the last in the network, defaults to False
        :type last: bool, optional
        :param down_mode: mode of downsampling, defaults to "maxpool"
        :type down_mode: str, optional
        :param up_mode: mode of upsampling, defaults to "transpose"
        :type up_mode: str, optional
        :param merge_mode: mode of merging, defaults to "concat"
        :type merge_mode: str, optional
        :param activation: activation function, defaults to "relu"
        :type activation: str, optional
        :param kwargs: additional keyword arguments
        :type kwargs: dict
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_conv = z_conv
        self.skip_out = skip_out
        self.dropout_p = dropout_p
        self.last = last
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.kwargs = kwargs

        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        n = group_norm > 0 and out_channels % group_norm == 0
        self.group_norm = nn.GroupNorm(group_norm, out_channels) if n else nn.Identity()
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0.01 else nn.Identity()
        self.merge = partial(merge, merge_mode=merge_mode)
        self.merge_conv = partial(merge_conv, z_conv=z_conv, mode=merge_mode)
        self.conv333 = partial(conv333, z_conv=z_conv)
        self.down_sample = partial(pool, down_mode=down_mode, z_conv=z_conv, last=last)
        self.up_sample = upconv222(in_channels, out_channels, z_conv, up_mode=up_mode)
        self.activation = activation_function(activation)
        self.__other__()

    @abstractmethod
    def __other__(self): ...


class DownConvDual3D(UnetBlockConv3D):
    """
    Simplified DownConv block with residual connection.
    Performs 2 convolutions and 1 MaxPool. A ReLU activation follows each convolution.
    """

    def __other__(self):
        self.residual = conv111(self.in_channels, self.out_channels)
        self.conv1 = self.conv333(self.in_channels, self.out_channels)
        self.conv2 = self.conv333(self.out_channels, self.out_channels)
        self.pool = self.down_sample(self.out_channels, self.out_channels)

    def forward(self, input: torch.Tensor) -> _EncoderOut:
        residual = self.residual(input)
        input = self.activation(self.conv1(input))
        input = self.activation(self.group_norm(self.conv2(input)))
        input = self.dropout(input)
        before_pool = input + residual
        output = self.pool(before_pool)
        return (output, before_pool) if self.skip_out else (output, None)


class UpConvDual3D(UnetBlockConv3D):
    def __other__(self):
        merge_channels = (
            self.in_channels if self.merge_mode == "concat" else self.out_channels
        )
        self.resconv = self.merge_conv(self.out_channels, self.out_channels)
        self.conv1 = self.conv333(merge_channels, self.out_channels)
        self.conv2 = self.conv333(self.out_channels, self.out_channels)

    def forward(
        self, input: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        input = self.up_sample(input)
        input = self.merge(input, skip)
        residual = self.group_norm(self.resconv(input)) if skip is not None else input
        input = self.activation(self.group_norm(self.conv1(input)))
        input = self.activation(self.group_norm(self.conv2(input)))
        input = self.dropout(input)
        output = input + residual
        return output


class DownConvTri3D(UnetBlockConv3D):
    """
    Helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __other__(self):
        self.resconv = self.conv333(self.in_channels, self.out_channels)
        self.conv2 = self.conv333(self.out_channels, self.out_channels)
        self.conv3 = self.conv333(self.out_channels, self.out_channels)
        self.pool = self.down_sample(self.out_channels, self.out_channels)

    def forward(self, input: torch.Tensor) -> _EncoderOut:
        residual = self.group_norm(self.resconv(input))
        input = self.activation(self.group_norm(self.conv2(residual)))
        input = self.activation(self.group_norm(self.conv3(input) + residual))
        before_pool = self.dropout(input)
        output = self.pool(before_pool)
        return (output, before_pool) if self.skip_out else (output, None)


class UpConvTri3D(UnetBlockConv3D):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __other__(self):
        self.resconv = self.merge_conv(self.out_channels, self.out_channels)
        self.conv2 = self.conv333(self.out_channels, self.out_channels)
        self.conv3 = self.conv333(self.out_channels, self.out_channels)

    def forward(
        self, input: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        input = self.up_sample(input)
        input = self.merge(input, skip)
        residual = self.group_norm(self.resconv(input)) if skip is not None else input
        input = self.activation(self.group_norm(self.conv2(residual)))
        input = self.activation(self.group_norm(self.conv3(input) + residual))
        output = self.dropout(input)
        return output


class LKDownConv3D(UnetBlockConv3D):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __other__(self):
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
        input = self.activation(self.group_norm(self.conv333_1(input)))
        input = self.activation(
            input + self.conv111(input) + self.conv333_2(input) + self.conv777(input)
        )
        before_pool = self.dropout(input)
        output = self.pool(before_pool)
        return (output, before_pool) if self.skip_out else (output, None)


class PartialDownConv3D(UnetBlockConv3D):
    def __other__(self):
        self.conv = partial333(self.in_channels, self.in_channels, z_conv=self.z_conv)
        self.MaxPool = nn.MaxPool3d(2)

    def forward(self, input: torch.Tensor, mask_in: torch.Tensor | None):
        input, mask_out = self.conv(input, mask_in=mask_in)
        input = self.activation(input)
        return (
            (self.MaxPool(input), self.MaxPool(mask_out))
            if not self.last
            else (input, mask_out)
        )
