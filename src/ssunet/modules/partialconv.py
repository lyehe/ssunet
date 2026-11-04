"""Partial convolutions with binary validity masks and padding correction."""

import math

import torch
import torch.nn.functional as tnf
from torch import nn

from ..exceptions import InvalidInputShapeError


class _PartialConv:
    """Shared mask bookkeeping for the two convolution dimensions."""

    def _initialize_masks(self, dimensions: int) -> None:
        self._dimensions = dimensions
        mask_channels = self.in_channels // self.groups if self.multi_channel else 1
        mask_outputs = self.out_channels if self.multi_channel else 1
        updater = torch.ones(mask_outputs, mask_channels, *self.kernel_size)
        # Derived masks are rebuilt on demand and must not change checkpoint keys.
        self.register_buffer("weight_maskUpdater", updater, persistent=False)
        self.register_buffer("update_mask", torch.empty(0), persistent=False)
        self.register_buffer("mask_ratio", torch.empty(0), persistent=False)
        self.slide_winsize = math.prod(updater.shape[1:])
        self._mask_cache_key = None
        self._last_mask_was_explicit = False

    def forward(
        self, input: torch.Tensor, mask_in: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convolve valid inputs and correct for missing samples at the boundary."""
        expected_dimensions = self._dimensions + 2
        if input.ndim != expected_dimensions:
            raise InvalidInputShapeError(expected_dimensions, input.shape)

        cache_key = (tuple(input.shape), input.device, input.dtype)
        if mask_in is not None or self._last_mask_was_explicit or self._mask_cache_key != cache_key:
            with torch.no_grad(), torch.autocast(input.device.type, enabled=False):
                mask_shape = (
                    tuple(input.shape)
                    if self.multi_channel
                    else (input.shape[0], 1, *input.shape[2:])
                )
                mask_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
                if mask_in is None:
                    mask = torch.ones(mask_shape, device=input.device, dtype=mask_dtype)
                else:
                    mask = torch.broadcast_to(
                        mask_in.to(device=input.device, dtype=mask_dtype), mask_shape
                    )
                convolution = tnf.conv3d if self._dimensions == 3 else tnf.conv2d
                counts = convolution(
                    mask,
                    self.weight_maskUpdater.to(device=input.device, dtype=mask_dtype),
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups if self.multi_channel else 1,
                )
                self.update_mask = (counts > 0).to(dtype=mask_dtype)
                self.mask_ratio = self.slide_winsize / counts.clamp_min(1)
                self.mask_ratio = self.mask_ratio * self.update_mask
            self._mask_cache_key = cache_key
            self._last_mask_was_explicit = mask_in is not None

        masked_input = input if mask_in is None else input * mask_in.to(input)
        raw_out = super().forward(masked_input)
        ratio = self.mask_ratio.to(raw_out)
        valid = self.update_mask.to(raw_out)
        if self.bias is None:
            output = raw_out * ratio
        else:
            bias = self.bias.to(raw_out).view(1, self.out_channels, *([1] * self._dimensions))
            output = ((raw_out - bias) * ratio + bias) * valid

        if self.return_mask:
            return output, valid
        return output


class PartialConv3d(_PartialConv, nn.Conv3d):
    """3D partial convolution using NVIDIA's padding-correction formulation."""

    def __init__(
        self, *args, multi_channel: bool = True, return_mask: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self._initialize_masks(3)


class PartialConv2d(_PartialConv, nn.Conv2d):
    """2D partial convolution using NVIDIA's padding-correction formulation."""

    def __init__(
        self, *args, multi_channel: bool = True, return_mask: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self._initialize_masks(2)
