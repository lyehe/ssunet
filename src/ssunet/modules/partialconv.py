"""Partial convolutions from torch-convnd, with one mask channel per input channel."""

import torch_convnd


class PartialConv3d(torch_convnd.PartialConv3d):
    """NVIDIA's 3D partial convolution, computed by torch-convnd.

    Without a mask, zero-padded border windows are rescaled by their valid fraction.
    ``forward(input, mask_in)`` also takes a validity mask with one channel or, by
    default, one per input channel; ``return_mask`` adds the update mask to the output.
    """

    def __init__(
        self, *args, multi_channel: bool = True, return_mask: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, multi_channel=multi_channel, return_mask=return_mask, **kwargs)


class PartialConv2d(torch_convnd.PartialConv2d):
    """NVIDIA's 2D partial convolution, computed by torch-convnd; see PartialConv3d."""

    def __init__(
        self, *args, multi_channel: bool = True, return_mask: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, multi_channel=multi_channel, return_mask=return_mask, **kwargs)
