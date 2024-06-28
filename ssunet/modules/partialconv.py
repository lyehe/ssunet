import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

EPSILON = 1e-8  # for mixed precision training, change 1e-8 to 1e-6
logger = logging.getLogger(__name__)


class PartialConv3d(nn.Conv3d):
    """3D Partial Convolutional Layer for Image Inpainting from NVIDIA.
    https://arxiv.org/abs/1811.11718 for padding in this application.
    """

    def __init__(
        self, *args, multi_channel: bool = False, return_mask: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.return_mask = return_mask

        self.weight_maskUpdater = (
            torch.ones(self.out_channels, self.in_channels, *self.kernel_size)
            if multi_channel
            else torch.ones(1, 1, *self.kernel_size)
        )
        self.slide_winsize = torch.prod(torch.tensor(self.weight_maskUpdater.shape[1:]))
        self.last_size = (None, None, None, None, None)
        self.update_mask = torch.tensor([])
        self.mask_ratio = torch.tensor([])

    def forward(self, input: torch.Tensor, mask_in: torch.Tensor | None = None):
        if len(input.shape) != 5:
            logger.error(f"Pconv3D: Input must be 5D , but got {len(input.shape)}")
            raise ValueError(f"Pconv3D: Input must be 5D , but got {len(input.shape)}")

        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                # if mask is not provided, create a mask
                mask = (
                    (
                        torch.ones(input.data.shape).to(input)
                        if self.multi_channel
                        else torch.ones(1, 1, *input.data.shape[2:]).to(input)
                    )
                    if mask_in is None
                    else mask_in
                )

                self.update_mask = F.conv3d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + EPSILON)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super().forward(
            torch.mul(input, mask_in) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        return output, self.update_mask if self.return_mask else output


class PartialConv2d(nn.Conv2d):
    def __init__(
        self, *args, multi_channel: bool = False, return_mask: bool = False, **kwargs
    ):
        self.multi_channel = multi_channel
        self.return_mask = return_mask

        super().__init__(*args, **kwargs)

        self.weight_maskUpdater = (
            torch.ones(self.out_channels, self.in_channels, *self.kernel_size)
            if multi_channel
            else torch.ones(1, 1, *self.kernel_size)
        )

        self.slide_winsize = torch.prod(torch.tensor(self.weight_maskUpdater.shape[1:]))
        self.last_size = (None, None, None, None)
        self.update_mask = torch.tensor([])
        self.mask_ratio = torch.tensor([])

    def forward(self, input: torch.Tensor, mask_in: torch.Tensor | None = None):
        if len(input.shape) != 4:
            logger.error(f"Pconv2D: Input must be 4D , but got {len(input.shape)}")
            raise ValueError(f"Pconv2D: Input must be 4D , but got {len(input.shape)}")

        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                mask = (
                    (
                        torch.ones(input.data.shape).to(input)
                        if self.multi_channel
                        else torch.ones(1, 1, *input.data.shape[2:]).to(input)
                    )
                    if mask_in is None
                    else mask_in
                )

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + EPSILON)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super().forward(
            torch.mul(input, mask) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        return output, self.update_mask if self.return_mask else output
