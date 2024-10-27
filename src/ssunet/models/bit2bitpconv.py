"""SSUnet model with partial convolution."""

import torch

from ..constants import EPSILON
from .bit2bit import Bit2Bit


class Bit2BitPConv(Bit2Bit):
    """Bit2Bit model with partial convolution."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param input_tensor: Input tensor
        :return: Output tensor
        """
        processed_input, input_mask = self._prepare_input(input_tensor)
        encoder_outputs, encoded_input = self._encode(processed_input, input_mask)
        decoded_output = self._decode(encoded_input, encoder_outputs)
        return self.conv_final(decoded_output)

    def _prepare_input(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensor and mask.

        :param input_tensor: Input tensor
        :return: Tuple of prepared input tensor and mask
        """
        if self.config.sin_encoding:
            sinusoidal_scales = [
                torch.sin(input_tensor.clone() * (self.config.scale_factor ** (-i)))
                for i in range(self.config.signal_levels)
            ]
            processed_input = torch.cat(sinusoidal_scales, dim=1)
            input_mask = (processed_input < 1).float()
            input_mask = torch.cat([input_mask for _ in range(self.config.signal_levels)], dim=1)
        else:
            processed_input = input_tensor
            input_mask = (input_tensor < 1).float()
        return processed_input, input_mask

    def _encode(
        self, input_tensor: torch.Tensor, input_mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Encoder path of the model.

        :param input_tensor: Input tensor
        :param input_mask: Input mask
        :return: List of encoder outputs and final encoder output
        """
        encoder_outputs = []
        current_mask = input_mask

        for layer_index, down_conv in enumerate(self.down_convs):
            if hasattr(down_conv, "partial"):
                if self.config.down_checkpointing:
                    (input_tensor, current_mask), skip_connection = (
                        torch.utils.checkpoint.checkpoint(
                            down_conv, input_tensor, current_mask, use_reentrant=False
                        )
                    )
                else:
                    input_tensor, skip_connection = down_conv(input_tensor, current_mask)
                    if isinstance(skip_connection, tuple):
                        current_mask = skip_connection[1]
            else:
                if self.config.down_checkpointing:
                    input_tensor, skip_connection = torch.utils.checkpoint.checkpoint(
                        down_conv, input_tensor, use_reentrant=False
                    )
                else:
                    input_tensor, skip_connection = down_conv(input_tensor)

            if layer_index < self.config.depth - 1:
                encoder_outputs.append(skip_connection)

        return encoder_outputs, input_tensor

    def _decode(
        self, input_tensor: torch.Tensor, encoder_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        """Decoder path of the model.

        :param input_tensor: Input tensor from encoder
        :param encoder_outputs: List of encoder outputs
        :return: Decoder output
        """
        for up_conv in self.up_convs:
            skip_connection = encoder_outputs.pop()
            if hasattr(up_conv, "partial"):
                if self.config.up_checkpointing:
                    input_tensor, _ = torch.utils.checkpoint.checkpoint(
                        up_conv, input_tensor, skip_connection, use_reentrant=False
                    )
                else:
                    input_tensor, _ = up_conv(input_tensor, skip_connection)
            else:
                if self.config.up_checkpointing:
                    input_tensor = torch.utils.checkpoint.checkpoint(
                        up_conv, input_tensor, skip_connection, use_reentrant=False
                    )
                else:
                    input_tensor = up_conv(input_tensor, skip_connection)

        return input_tensor

    def _normalize_log_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize the image for logging with mask consideration."""
        normalization_method = self.kwargs.get("log_image_normalization", "min-max")
        middle_slice = image[:, image.shape[1] // 2, ...]
        valid_pixel_mask = (middle_slice > EPSILON).float()

        match normalization_method:
            case "min-max":
                normalized_image = (
                    (middle_slice - middle_slice.min())
                    / (middle_slice.max() - middle_slice.min())
                    * 255
                )
                return (normalized_image * valid_pixel_mask).to(torch.uint8)
            case "mean-std":
                normalized_image = (middle_slice - middle_slice.mean()) / middle_slice.std() * 255
                return (normalized_image * valid_pixel_mask).to(torch.uint8)
            case "mean":
                normalized_image = middle_slice / middle_slice.mean() * 128
                return (normalized_image * valid_pixel_mask).to(torch.uint8)
            case _:
                normalized_image = (
                    (middle_slice - middle_slice.min())
                    / (middle_slice.max() - middle_slice.min())
                    * 255
                )
                return (normalized_image * valid_pixel_mask).to(torch.uint8)
