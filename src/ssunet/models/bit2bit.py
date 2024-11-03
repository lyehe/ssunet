"""SSUnet model."""

import pytorch_lightning as pl
import torch
import torchmetrics.image as tmi
from torch import nn
from torch.nn import init
from torch.optim import SGD, Adam, AdamW
from torch.utils.checkpoint import checkpoint

from ..configs.configs import ModelConfig
from ..constants import EPSILON, LOGGER
from ..exceptions import InvalidUpModeError
from ..losses import loss_functions
from ..modules import BLOCK, conv111

OPTIMIZER = {
    "adam": Adam,
    "sgd": SGD,
    "adamw": AdamW,
}


class Bit2Bit(pl.LightningModule):
    """Bit2Bit model."""

    def __init__(
        self,
        config: ModelConfig,
        **kwargs,
    ) -> None:
        """Initialize the Bit2Bit model.

        :param config: configuration for the model
        :param loss_function: loss function
        :param psnr_metric: peak signal-to-noise ratio metric
        :param ssim_metric: structural similarity index metric
        :param kwargs: additional arguments
        """
        super().__init__()

        # Replace pyiqa metrics with torchmetrics
        self.psnr_metric = tmi.PeakSignalNoiseRatio()
        self.ssim_metric = tmi.StructuralSimilarityIndexMeasure()

        self.config = config
        self.loss_function = loss_functions[config.loss_function]
        self.kwargs = kwargs
        self._check_conflicts()

        self.down_convs = self._down_conv_list()
        self.up_convs = self._up_conv_list()
        self.conv_final = self._final_conv()

        self.save_hyperparameters()
        self._reset_params()

    def _down_conv_list(self) -> nn.ModuleList:
        """Generate the list of down convolutional layers."""
        down_convs = []
        down_conv_function = BLOCK[self.config.block_type][0]
        init = (
            self.config.channels * self.config.signal_levels
            if self.config.sin_encoding
            else self.config.channels
        )
        for i in range(self.config.depth):
            z_conv = i < self.config.z_conv_stage
            skip_out = i >= self.config.skip_depth
            in_channels = (
                init if i == 0 else self.config.start_filts * (self.config.depth_scale ** (i - 1))
            )
            out_channels = self.config.start_filts * (self.config.depth_scale**i)
            last = True if i == self.config.depth - 1 else False
            down_conv = down_conv_function(
                int(in_channels),
                int(out_channels),
                last=last,
                skip_out=skip_out,
                z_conv=z_conv,
                dropout_p=self.config.dropout_p,
                group_norm=self.config.group_norm,
                down_mode=self.config.down_mode,
                activation=self.config.activation,
            )
            down_convs.append(down_conv)
        return nn.ModuleList(down_convs)

    def _up_conv_list(self) -> nn.ModuleList:
        """Generate the list of up convolutional layers."""
        up_convs = []
        up_conv_function = BLOCK[self.config.block_type][1]
        for i in range(self.config.depth - 1, 0, -1):
            z_conv = (i - 1) < self.config.z_conv_stage
            skip_out = i >= self.config.skip_depth
            in_channels = self.config.start_filts * (self.config.depth_scale**i)
            out_channels = self.config.start_filts * (self.config.depth_scale ** (i - 1))
            up_conv = up_conv_function(
                int(in_channels),
                int(out_channels),
                z_conv=z_conv,
                skip_out=skip_out,
                dropout_p=self.config.dropout_p,
                group_norm=self.config.group_norm,
                up_mode=self.config.up_mode,
                activation=self.config.activation,
            )
            up_convs.append(up_conv)
        return nn.ModuleList(up_convs)

    def _final_conv(self):
        """Generate the final convolutional layer."""
        return nn.Sequential(
            conv111(self.config.start_filts, self.config.channels),
        )

    @staticmethod
    def _weight_init(module: nn.Module) -> None:
        """Initialize the weights of the model."""
        if isinstance(module, nn.Conv3d):
            init.xavier_normal_(module.weight)
            init.constant_(module.bias, 0)  # type: ignore

    def _reset_params(self) -> None:
        """Reset the parameters of the model."""
        for module in self.modules():
            self._weight_init(module)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        if self.config.sin_encoding:
            scales = [
                torch.sin(input.clone() * (self.config.scale_factor ** (-i)))
                for i in range(self.config.signal_levels)
            ]
            input = torch.cat(scales, dim=1)

        encoder_outs = []
        for i, down_conv in enumerate(self.down_convs):
            if self.config.down_checkpointing:
                input, skip = checkpoint(down_conv, input, use_reentrant=False)  # type: ignore
            else:
                input, skip = down_conv(input)
            encoder_outs.append(skip) if i < self.config.depth - 1 else ...
            del skip

        for up_conv in self.up_convs:
            skip = encoder_outs.pop()
            if self.config.up_checkpointing:
                input = checkpoint(up_conv, input, skip, use_reentrant=False)  # type: ignore
            else:
                input = up_conv(input, skip)
        return self.conv_final(input)

    def configure_optimizers(self) -> dict:
        """Configure the optimizer and scheduler."""
        config = self.config.optimizer_config
        optimizer = (
            OPTIMIZER[config["name"]](self.parameters(), lr=config["lr"], fused=False)
            if config["name"] in ("adam", "adamw")
            else OPTIMIZER[config["name"]](self.parameters(), lr=config["lr"])
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["mode"],
            factor=config["factor"],
            patience=config["patience"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(
        self,
        batch: list[torch.Tensor],  # batch of training data
        batch_idx,
    ) -> torch.Tensor:
        """Training step of the model."""
        target = batch[0]
        input = batch[1]
        output = self(input)
        loss = (
            self.loss_function(output, target, (input < 1).float())
            if self.config.masked
            else self.loss_function(output, target)
        )
        self._tb_train_log(loss, output, target, batch_idx)
        return loss

    def _tb_train_log(
        self,
        loss: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Log the training step."""
        self.log("train_loss", loss)
        self._log_image(output[0], "train_image", batch_idx, frequency=100)

    def validation_step(
        self,
        batch: list[torch.Tensor],  # batch of validation data
        batch_idx: int,
    ) -> None:
        """Validation step of the model."""
        target = batch[0]
        input = batch[1]
        ground_truth = batch[2] if len(batch) == 3 else None
        output = self(input)
        loss = self.loss_function(output, target)
        self._tb_val_log(loss, output, target, ground_truth, batch_idx)

    def _tb_val_log(
        self,
        loss: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        ground_truth: torch.Tensor | None,
        batch_idx: int,
    ) -> None:
        """Log the validation step. Can be extended for logging metrics."""
        if ground_truth is not None:
            self._log_metrics(output, ground_truth, batch_idx)
        self.log("val_loss", loss)
        self._log_image(output[0], "val_image", batch_idx, frequency=10)

    def _log_metrics(
        self,
        output: torch.Tensor,
        ground_truth: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Log the metrics.

        :param output: Model output tensor
        :param ground_truth: Ground truth tensor
        :param batch_idx: Current batch index
        """
        size_z = ground_truth.shape[2]
        index_z = size_z // 2

        # Extract middle slice
        output_slice = output[:, :, index_z, ...]
        ground_truth_slice = ground_truth[:, :, index_z, ...]

        # Scale output to match ground truth mean intensity
        output_mean = torch.mean(output_slice) + EPSILON
        ground_truth_mean = torch.mean(ground_truth_slice) + EPSILON
        output_normalized = output_slice / output_mean * ground_truth_mean

        psnr = self.psnr_metric(output_normalized, ground_truth_slice)
        ssim = self.ssim_metric(output_normalized, ground_truth_slice)

        self.log("val_psnr", psnr)
        self.log("val_ssim", ssim)

    def _normalize_log_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the image for logging."""
        normalization = self.kwargs.get("log_image_normalization", "min-max")
        img = image[:, image.shape[1] // 2, ...]
        match normalization:
            case "min-max":
                return ((img - img.min()) / (img.max() - img.min()) * 255).to(torch.uint8)
            case "mean-std":
                return ((img - img.mean()) / img.std() * 255).to(torch.uint8)
            case "mean":
                return (img / img.mean() * 128).to(torch.uint8)
            case _:
                LOGGER.warning("Normalization method not recognized. Using min-max.")
                return ((img - img.min()) / (img.max() - img.min()) * 255).to(torch.uint8)

    def _log_image(
        self,
        image: torch.Tensor,
        name: str,
        batch_idx: int,
        frequency: int = 10,
    ) -> None:
        """Log the image."""
        if batch_idx % frequency == 0:
            img = self._normalize_log_image(image)
            self.logger.experiment.add_image(name, img, self.current_epoch)  # type: ignore

    def _check_conflicts(self) -> None:
        """Check for conflicts in the model configuration."""
        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if (
            self.config.up_mode == "upsample"
            and self.config.merge_mode == "add"
            or self.config.up_mode not in ["upsample", "transpose", "pixelshuffle"]
        ):
            raise InvalidUpModeError(self.config.up_mode)

    def test_step(
        self,
        batch: list[torch.Tensor],  # batch of test data
        batch_idx: int,
    ) -> None:
        """Test step of the model."""
        target = batch[0]
        input = batch[1]
        output = self(input)
        loss = self.loss_function(output, target)
        self._tb_test_log(loss, output, target, batch_idx)

    def _tb_test_log(
        self,
        loss: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Log the test step. Can be extended for logging metrics."""
        self.log("test_loss", loss)

    def on_train_end(self) -> None:
        """Save the model at the end of training."""
        self.trainer.save_checkpoint(self.trainer.default_root_dir)

    def reset_lr(self) -> None:
        """Reset the learning rate."""
        self.optimizers().param_groups[0]["lr"] = self.config.optimizer_config["lr"]  # type: ignore

    def freeze_encoder(self) -> None:
        """Freeze encoder layers for transfer learning."""
        for param in self.down_convs.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder layers."""
        for param in self.down_convs.parameters():
            param.requires_grad = True
