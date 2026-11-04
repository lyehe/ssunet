"""SSUnet model."""

import pytorch_lightning as pl
import torch
import torchmetrics.image as tmi
from torch import nn
from torch.nn import init
from torch.optim import SGD, Adam, AdamW
from torch.utils.checkpoint import checkpoint

from ..checkpoints import checkpoint_config_imports
from ..configs.configs import ModelConfig
from ..constants import EPSILON, LOGGER
from ..distributions import logits_to_probabilities
from ..exceptions import InvalidUpModeError
from ..losses import loss_functions, photon_nll
from ..modules import BLOCK, conv111
from ..modules.partialconv import PartialConv3d

OPTIMIZER = {
    "adam": Adam,
    "sgd": SGD,
    "adamw": AdamW,
}


class Bit2Bit(pl.LightningModule):
    """Bit2Bit model."""

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        """Load installed and repository-created checkpoints with their original architecture."""
        with checkpoint_config_imports(ModelConfig):
            return super().load_from_checkpoint(checkpoint_path, *args, **kwargs)

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

        if isinstance(config, dict):
            config = ModelConfig(**config)

        # Replace pyiqa metrics with torchmetrics
        self.psnr_metric = tmi.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = tmi.StructuralSimilarityIndexMeasure(data_range=1.0)

        self.config = config
        self.architecture_version = getattr(config, "architecture_version", 1)
        if self.architecture_version not in (1, 2):
            message = "Unsupported architecture_version; expected 1 or 2"
            raise ValueError(message)
        self._architecture_options = {
            "partial_conv": config.partial_conv if self.architecture_version == 2 else False,
            "merge_mode": config.merge_mode if self.architecture_version == 2 else "concat",
        }
        if self.architecture_version == 1 and (
            config.partial_conv or config.merge_mode != "concat"
        ):
            LOGGER.warning(
                "Architecture version 1 preserves ordinary convolutions and concatenation. "
                "Use architecture_version=2 to honor partial_conv and merge_mode for a new model."
            )
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
                **(
                    {"partial_conv": self.config.partial_conv}
                    if self.architecture_version == 2
                    else {}
                ),
            )
            down_convs.append(down_conv)
        return nn.ModuleList(down_convs)

    def _up_conv_list(self) -> nn.ModuleList:
        """Generate the list of up convolutional layers."""
        up_convs = []
        up_conv_function = BLOCK[self.config.block_type][1]
        for i in range(self.config.depth - 1, 0, -1):
            z_conv = (i - 1) < self.config.z_conv_stage
            skip_out = (
                (i - 1) >= self.config.skip_depth
                if self.architecture_version == 2
                else i >= self.config.skip_depth
            )
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
                **(
                    {"partial_conv": self.config.partial_conv, "merge_mode": self.config.merge_mode}
                    if self.architecture_version == 2
                    else {}
                ),
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
        # torch-convnd's partial layers are not nn.Conv3d subclasses.
        if isinstance(module, nn.Conv3d | PartialConv3d):
            init.xavier_normal_(module.weight)
            init.constant_(module.bias, 0)  # type: ignore

    def _reset_params(self) -> None:
        """Reset the parameters of the model."""
        for module in self.modules():
            self._weight_init(module)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return unnormalized photon-location logits for the input volume."""
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

    def predict_distribution(self, input: torch.Tensor) -> torch.Tensor:
        """Return photon probabilities over the same axes as the configured loss."""
        dimensions = (-1, -2, -4) if self.config.loss_function == "photon_2d" else None
        return logits_to_probabilities(self(input), dim=dimensions)

    def _batch_loss(self, output: torch.Tensor, target: torch.Tensor, input: torch.Tensor):
        """Apply the same binary-event masking convention in each training stage."""
        mask = (input < 1).float() if self.config.masked else None
        return self.loss_function(output, target, mask)

    def _photon_score(self, output: torch.Tensor, target: torch.Tensor, input: torch.Tensor):
        """Score observed target events without requiring an intensity reference."""
        mask = (input < 1).float() if self.config.masked else None
        dimensions = (-1, -2, -4) if self.config.loss_function == "photon_2d" else None
        return photon_nll(output, target, mask, dim=dimensions)

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
            "monitor": config.get("monitor", "train_loss"),
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
        loss = self._batch_loss(output, target, input)
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
        self.log("train_loss", loss, on_step=True, on_epoch=True)
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
        loss = self._batch_loss(output, target, input)
        score = self._photon_score(output, target, input)
        self._tb_val_log(loss, output, target, ground_truth, batch_idx, photon_score=score)

    def _tb_val_log(
        self,
        loss: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        ground_truth: torch.Tensor | None,
        batch_idx: int,
        photon_score: torch.Tensor | None = None,
    ) -> None:
        """Log the validation step. Can be extended for logging metrics."""
        if ground_truth is not None and getattr(self.config, "log_reference_metrics", False):
            self._log_metrics(output, ground_truth, batch_idx)
        self.log("val_loss", loss)
        if photon_score is not None:
            name = "val_masked_photon_nll" if self.config.masked else "val_photon_nll"
            self.log(name, photon_score, on_step=False, on_epoch=True)
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
        # Reference comparisons are opt-in diagnostics in matching intensity units.
        output_slice = logits_to_probabilities(output[:, :, output.shape[2] // 2])
        ground_truth_slice = ground_truth[:, :, ground_truth.shape[2] // 2]
        dimensions = (-1, -2, -3)
        output_mean = output_slice.mean(dim=dimensions, keepdim=True).clamp_min(
            torch.finfo(output_slice.dtype).tiny
        )
        ground_truth_mean = ground_truth_slice.mean(dim=dimensions, keepdim=True)
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
        center = image[:, image.shape[1] // 2, ...]
        img = logits_to_probabilities(center.unsqueeze(0))[0]
        match normalization:
            case "min-max":
                return ((img - img.min()) / (img.max() - img.min()).clamp_min(EPSILON) * 255).to(
                    torch.uint8
                )
            case "mean-std":
                return ((img - img.mean()) / img.std(unbiased=False).clamp_min(EPSILON) * 255).to(
                    torch.uint8
                )
            case "mean":
                return (img / img.mean().clamp_min(EPSILON) * 128).to(torch.uint8)
            case _:
                LOGGER.warning("Normalization method not recognized. Using min-max.")
                return ((img - img.min()) / (img.max() - img.min()).clamp_min(EPSILON) * 255).to(
                    torch.uint8
                )

    def _log_image(
        self,
        image: torch.Tensor,
        name: str,
        batch_idx: int,
        frequency: int = 10,
    ) -> None:
        """Log the image."""
        experiment = getattr(self.logger, "experiment", None)
        if batch_idx % frequency == 0 and callable(getattr(experiment, "add_image", None)):
            img = self._normalize_log_image(image)
            experiment.add_image(name, img, self.current_epoch)

    def _check_conflicts(self) -> None:
        """Check for conflicts in the model configuration."""
        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if (
            self.config.up_mode == "upsample" and self.config.merge_mode == "add"
        ) or self.config.up_mode not in ["upsample", "transpose", "pixelshuffle"]:
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
        loss = self._batch_loss(output, target, input)
        self._tb_test_log(loss, output, target, batch_idx)
        name = "test_masked_photon_nll" if self.config.masked else "test_photon_nll"
        self.log(name, self._photon_score(output, target, input), on_step=False, on_epoch=True)

    def _tb_test_log(
        self,
        loss: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Log the test step. Can be extended for logging metrics."""
        self.log("test_loss", loss)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Record the effective architecture independently of historical config flags."""
        checkpoint["ssunet_architecture_version"] = self.architecture_version
        checkpoint["ssunet_architecture_options"] = self._architecture_options.copy()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Reject loading weights under a different effective architecture."""
        saved_config = checkpoint.get("hyper_parameters", {}).get("config", {})
        config_version = (
            saved_config.get("architecture_version", 1)
            if isinstance(saved_config, dict)
            else getattr(saved_config, "architecture_version", 1)
        )
        saved_version = checkpoint.get("ssunet_architecture_version", config_version)
        if saved_version != self.architecture_version:
            message = (
                f"Checkpoint architecture version {saved_version} does not match model "
                f"version {self.architecture_version}; load with the checkpoint's architecture."
            )
            raise ValueError(message)
        if saved_version == 2:
            saved_options = checkpoint.get("ssunet_architecture_options")
            if saved_options is None:
                get_option = (
                    saved_config.get
                    if isinstance(saved_config, dict)
                    else lambda key: getattr(saved_config, key)
                )
                saved_options = {
                    "partial_conv": get_option("partial_conv"),
                    "merge_mode": get_option("merge_mode"),
                }
            if saved_options != self._architecture_options:
                message = (
                    "Checkpoint architecture options do not match the requested partial_conv "
                    "and merge_mode; load with the checkpoint's configured layers."
                )
                raise ValueError(message)

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
