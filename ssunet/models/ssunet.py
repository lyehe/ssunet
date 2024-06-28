from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import logging
import pyiqa

from torch.nn import init
from torch.utils.checkpoint import checkpoint

from ssunet.loss import loss_functions
from ssunet.modules import (
    DownConvDual3D,
    UpConvDual3D,
    DownConvTri3D,
    UpConvTri3D,
    LKDownConv3D,
    conv111,
)
from lightning.pytorch.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)

EPSILON = 1e-8

DEFAULT_OPTIMIZER_CONFIG = {
    "name": "adam",  # optimizer name
    "lr": 2e-5,  # learning rate
    "mode": "min",  # mode for ReduceLROnPlateau
    "factor": 0.5,  # factor for ReduceLROnPlateau
    "patience": 5,  # patience for ReduceLROnPlateau
}

BLOCK = {
    "dual": (DownConvDual3D, UpConvDual3D),
    "tri": (DownConvTri3D, UpConvTri3D),
    "LK": (LKDownConv3D, UpConvTri3D),
}

OPTIMIZER = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
}


@dataclass
class ModelConfig:
    channels: int = 1
    depth: int = 4
    start_filts: int = 24
    depth_scale: int = 2
    depth_scale_stop: int = 10
    z_conv_stage: int = 5
    group_norm: int = 0
    skip_depth: int = 0
    dropout_p: float = 0.0
    scale_factor: float = 10.0
    sin_encoding: bool = True
    signal_levels: int = 10
    masked: bool = True
    down_checkpointing: bool = False
    up_checkpointing: bool = False
    loss_function: str = "photon"
    up_mode: str = "transpose"
    merge_mode: str = "concat"
    down_mode: str = "maxpool"
    activation: str = "relu"
    block_type: str = "dual"
    note: str = ""

    @property
    def name(self) -> str:
        name_str = [
            f"l={self.signal_levels}",
            f"d={self.depth}",
            f"sf={self.start_filts}",
            f"ds={self.depth_scale}at{self.depth_scale_stop}",
            f"f={self.scale_factor}",
            f"z={self.z_conv_stage}",
            f"g={self.group_norm}",
            f"sd={self.skip_depth}",
            f"b={self.block_type}",
            f"a={self.activation}",
        ]
        return "_".join(name_str)


class SSUnet(pl.LightningModule):
    def __init__(
        self,
        config: ModelConfig,
        optimizer_config: dict = DEFAULT_OPTIMIZER_CONFIG,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.optimizer_config = optimizer_config
        self.loss_function = loss_functions[config.loss_function]

        self._check_conflicts()

        self.down_convs = self._down_conv_list()
        self.up_convs = self._up_conv_list()
        self.conv_final = self._final_conv()

        self._psnr_metric = pyiqa.create_metric("psnr", device=self.device)
        self._ssim_metric = pyiqa.create_metric("ssim", channels=1, device=self.device)

        self.save_hyperparameters()
        self._reset_params()

    def _down_conv_list(self) -> nn.ModuleList:
        down_convs = []
        DownConv = BLOCK[self.config.block_type][0]
        init = (
            self.config.channels * self.config.signal_levels
            if self.config.sin_encoding
            else self.config.channels
        )
        for i in range(self.config.depth):
            z_conv = i < self.config.z_conv_stage
            skip_out = i >= self.config.skip_depth
            in_channels = (
                init
                if i == 0
                else self.config.start_filts * (self.config.depth_scale ** (i - 1))
            )
            out_channels = self.config.start_filts * (self.config.depth_scale**i)
            last = True if i == self.config.depth - 1 else False
            down_conv = DownConv(
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
        up_convs = []
        UpConv = BLOCK[self.config.block_type][1]
        for i in range(self.config.depth - 1, 0, -1):
            z_conv = (i - 1) < self.config.z_conv_stage
            skip_out = i >= self.config.skip_depth
            in_channels = self.config.start_filts * (self.config.depth_scale**i)
            out_channels = self.config.start_filts * (
                self.config.depth_scale ** (i - 1)
            )
            up_conv = UpConv(
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
        return nn.Sequential(
            conv111(self.config.start_filts, self.config.channels),
        )

    @staticmethod
    def _weight_init(module: nn.Module):
        if isinstance(module, nn.Conv3d):
            init.xavier_normal_(module.weight)
            init.constant_(module.bias, 0)  # type: ignore

    def _reset_params(self):
        for module in self.modules():
            self._weight_init(module)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
            encoder_outs.append(skip) if i < self.depth - 1 else ...
            del skip

        for i, up_conv in enumerate(self.up_convs):
            skip = encoder_outs.pop()
            if self.config.up_checkpointing:
                input = checkpoint(up_conv, input, skip, use_reentrant=False)  # type: ignore
            else:
                input = up_conv(input, skip)
        return self.conv_final(input)

    def configure_optimizers(self) -> dict:
        config = self.optimizer_config
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
        input = batch[1]
        target = batch[0]
        output = self(input)
        loss = (
            self.loss_function(output, target, (input < 1).float())
            if self.masked
            else self.loss_function(output, target)
        )
        self.tb_train_log(loss, output, target, batch_idx)
        return loss

    def tb_train_log(
        self,
        loss: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        batch_idx: int,
    ):
        self.log("train_loss", loss)
        self._log_image(output[0], "train_image", batch_idx, frequency=100)

    def validation_step(
        self,
        batch: list[torch.Tensor],  # batch of validation data
        batch_idx: int,
    ) -> None:
        input = batch[1]
        target = batch[0]
        ground_truth = batch[2] if len(batch) == 3 else None
        output = self(input)
        loss = self.loss_function(output, target)
        self.tb_val_log(loss, output, target, ground_truth, batch_idx)

    def tb_val_log(
        self,
        loss: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        ground_truth: torch.Tensor | None,
        batch_idx: int,
    ):
        if ground_truth is not None:
            self._log_metrics(output, ground_truth, batch_idx)
        self.log("val_loss", loss)
        self._log_image(output[0], "val_image", batch_idx, frequency=10)

    def test_step(
        self,
        batch: list[torch.Tensor],  # batch of test data
        batch_idx: int,
    ) -> None:
        input = batch[1]
        target = batch[0]
        output = self(input)
        loss = self.loss_function(output, target)
        self.tb_test_log(loss, output, target, batch_idx)

    def tb_test_log(
        self,
        loss: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        batch_idx: int,
    ):
        self.log("test_loss", loss)

    def _log_image(
        self,
        image: torch.Tensor,
        name: str,
        batch_idx: int,
        frequency: int = 10,
        normalization: str = "min-max",
    ) -> None:
        if normalization not in ("min-max", "mean-std", "mean"):
            normalization = "min-max"
            logger.warning(
                f"Normalization method not recognized. Using {normalization}."
            )
        if batch_idx % frequency == 0:
            image_shape = image.shape
            img = image[:, image_shape[1] // 2, ...]
            match normalization:
                case "min-max":
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).to(
                        torch.uint8
                    )
                    self.logger.experiment.add_image(name, img, self.current_epoch)  # type: ignore
                case "mean-std":
                    img = ((img - img.mean()) / img.std() * 255).to(torch.uint8)
                    self.logger.experiment.add_image(name, img, self.current_epoch)  # type: ignore
                case "mean":
                    img = (img / img.mean() * 128).to(torch.uint8)
                    self.logger.experiment.add_image(name, img, self.current_epoch)  # type: ignore

    def _log_metrics(
        self,
        output: torch.Tensor,
        ground_truth: torch.Tensor,
        batch_idx: int,
    ) -> None:
        size_z = ground_truth.shape[2]
        index_z = size_z // 2

        normalized_output = output[:, :, index_z, ...]
        ground_truth = ground_truth[:, :, index_z, ...]
        output_mean = torch.mean(normalized_output) + EPSILON
        ground_truth_mean = torch.mean(ground_truth) + EPSILON
        normalized_output = normalized_output / output_mean * ground_truth_mean

        psnr = self._psnr_metric(normalized_output, ground_truth)
        ssim = self._ssim_metric(normalized_output, ground_truth)
        self.log("val_psnr", psnr)
        self.log("val_ssim", ssim)

    def _check_conflicts(self):
        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.config.up_mode == "upsample" and self.config.merge_mode == "add":
            raise ValueError(
                'up_mode "upsample" is incompatible '
                'with merge_mode "add" at the moment '
                "because it doesn't make sense to use "
                "nearest neighbour to reduce "
                "depth channels (by half)."
            )
