"""Train Bit2Bit on binned SPAD data (sum of N binary frames per pixel).

All files in a scene form one continuous stream of binary frames.
Each sample picks a random starting offset (any frame, not just multiples of
bin_factor) and loads z_size * bin_factor consecutive binary frames from there.
Those are binned into z_size count frames, then trained with binomial splitting
and photon loss.

Usage:
    python train_binned.py --scene-dir D:/Yehe/PC/reconstruction/test/basement --device 0
    python train_binned.py --scene-dir D:/Yehe/PC/reconstruction/test/bedroom-interior --device 1
"""

import argparse
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from numpy.random import choice, rand, randint
from torch.utils.data import Dataset

from src.ssunet.configs.data_config import DataConfig
from src.ssunet.configs.file_config import SplitParams
from src.ssunet.configs.model_config import ModelConfig
from src.ssunet.configs.train_config import LoaderConfig, TrainConfig
from src.ssunet.constants import EPSILON, LOGGER
from src.ssunet.datasets.spad import SPADScene, gpu_unpackbits, set_file_cache
from src.ssunet.models import Bit2Bit

torch.backends.cudnn.benchmark = True
torch.serialization.add_safe_globals([ModelConfig])


def setup_logging() -> None:
    """Log INFO messages with timestamps to the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class SPADBinnedDataset(Dataset):
    """Dataset that loads z_size * bin_factor binary frames per sample.

    All files are treated as one continuous sequence. Binning happens on GPU.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        config: DataConfig,
        bin_factor: int = 8,
    ) -> None:
        self.config = config
        self.bin_factor = bin_factor
        self.scene = SPADScene(Path(scene_dir))
        self.z_raw = config.z_size * bin_factor  # binary frames to load per sample

        LOGGER.info(
            "Binned dataset '%s': %d files (%d frames), bin_factor=%d, z_raw=%d",
            self.scene.name,
            self.scene.n_files,
            self.scene.total_frames,
            bin_factor,
            self.z_raw,
        )

    def __len__(self) -> int:
        """Samples per epoch: virtual_size when set, otherwise every start frame."""
        if self.config.virtual_size > 0:
            return self.config.virtual_size
        return self.scene.total_frames - self.z_raw

    def __getitem__(self, index: int) -> torch.Tensor:
        """Bit-packed z_size * bin_factor frames from a random start, with a random H crop."""
        t_max = self.scene.total_frames - self.z_raw
        t_start = randint(0, max(1, t_max))

        h_max = self.scene.height - self.config.xy_size
        h_start = randint(0, max(1, h_max)) if h_max > 0 else 0

        packed = self.scene.get_packed_patch(
            t_start,
            self.z_raw,
            h_start,
            self.config.xy_size,
        )
        return torch.from_numpy(packed)


class SPADBinnedDataModule(pl.LightningDataModule):
    """DataModule that bins binary SPAD frames on GPU and applies binomial splitting.

    Pipeline:
    1. Dataset loads z_size * bin_factor bit-packed binary frames
    2. GPU: unpack bits, random W crop, augment
    3. GPU: Bernoulli-split each binary frame, then sum groups of bin_factor
       → binned target and noise (mathematically = Binomial(count, p) split)
    4. Model receives [target, noise] with values in 0..bin_factor
    """

    def __init__(
        self,
        scene_dir: str | Path,
        data_config: DataConfig,
        split_params: SplitParams,
        loader_config: LoaderConfig,
        bin_factor: int = 8,
    ) -> None:
        super().__init__()
        self.scene_dir = scene_dir
        self.data_config = data_config
        self.split_params = split_params
        self.loader_config = loader_config
        self.bin_factor = bin_factor

    def setup(self, stage: str | None = None) -> None:
        """Build the binned training dataset for fitting."""
        if stage in ("fit", None):
            self.train_dataset = SPADBinnedDataset(
                self.scene_dir,
                self.data_config,
                bin_factor=self.bin_factor,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Loader over the binned training dataset."""
        return self.loader_config.loader(self.train_dataset)

    def val_dataloader(self) -> list[torch.utils.data.DataLoader]:
        """No validation loaders for binned data."""
        return []

    def on_after_batch_transfer(self, batch: torch.Tensor, dataloader_idx: int) -> list:
        """Unpack, bin, and split on GPU.

        Input: (B, Z*bin, H, W_packed, C) uint8
        Output: [target, noise] each (B, C, Z, H, W) float32
        """
        packed = batch  # (B, Z*bin, H, W_packed, C)

        # Unpack bits along W
        unpacked = gpu_unpackbits(packed, dim=3)  # (B, Z*bin, H, 800, C)

        # Random W crop
        w_size = self.data_config.xy_size
        w_max = unpacked.shape[3] - w_size
        if w_max > 0:
            w_start = torch.randint(0, w_max + 1, (1,), device=unpacked.device).item()
            unpacked = unpacked[:, :, :, w_start : w_start + w_size, :]

        # Permute to (B, C, Z*bin, H, W)
        data = unpacked.permute(0, 4, 1, 2, 3).contiguous()

        # GPU augmentation
        if dataloader_idx == 0 and self.data_config.augments:
            if torch.rand(1).item() > 0.5:
                data = data.transpose(-1, -2)
            if torch.rand(1).item() > 0.5:
                data = data.flip(-1)
            if torch.rand(1).item() > 0.5:
                data = data.flip(-2)

        batch_size, channels, z_binned, height, width = data.shape
        depth = z_binned // self.bin_factor

        # Bernoulli split at binary level, then bin.
        # Memory-efficient: compute noise = sum(data * mask) per bin group,
        # then target = binned_data - noise (avoids holding 3 full binary tensors).
        p = self._sample_p()
        mask = torch.rand_like(data) < p
        noise_binary = data * mask.float()
        del mask

        # Bin both: reshape and sum groups of bin_factor
        binned_data = data.reshape(batch_size, channels, depth, self.bin_factor, height, width).sum(
            dim=3
        )
        del data
        noise = noise_binary.reshape(
            batch_size, channels, depth, self.bin_factor, height, width
        ).sum(dim=3)
        del noise_binary
        target = binned_data - noise

        if self.split_params.normalize_target:
            target = target / (target.mean() + EPSILON)

        return [target, noise]

    def _sample_p(self) -> float:
        method = self.split_params.method
        min_p = min(self.split_params.min_p, self.split_params.max_p)
        max_p = max(self.split_params.min_p, self.split_params.max_p)

        if method == "signal":
            return float(rand() * (max_p - min_p) + min_p)
        if method == "fixed":
            return float(self.split_params.min_p)
        if method == "list" and self.split_params.p_list:
            return float(choice(self.split_params.p_list))
        return float(rand() * (max_p - min_p) + min_p)


def main() -> None:
    """Train Bit2Bit on binned SPAD data."""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train Bit2Bit on binned SPAD data")
    parser.add_argument("--config", type=str, default="config_spad.yml")
    parser.add_argument("--scene-dir", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bin-factor", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--z-size", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None)
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    scene_dir = Path(args.scene_dir)
    scene_name = scene_dir.name

    cfg["TRAIN"]["devices"] = [args.device]
    cfg["TRAIN"]["logger_name"] = f"{scene_name}-bin{args.bin_factor}"
    if args.precision:
        cfg["TRAIN"]["precision"] = (
            int(args.precision) if args.precision.isdigit() else args.precision
        )
    if args.max_epochs:
        cfg["TRAIN"]["max_epochs"] = args.max_epochs
    if args.z_size:
        cfg["DATA"]["z_size"] = args.z_size

    # Ensure photon loss
    cfg["MODEL"]["loss_function"] = "photon"

    # Reduce workers for binned loading (256 frames/sample is heavy)
    cfg["LOADER"]["num_workers"] = 2
    cfg["LOADER"]["persistent_workers"] = False

    data_config = DataConfig(**cfg["DATA"])
    split_params = SplitParams(**cfg["SPLIT"])
    model_config = ModelConfig(**cfg["MODEL"])
    loader_config = LoaderConfig(**cfg["LOADER"])
    train_config = TrainConfig(**cfg["TRAIN"])

    cache_max_gb = cfg["PATH"].get("cache_max_gb")
    set_file_cache(max_gb=cache_max_gb)

    logger.info("Scene: %s, bin_factor=%d", scene_name, args.bin_factor)
    logger.info(
        "z_size=%d (loads %d binary frames per sample)",
        data_config.z_size,
        data_config.z_size * args.bin_factor,
    )
    logger.info("GPU: %d", args.device)

    dm = SPADBinnedDataModule(
        scene_dir=scene_dir,
        data_config=data_config,
        split_params=split_params,
        loader_config=loader_config,
        bin_factor=args.bin_factor,
    )

    model = Bit2Bit(model_config)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"C:/tc{args.device}"
    os.environ["TRITON_CACHE_DIR"] = f"C:/tr{args.device}"
    model = torch.compile(model)
    logger.info("Model compiled with torch.compile()")

    trainer = train_config.trainer
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

    save_dir = Path(train_config.default_root_dir)
    save_path = save_dir / f"{scene_name}-bin{args.bin_factor}.ckpt"
    trainer.save_checkpoint(save_path)
    logger.info("Saved to %s", save_path)


if __name__ == "__main__":
    main()
