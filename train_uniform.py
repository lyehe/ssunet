"""Train Bit2Bit on preprocessed uniform-sampled SPAD data.

Usage:
    python train_uniform.py --scene blue-bathroom --device 0
    python train_uniform.py --scene train-passby --device 1
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

torch.backends.cudnn.benchmark = True

from src.ssunet.configs.data_config import DataConfig
from src.ssunet.configs.file_config import SplitParams
from src.ssunet.configs.model_config import ModelConfig
from src.ssunet.configs.train_config import LoaderConfig, TrainConfig

torch.serialization.add_safe_globals([ModelConfig])
from src.ssunet.datasets.spad import SPADUniformDataModule
from src.ssunet.models import Bit2Bit


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train Bit2Bit on uniform-sampled SPAD data")
    parser.add_argument("--config", type=str, default="config_spad.yml")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name (blue-bathroom or train-passby)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--offset", type=int, default=24,
                        help="Sampling offset (default: 24 = primary). Use -1 for all offsets.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume/finetune from")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--precision", type=str, default="32",
                        help="Precision (default: 32)")
    args = parser.parse_args()

    # Load config
    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    # Override for uniform training
    cfg["TRAIN"]["devices"] = [args.device]
    cfg["TRAIN"]["precision"] = int(args.precision) if args.precision.isdigit() else args.precision
    cfg["TRAIN"]["max_epochs"] = args.max_epochs

    # Determine data path and suffix
    data_dir = Path(f"D:/Yehe/PC/reconstruction/test/{args.scene}_step25")
    if args.offset == -1:
        # Multi-offset: pass the directory, dataset randomly picks offset each sample
        npy_path = data_dir
        suffix = "uniform_all"
        cfg["TRAIN"]["logger_name"] = f"{args.scene}_uniform_all"
    else:
        npy_path = data_dir / f"off{args.offset:02d}.npy"
        suffix = "uniform"
        cfg["TRAIN"]["logger_name"] = f"{args.scene}_uniform"

    if not npy_path.exists():
        logger.error("Data not found: %s", npy_path)
        return

    # Build configs
    data_config = DataConfig(**cfg["DATA"])
    split_params = SplitParams(**cfg["SPLIT"])
    model_config = ModelConfig(**cfg["MODEL"])
    loader_config = LoaderConfig(**cfg["LOADER"])
    train_config = TrainConfig(**cfg["TRAIN"])

    logger.info("Scene: %s, data: %s", args.scene, npy_path)
    logger.info("GPU: %d, precision: %s, epochs: %d, resume: %s",
                args.device, args.precision, args.max_epochs, args.resume)

    # Data module (accepts file or directory)
    dm = SPADUniformDataModule(
        npy_path=npy_path,
        data_config=data_config,
        split_params=split_params,
        loader_config=loader_config,
        frames_per_chunk=41,
    )

    # Model
    model = Bit2Bit(model_config)

    import os
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"C:/tc{args.device}"
    os.environ["TRITON_CACHE_DIR"] = f"C:/tr{args.device}"
    model = torch.compile(model)
    logger.info("Model compiled with torch.compile()")

    # Train
    trainer = train_config.trainer
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

    # Save
    save_path = Path(train_config.default_root_dir) / f"{args.scene}_{suffix}.ckpt"
    trainer.save_checkpoint(save_path)
    logger.info("Saved to %s", save_path)


if __name__ == "__main__":
    main()
