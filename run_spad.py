"""Training script for SPAD Single Photon Challenge.

Usage:
    # Train "attic" scene on GPU 0
    python run_spad.py --scene attic --device 0

    # Train "bathroom1" scene on GPU 1
    python run_spad.py --scene bathroom1 --device 1
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from pytorch_lightning.callbacks import Callback

torch.backends.cudnn.benchmark = True

from src.ssunet.configs.data_config import DataConfig
from src.ssunet.configs.file_config import SplitParams
from src.ssunet.configs.model_config import ModelConfig
from src.ssunet.configs.train_config import LoaderConfig, TrainConfig

# Allow checkpoint loading with weights_only=True (PyTorch 2.6+)
torch.serialization.add_safe_globals([ModelConfig])
from src.ssunet.datasets.spad import SPADDataModule, set_file_cache
from src.ssunet.models import Bit2Bit


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Train Bit2Bit on a single SPAD scene."""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train Bit2Bit on SPAD data")
    parser.add_argument("--config", type=str, default="config_spad.yml")
    parser.add_argument("--scene", type=str, default=None, help="Scene name (overrides config)")
    parser.add_argument("--scene-dir", type=str, default=None, help="Full path to scene directory")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--profile", action="store_true", help="Run 1 epoch with PyTorch profiler")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--precision", type=str, default=None, help="Override precision (e.g. 32, bf16-mixed)")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--z-size", type=int, default=None, help="Override temporal depth (z_size)")
    args = parser.parse_args()

    # Load config
    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    # Resolve scene directory
    if args.scene_dir:
        scene_dir = Path(args.scene_dir)
        scene_name = scene_dir.name
    else:
        scene_name = args.scene or cfg["PATH"]["scene"]
        scene_dir = Path(cfg["PATH"]["data_dir"]) / scene_name

    # Override device and per-scene log directory
    cfg["TRAIN"]["devices"] = [args.device]
    cfg["TRAIN"]["logger_name"] = scene_name
    if args.precision:
        cfg["TRAIN"]["precision"] = int(args.precision) if args.precision.isdigit() else args.precision
    if args.max_epochs:
        cfg["TRAIN"]["max_epochs"] = args.max_epochs
    if args.z_size:
        cfg["DATA"]["z_size"] = args.z_size

    # Build configs
    data_config = DataConfig(**cfg["DATA"])
    split_params = SplitParams(**cfg["SPLIT"])
    model_config = ModelConfig(**cfg["MODEL"])
    loader_config = LoaderConfig(**cfg["LOADER"])
    train_config = TrainConfig(**cfg["TRAIN"])

    # Configure file cache
    cache_max_gb = cfg["PATH"].get("cache_max_gb")
    set_file_cache(max_gb=cache_max_gb)

    logger.info("Scene: %s", scene_name)
    logger.info("GPU: %d", args.device)

    # Data module (handles GPU bit unpacking + binomial split)
    dm = SPADDataModule(
        scene_dir=scene_dir,
        data_config=data_config,
        split_params=split_params,
        loader_config=loader_config,
    )

    # Model
    model = Bit2Bit(model_config)

    # torch.compile fuses kernels and optimizes layouts automatically
    if not args.profile:
        import os
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"C:/tc{args.device}"
        os.environ["TRITON_CACHE_DIR"] = f"C:/tr{args.device}"
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile()")

    if args.profile:
        # Profile mode: run a few steps with PyTorch profiler
        logger.info("Profiling mode: 1 epoch with step-level tracing")
        cfg["TRAIN"]["max_epochs"] = 1
        cfg["TRAIN"]["profiler"] = None
        train_config = TrainConfig(**cfg["TRAIN"])

        profile_dir = Path(train_config.default_root_dir) / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        class ProfilerCallback(Callback):
            """Step the PyTorch profiler each training step."""

            def on_train_start(self, trainer, pl_module):
                profiler.start()

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                profiler.step()

            def on_train_end(self, trainer, pl_module):
                profiler.stop()
                logger.info("Profile saved to %s", profile_dir)
                logger.info(
                    profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                )

        trainer = train_config.trainer
        trainer.callbacks.append(ProfilerCallback())
        trainer.fit(model, datamodule=dm)
    else:
        # Normal training
        trainer = train_config.trainer
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

        # Save final checkpoint
        save_path = Path(train_config.default_root_dir) / f"{scene_name}.ckpt"
        trainer.save_checkpoint(save_path)
        logger.info("Saved to %s", save_path)


if __name__ == "__main__":
    main()
