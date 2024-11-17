import argparse
import logging

import numpy as np
from tifffile import imwrite

from src.ssunet.configs import MasterConfig
from src.ssunet.constants import DEFAULT_CONFIG_PATH
from src.ssunet.datasets import BinomDataset, ValidationDataset
from src.ssunet.models import Bit2Bit
from src.tools.gpuinference import gpu_patch_inference
from src.tools.tools import clear_vram


def setup_logging():
    """Configure logging with appropriate format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Train and run inference with the Bit2Bit model")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the configuration file (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        setup_logging()
        logger = logging.getLogger(__name__)

        args = parse_arguments()
        logger.info(f"Using configuration file: {args.config}")

        # Load configuration
        config = MasterConfig.from_config(args.config)
        config.copy_config(args.config)

        # Load datasets
        logger.info("Loading datasets...")
        data = config.path_config.load_data_only()
        validation_data = config.path_config.load_reference_and_ground_truth()

        # Configure datasets
        data_config = config.data_config
        validation_config = data_config.validation_config

        # Create datasets and loaders
        logger.info("Preparing data loaders...")
        training_data = BinomDataset(data, data_config, config.split_params)
        validation_data = ValidationDataset(validation_data, validation_config)

        training_loader = config.loader_config.loader(training_data)
        validation_loader = config.loader_config.loader(validation_data)

        # Initialize model
        logger.info("Initializing model...")
        model = Bit2Bit(config.model_config)

        batch_shape = tuple(next(iter(training_loader))[1].shape)
        logger.info(f"Input batch shape: {batch_shape}")

        # Train model
        logger.info("Starting model training...")
        trainer = config.trainer
        trainer.fit(model, training_loader, validation_loader)
        logger.info("Training completed successfully")

        # Inference
        logger.info("Running inference...")
        clear_vram()

        output = gpu_patch_inference(
            model,
            data.primary_data.astype(np.float32),
            min_overlap=16,
            initial_patch_depth=32,
            device=config.device,
        )
        trainer.save_checkpoint(config.train_config.default_root_dir / "model.ckpt")

        # Post-process and save output
        output = output / np.mean(output) * np.mean(data.primary_data)
        output_path = config.train_config.default_root_dir / "output.tif"
        imwrite(output_path, output)
        logger.info(f"Results saved to {output_path}")

        clear_vram()
        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {e!s}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
