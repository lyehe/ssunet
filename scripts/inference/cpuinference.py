"""CPU inference script.

This script is used to run inference on CPU. It will be very slow for large images.
"""

from logging import getLogger

import numpy as np
import pytorch_lightning as pl
import torch

logger = getLogger(__name__)


def cpu_inference(model: pl.LightningModule, data: np.ndarray) -> np.ndarray:
    """Run inference on CPU."""
    logger.info("Starting CPU inference")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    if data.dtype == bool:
        data = data.astype(np.float32)
        logger.debug("Converted boolean data to float32")
    logger.debug(f"Input data shape: {data.shape}")
    with torch.inference_mode():
        torch_data = torch.from_numpy(data)[None, None, ...]
        logger.debug(f"Torch data shape: {torch_data.shape}")
        output = torch.exp(model(torch_data))[0, 0]
    logger.debug(f"Output shape: {output.shape}")
    logger.info("CPU inference completed")
    return output.detach().numpy()
