"""CPU inference script.

This script is used to run inference on CPU. It will be very slow for large images.
"""

import numpy as np
import pytorch_lightning as pl
import torch

from ssunet.constants import LOGGER


def cpu_inference(model: pl.LightningModule, data: np.ndarray) -> np.ndarray:
    """Run inference on CPU."""
    LOGGER.info("Starting CPU inference")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    if data.dtype == bool:
        data = data.astype(np.float32)
        LOGGER.debug("Converted boolean data to float32")
    LOGGER.debug(f"Input data shape: {data.shape}")
    with torch.inference_mode():
        torch_data = torch.from_numpy(data)[None, None, ...]
        LOGGER.debug(f"Torch data shape: {torch_data.shape}")
        output = torch.exp(model(torch_data))[0, 0]
    LOGGER.debug(f"Output shape: {output.shape}")
    LOGGER.info("CPU inference completed")
    return output.detach().numpy()
