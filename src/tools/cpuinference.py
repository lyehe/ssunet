"""CPU inference in observed photon-count units."""

import numpy as np
import pytorch_lightning as pl
import torch

from .gpuinference import _predict_counts, _volume


def cpu_inference(model: pl.LightningModule, data: np.ndarray) -> np.ndarray:
    """Predict a ZXY or ZCXY volume and preserve its observed photon count."""
    volume, grayscale = _volume(data)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    output = _predict_counts(model, volume, device)
    return output[:, 0] if grayscale else output
