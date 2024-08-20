import numpy as np
import pytorch_lightning as pl
import torch


def cpu_inference(model: pl.LightningModule, data: np.ndarray) -> np.ndarray:
    """Run inference on CPU"""
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    if data.dtype == bool:
        data = data.astype(np.float32)
    with torch.inference_mode():
        troch_data = torch.from_numpy(data)[None, None, ...]
        output = torch.exp(model(troch_data))[0, 0]
    return output.detach().numpy()


