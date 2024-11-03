import gc
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .metrics import StackMetrics, StackMetricsGroups


def show_image_sum(image: np.ndarray | torch.Tensor):
    """Imshow for Tensor."""
    image_sum = image.reshape((-1, image.shape[-2], image.shape[-1]))
    if isinstance(image_sum, torch.Tensor):
        image_sum = torch.sum(image_sum, dim=0).detach().cpu().numpy()
    else:
        image_sum = np.sum(image_sum, axis=0)
    plt.figure(figsize=(15, 15))
    plt.imshow(image_sum, cmap="gray")
    plt.title(f"Image shape: {image.shape}")
    plt.axis("off")
    plt.show()


def imshow(image: np.ndarray):
    """Simple imshow for numpy array."""
    image = image - np.min(image)
    image = image / np.max(image)
    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def clear_vram():
    """Run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()


def remove_empty_directory(path: Path):
    """Remove empty directory."""
    list_dir = list(path.iterdir())
    for dir in list_dir:
        if dir.is_dir():
            remove_empty_directory(dir)
    if not list(path.iterdir()):
        path.rmdir()
    return path


def load_dir_path(path: Path = Path("./config.yml")):
    """Load directory path."""
    if path.exists():
        with open(path) as file:
            config = yaml.safe_load(file)
        return Path(config["DATA"]["dir_path"])
    else:
        raise FileNotFoundError(path)


def load_data_path(path: Path = Path("./config.yml")):
    """Load data path."""
    if path.exists():
        with open(path) as file:
            config = yaml.safe_load(file)
        return Path(config["DATA"]["data_path"])
    else:
        raise FileNotFoundError(path)


def list_dir(path: str | Path) -> list[Path]:
    """List directory and return list of Path objects."""
    dirctory_path = Path(path)
    dirctory_list = list(dirctory_path.iterdir())
    for _i, _dir in enumerate(dirctory_list):
        pass
    return dirctory_list


def clean_directories(model_path: Path = Path("../models")):
    """Remove empty model directories."""
    model_list = [model for model in model_path.iterdir() if model.is_dir()]
    for model in model_list:
        if not list(model.glob("**/*.ckpt")):
            shutil.rmtree(model)


def group_metrics(
    input: np.ndarray,
    image: np.ndarray,
    ground_truth: np.ndarray,
    default_root_dir: Path,
    length: int = 512,
    device: torch.device | None = None,
):
    """Output group metrics."""
    ground_truth_new = ground_truth.copy()
    for i in range(length):
        scale = np.mean(ground_truth[i]) / 255
        ground_truth_new[i] = ground_truth_new[i] / 255
        image[i] = image[i] / np.mean(image[i]) * scale
        input[i] = input[i] / np.mean(input[i]) * scale

    metric_list = ["mse", "psnr", "ssim"]
    metric1 = StackMetrics(
        image,
        ground_truth_new,
        metric_list=metric_list,
        device=device,
    )
    metric2 = StackMetrics(
        input,
        ground_truth_new,
        metric_list=metric_list,
        device=device,
    )
    metric_group = StackMetricsGroups([metric1, metric2], ["processed", "raw"], metric_list)
    metric_group.plot_group_stats(save=True, save_dir=default_root_dir, save_name="group_stats")
    metric_group.plot_group_trends(
        save=True,
        save_dir=default_root_dir,
        save_name="group_trends",
    )
    metric1.stats_df.to_csv(default_root_dir / "processed_stats.csv")
    metric2.stats_df.to_csv(default_root_dir / "raw_stats.csv")
    metric1.values_df.to_csv(default_root_dir / "processed_values.csv")
    metric2.values_df.to_csv(default_root_dir / "raw_values.csv")
    out = metric1.stats_df.loc["Mean", "psnr"]
    return out
