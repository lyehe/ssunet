from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import pyiqa
import seaborn as sns
import torch
import yaml

DEFALT_METRICS = [
    "mse",
    "mae",
    "ncc",
    "psnr",
    "ssim",
    "ms_ssim",
    "niqe",
    "brisque",
]


def import_config(config_path: Path) -> dict:
    """Import configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


class ImageMetrics:
    """Class to handle image metrics."""

    def __init__(
        self,
        image: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        device: torch.device | None = None,
        **kwargs,
    ):
        self._psnr_metric = pyiqa.create_metric("psnr", device=torch.device("cpu"))
        self._ssim_metric = pyiqa.create_metric("ssim", channels=1, device=torch.device("cpu"))
        self._ms_ssim_metric = pyiqa.create_metric("ms_ssim", device=torch.device("cpu"))
        self._nique_metric = pyiqa.create_metric("niqe", device=torch.device("cpu"))
        self._brisque_metric = pyiqa.create_metric("brisque", device=torch.device("cpu"))

        self.device = device or torch.device("cpu")
        if self.device != self._psnr_metric.device:
            self.set_device(self.device)
        self._image = self._to_tensor(image)
        self._target = self._to_tensor(target)
        if self._image.shape != self._target.shape:
            raise ValueError("MismatchError")

        self._grayscale = self._image.ndim == 2
        self._rgb = self._image.shape[0] == 3
        if not self._grayscale and not self._rgb:
            raise ValueError("ColorError")

        self.kwargs = kwargs

        if self._grayscale:
            self._image = self._image[None, None, ...]
            self._target = self._target[None, None, ...]
        else:
            self._image = self._image[None, ...]
            self._target = self._target[None, ...]

    def _to_tensor(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert numpy array to tensor."""
        if isinstance(data, torch.Tensor):
            return data
        else:
            try:
                output = torch.from_numpy(data)
            except TypeError:
                data = data.astype(np.float32)
                output = torch.from_numpy(data)
            return output.to(self.device)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set device for metrics."""
        cls._psnr_metric = pyiqa.create_metric("psnr", device=device)
        cls._ssim_metric = pyiqa.create_metric("ssim", channels=1, device=device)
        cls._ms_ssim_metric = pyiqa.create_metric("ms_ssim", device=device)
        cls._nique_metric = pyiqa.create_metric("niqe", device=device)
        cls._brisque_metric = pyiqa.create_metric("brisque", device=device)

    @staticmethod
    def normalize(data: torch.Tensor) -> torch.Tensor:
        """Normalize tensor."""
        output = data - data.min()
        output = output / output.max()
        return output

    @property
    def mse(self) -> float:
        """Calculate mean squared error."""
        return torch.mean((self._image - self._target) ** 2).item()

    @property
    def mae(self) -> float:
        """Calculate L1 loss."""
        return torch.mean(torch.abs(self._image - self._target)).item()

    @property
    def ncc(self) -> float:
        """Calculate normalized cross-correlation."""
        mean_image = torch.mean(self._image)
        mean_target = torch.mean(self._target)
        std_image = torch.std(self._image)
        std_target = torch.std(self._target)
        covariance = torch.mean((self._image - mean_image) * (self._target - mean_target))
        ncc_val = covariance / (std_image * std_target)
        return ncc_val.item()

    # @property
    def dice(self) -> float:
        """Calculate dice coefficient."""
        intersection = torch.sum(self._image * self._target)
        union = torch.sum(self._image) + torch.sum(self._target)
        dice_val = (2 * intersection) / union
        return dice_val.item()

    @property
    def psnr(self) -> float:
        """Calculate peak signal-to-noise ratio."""
        return self._psnr_metric(self._image, self._target, **self.kwargs).item()

    @property
    def ssim(self) -> float:
        """Calculate structural similarity index."""
        return self._ssim_metric(self._image, self._target, **self.kwargs)[0].item()

    @property
    def ms_ssim(self) -> float:
        """Calculate multi-scale structural similarity index."""
        return self._ms_ssim_metric(self._image, self._target, **self.kwargs).item()

    @property
    def niqe(self) -> float:
        """Calculate natural image quality evaluator."""
        return self._nique_metric(self._image, **self.kwargs).item()

    @property
    def niqe_ref(self) -> float:
        """Calculate natural image quality evaluator."""
        return self._nique_metric(self._target, **self.kwargs).item()

    @property
    def brisque(self) -> float:
        """Calculate brisque."""
        return self._brisque_metric(self._image, **self.kwargs).item()

    @property
    def brisque_ref(self) -> float:
        """Calculate brisque."""
        return self._brisque_metric(self._target, **self.kwargs).item()

    @classmethod
    def metric_list(cls) -> list[str]:
        """List of metrics."""
        return [p for p, v in vars(cls).items() if isinstance(v, property)]

    def export_metrics(self, metrics: list[str] | None = None) -> dict[str, float]:
        """Export metrics."""
        if metrics is None:
            metrics = self.metric_list()
        return {p: getattr(self, p) for p in self.metric_list()}

    def set_image(self, image: torch.Tensor | np.ndarray) -> None:
        """Set a new image for comparison."""
        self._image = self._to_tensor(image)

    def set_target(self, target: torch.Tensor | np.ndarray) -> None:
        """Set a new target for comparison."""
        self._target = self._to_tensor(target)


@dataclass
class MetricStats:
    """A data class to handle and store metric statistics of image stack."""

    data: list[float] = field(repr=False, default_factory=list)
    mean: float = field(init=False)
    std: float = field(init=False)
    max: float = field(init=False)
    min: float = field(init=False)

    def __post_init__(self):
        """Post initialization."""
        self.mean = np.mean(self.data).item()
        self.std = np.std(self.data).item()
        self.max = np.max(self.data)
        self.min = np.min(self.data)

    def __str__(self):
        """String representation."""
        return f"Mean: {self.mean}, Std: {self.std}, Min: {self.min}, Max: {self.max}"

    def __call__(self) -> list[float]:
        """Call."""
        return [self.mean, self.std, self.min, self.max]


class StackMetrics:
    """Class to handle image metrics."""

    def __init__(
        self,
        image: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        device: torch.device | None = None,
        metric_list: list[str] | None = None,
        **kwargs,
    ):
        """Class constructor."""
        self.device = device or torch.device("cpu")

        if image.shape != target.shape:
            raise ValueError("MismatchError")
        if image.ndim != 3:
            raise ValueError("ColorError")

        self.metric_list = metric_list or ImageMetrics.metric_list()
        self.data_list = [
            ImageMetrics(image[i], target[i], device, **kwargs) for i in range(image.shape[0])
        ]

        self._values_df = None
        self._stats_df = None

    def get_metrics(self, metric_type: str) -> list[float]:
        """Get metrics."""
        if not hasattr(self, metric_type):
            setattr(self, metric_type, [getattr(m, metric_type) for m in self.data_list])
        return getattr(self, metric_type)

    def metric_stats(self, metric_type: str) -> MetricStats:
        """Get metric statistics."""
        return MetricStats(self.get_metrics(metric_type))

    @property
    def values_df(self) -> pd.DataFrame:
        """Get values dataframe."""
        if self._values_df is None:
            values = {m: self.get_metrics(m) for m in self.metric_list}
            self._values_df = pd.DataFrame(values)
            self._values_df.index.name = "Frame"
            self._values_df.columns.name = "Metric"
        return self._values_df

    @property
    def stats_df(self) -> pd.DataFrame:
        """Get stats dataframe."""
        if self._stats_df is None:
            data = {p: self.metric_stats(p)() for p in self.metric_list}
            self._stats_df = pd.DataFrame(data, index=["Mean", "Std", "Min", "Max"])
            self._stats_df.index.name = "Stat"
            self._stats_df.columns.name = "Metric"
        return self._stats_df

    @property
    def stats_string(self) -> dict[str, str]:
        """Get stats string."""
        data = {p: self.metric_stats(p)() for p in self.metric_list}
        stats_str = {}
        nl = "\n"
        for k, v in data.items():
            stats_str_list = [
                f"Mean: {v[0]:.3f}",
                f"Std: {v[1]:.3f}",
                f"Min: {v[2]:.3f}",
                f"Max: {v[3]:.3f}",
            ]
            stats_str[k] = nl.join(stats_str_list)
        return stats_str

    def plot_trends(self, **kwargs):
        """Plot trends."""
        data_values = self.values_df.reset_index().melt(
            id_vars="Frame", var_name="Metric", value_name="Value"
        )
        y_range = self.stats_df.loc[["Min", "Max"]].to_numpy()
        params = {
            "col_wrap": kwargs.get("col_wrap", 3),
            "kind": kwargs.get("kind", "line"),
            "aspect": kwargs.get("aspect", 1.5),
            "height": kwargs.get("height", 1.5),
            "markers": kwargs.get("markers", None),
        }
        plot = sns.relplot(
            data=data_values,
            col="Metric",
            x="Frame",
            y="Value",
            hue="Metric",
            style="Metric",
            facet_kws={"sharey": False},
            **params,
        )
        for i, ax in enumerate(plot.axes):
            ax.set_title("")
            ax.set_ylabel("")
            y_min = f"{y_range[0][i]:.3}"
            y_max = f"{y_range[1][i]:.3}"
            ax.set_yticks([y_range[0][i], y_range[1][i]])
            ax.set_yticklabels([y_min, y_max])
        if kwargs.get("save", False):
            self._save_plot(plot, kwargs)
        plot.tight_layout()

    def _save_plot(self, plot, kwargs):
        dir_path = Path(kwargs.get("save_dir", "."))
        dir_path.mkdir(parents=True, exist_ok=True)
        png_path = dir_path / (kwargs.get("save_name", "trend") + ".png")
        svg_path = dir_path / (kwargs.get("save_name", "trend") + ".svg")
        plot.savefig(png_path, format="png", dpi=300)
        plot.savefig(svg_path, format="svg")

    def __len__(self) -> int:
        """Length."""
        return len(self.data_list)

    def __get_item__(self, index: int) -> ImageMetrics:
        """Get item."""
        return self.data_list[index]

    def __iter__(self) -> Iterator:
        """Iterator."""
        return iter(self.data_list)


class StackMetricsGroups:
    """Class to handle image metrics."""

    def __init__(
        self,
        metrics_list: list[StackMetrics],
        group_names: list[str] | None = None,
        metric_list: list[str] | None = None,
        **kwargs,
    ):
        """Class constructor."""
        self.group_names = group_names or [f"G{i+1:>02}" for i in range(len(metrics_list))]
        self.length = len(self.group_names)
        self.data = {self.group_names[i]: metrics_list[i] for i in range(self.length)}
        self.metric_list = metric_list or metrics_list[0].metric_list
        self.kwargs = kwargs

    @classmethod
    def from_image_pairs(
        cls,
        image_list: list[np.ndarray],
        target_list: list[np.ndarray],
        device: torch.device | None = None,
        metric_list: list[str] | None = None,
        group_names: list[str] | None = None,
        **kwargs,
    ):
        """From image pairs."""
        data_list = []
        for i, t in zip(image_list, target_list, strict=False):
            data_list.append(StackMetrics(i, t, device, metric_list, **kwargs))
        return cls(data_list, group_names, **kwargs)

    @classmethod
    def from_dict(
        cls,
        data_dict: dict[str, list[np.ndarray]],
        device: torch.device | None = None,
        metric_list: list[str] | None = None,
        **kwargs,
    ):
        """From dictionary."""
        data_list = []
        for _k, v in data_dict.items():
            data_list.append(StackMetrics(v[0], v[1], device, metric_list, **kwargs))
        return cls(data_list, list(data_dict.keys()), **kwargs)

    @property
    def group_values(self) -> pd.DataFrame:
        """Get group values."""
        data_dict = {}
        for k, v in self.data.items():
            data_dict[k] = v.values_df.reset_index().melt(
                id_vars="Frame", var_name="Metric", value_name="Value"
            )
        return pd.concat(data_dict, names=["Group"])

    @property
    def group_stats(self) -> pd.DataFrame:
        """Get group stats."""
        data_dict = {}
        for k, v in self.data.items():
            data_dict[k] = v.stats_df
        return pd.concat(data_dict, names=["Group"])

    @property
    def y_range(self) -> tuple[np.ndarray, np.ndarray]:
        """Get y range."""
        y_maxs = np.array([])
        y_mins = np.array([])
        for v in self.data.values():
            y_range = v.stats_df.loc[["Min", "Max"]].to_numpy()
            y_mins = y_range[0] if y_mins.size == 0 else np.minimum(y_mins, y_range[0])
            y_maxs = y_range[1] if y_maxs.size == 0 else np.maximum(y_maxs, y_range[1])
        return y_mins, y_maxs

    @property
    def x_range(self) -> tuple[int, int]:
        """Get x range."""
        x_min = 0
        x_max = len(self.data[self.group_names[0]]) - 1
        return x_min, x_max

    def plot_group_trends(self, **kwargs):
        """Plot group trends."""
        data_df = self.group_values
        y_mins, y_maxs = self.y_range
        x_min, x_max = self.x_range
        params = {
            "col_wrap": kwargs.get("col_wrap", 3),
            "kind": kwargs.get("kind", "line"),
            "aspect": kwargs.get("aspect", 1.5),
            "height": kwargs.get("height", 1.5),
            "markers": kwargs.get("markers", None),
        }
        plot = sns.relplot(
            data=data_df,
            col="Metric",
            x="Frame",
            y="Value",
            hue="Group",
            style="Group",
            facet_kws={"sharey": False},
            **params,
        )
        for i, ax in enumerate(plot.axes):
            ax.set_xticks([x_min, x_max])
            ax.set_ylabel("")
            y_min = f"{y_mins[i]:.3}"
            y_max = f"{y_maxs[i]:.3}"
            ax.set_yticks([y_mins[i], y_maxs[i]])
            ax.set_yticklabels([y_min, y_max])
        if kwargs.get("save", False):
            self._save_plot(plot, "group_trends", kwargs)
        plot.tight_layout()

    def plot_group_stats(self, **kwargs):
        """Plot group stats."""
        data_df = self.group_values
        y_mins, y_maxs = self.y_range
        params = {
            "col_wrap": kwargs.get("col_wrap", 3),
            "kind": kwargs.get("kind", "strip"),
            "aspect": kwargs.get("aspect", 1.5),
            "height": kwargs.get("height", 1.5),
        }

        plot = sns.catplot(
            data=data_df,
            x="Group",
            hue="Group",
            y="Value",
            col="Metric",
            native_scale=True,
            sharey=False,
            s=1,
            **params,
        )

        data_stats = self.group_stats
        mean_df = data_stats.xs("Mean", level="Stat")
        std_df = data_stats.xs("Std", level="Stat")

        for i, ax in enumerate(plot.axes):
            ax.set_xlabel("")
            ax.set_ylabel("")
            p = self.metric_list[i]
            means = [mean_df.loc[g, p] for g in self.group_names]
            stds = [std_df.loc[g, p] for g in self.group_names]
            nl = "\n"
            stats_str = [f"{m:.3}{nl}+/-{nl}{s:.3}" for m, s in zip(means, stds, strict=False)]
            text_y = (y_maxs[i] + y_mins[i]) / 2
            va = "top"
            path_effect = [
                pe.SimplePatchShadow(offset=(0.5, -0.5), alpha=1, shadow_rgbFace="white"),
                pe.Normal(),
            ]

            if params["kind"] != "bar":
                y_min = f"{y_mins[i]:.3}"
                y_max = f"{y_maxs[i]:.3}"
                ax.set_yticks([y_mins[i], y_maxs[i]])
                ax.set_yticklabels([y_min, y_max])
                text_y = (y_maxs[i] + y_mins[i]) / 2
                va = "bottom"

            for text_x, stat in enumerate(stats_str):
                ax.text(
                    text_x,
                    text_y,
                    stat,
                    ha="center",
                    va=va,
                    fontsize=8,
                    path_effects=path_effect,
                )
        if kwargs.get("save", False):
            self._save_plot(plot, "group_stats", kwargs)
        plot.tight_layout()

    def _save_plot(self, plot, name, kwargs):
        dir_path = Path(kwargs.get("save_dir", "."))
        dir_path.mkdir(parents=True, exist_ok=True)
        png_path = dir_path / (kwargs.get("save_name", f"{name}") + ".png")
        svg_path = dir_path / (kwargs.get("save_name", f"{name}") + ".svg")
        plot.savefig(png_path, format="png", dpi=300)
        plot.savefig(svg_path, format="svg")

    def __len__(self) -> int:
        """Length."""
        return self.length

    def __get_item__(self, index: int) -> StackMetrics:
        """Get item."""
        return self.data[self.group_names[index]]

    def __iter__(self) -> Iterator:
        """Iterator."""
        return iter(self.data.values())

    @classmethod
    def from_config(cls, config: Path | dict, **kwargs):
        """From config."""
        if isinstance(config, Path):
            config = import_config(config)
