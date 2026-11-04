"""Patch-based inference with stable photon-distribution normalization."""

import math
from dataclasses import dataclass, field
from itertools import pairwise, product

import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

if __package__ == "src.tools":
    from ..ssunet.constants import LOGGER
    from ..ssunet.inference import (
        logits_to_expected_counts,
        model_distribution_dims,
        reflected_window,
        temporal_window_size,
    )
else:
    from ssunet.constants import LOGGER
    from ssunet.inference import (
        logits_to_expected_counts,
        model_distribution_dims,
        reflected_window,
        temporal_window_size,
    )


class InvalidDataDimensionError(ValueError):
    """Exception raised when the input data has invalid dimensions."""

    def __init__(self):
        super().__init__("Data must be 3D or 4D")


class PatchSizeTooLargeError(RuntimeError):
    """Exception raised when the patch size is too large for available VRAM."""

    def __init__(self):
        super().__init__("Patch size too large for available VRAM")


class InvalidPatchValuesError(ValueError):
    """Exception raised when patch values cannot cover the requested volume."""

    def __init__(self):
        super().__init__("Patch values are too small")


def _volume(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert ZXY or ZCXY data to float32 ZCXY without changing its owner."""
    data = np.asarray(data)
    grayscale = data.ndim == 3
    if grayscale:
        data = data[:, None, ...]
    elif data.ndim != 4:
        raise InvalidDataDimensionError()
    if any(size == 0 for size in data.shape):
        message = "Inference requires a nonempty volume"
        raise ValueError(message)
    return data.astype(np.float32, copy=False), grayscale


def _device(device: int | str | torch.device) -> torch.device:
    """Resolve the historically supported integer GPU indices."""
    return torch.device(f"cuda:{device}" if isinstance(device, int) else device)


def _predict_counts(
    model: torch.nn.Module,
    patch: np.ndarray,
    device: torch.device,
    mixed_precision: bool = False,
    valid_length: int | None = None,
) -> np.ndarray:
    """Predict a ZCXY patch in the same expected-count units as its input."""
    valid_length = patch.shape[0] if valid_length is None else valid_length
    depth = temporal_window_size(model, patch.shape[0])
    if depth != patch.shape[0]:
        patch = reflected_window(patch, 0, depth)
    batch = torch.from_numpy(np.ascontiguousarray(patch.swapaxes(0, 1)))[None].to(device)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed_precision),
    ):
        logits = model(batch)
        prediction = logits_to_expected_counts(
            logits[:, :, :valid_length],
            batch[:, :, :valid_length],
            dim=model_distribution_dims(model),
        )
    return prediction[0].float().cpu().numpy().swapaxes(0, 1)


def gpu_inference(model: pl.LightningModule, data: np.ndarray, device_num: int = 0) -> np.ndarray:
    """Run full-volume GPU inference and preserve the observed photon count."""
    data, grayscale = _volume(data)
    device = _device(device_num)
    model.to(device)
    model.eval()
    output = _predict_counts(model, data, device)
    return output[:, 0] if grayscale else output


def gpu_patch_inference(
    model: pl.LightningModule,
    data: np.ndarray,
    min_overlap: int,
    device: int | str | torch.device = 0,
    initial_patch_depth: int = 64,
    test_vram=False,
    mixed_precision=False,
) -> np.ndarray:
    """Run temporal patch inference on CPU or GPU with aligned overlap blending.

    Each window is normalized to its observed count before blending. Short
    sequences are reflected to the requested depth and trimmed to recorded time.
    """
    if initial_patch_depth <= 0 or min_overlap < 0 or min_overlap >= initial_patch_depth:
        message = "Require 0 <= min_overlap < initial_patch_depth"
        raise ValueError(message)
    data, grayscale = _volume(data)
    device = _device(device)
    model.to(device)
    model.eval()
    z, c, x, y = data.shape
    depth = temporal_window_size(model, initial_patch_depth)
    if test_vram:
        depth = patch_sizer(model, (c, depth, x, y), depth, min_overlap, mixed_precision)

    if z <= depth:
        starts = [0]
    else:
        starts = list(range(0, z - depth + 1, depth - min_overlap))
        if starts[-1] != z - depth:
            starts.append(z - depth)
    output = np.zeros(data.shape, dtype=np.float32)
    coverage = np.zeros(z, dtype=np.int32)
    for start in tqdm(starts, desc="Inference #"):
        window = reflected_window(data, start, depth)
        end = min(start + depth, z)
        prediction = _predict_counts(
            model, window, device, mixed_precision, valid_length=end - start
        )
        output[start:end] += prediction[: end - start]
        coverage[start:end] += 1
    output /= coverage[:, None, None, None]
    return output[:, 0] if grayscale else output


def gpu_skip_inference(
    model: pl.LightningModule,
    data: np.ndarray,
    skip: int,
    device: int | str | torch.device = 0,
    patch_depth: int = 32,
    mixed_precision=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict selected center frames using temporally aligned context windows."""
    if skip <= 0 or patch_depth <= 0:
        message = "skip and patch_depth must be positive"
        raise ValueError(message)
    data, grayscale = _volume(data)
    device = _device(device)
    model.to(device)
    model.eval()
    patch_depth = temporal_window_size(model, patch_depth)
    half = patch_depth // 2
    frame_idx = (
        list(range(half, data.shape[0] - half, skip))
        if data.shape[0] > patch_depth
        else list(range(0, data.shape[0], skip))
    )
    skipped_data = data[frame_idx]
    output = np.zeros_like(skipped_data)
    for i, frame in enumerate(tqdm(frame_idx, desc="Inference #")):
        window = reflected_window(data, frame - half, patch_depth)
        output[i] = _predict_counts(model, window, device, mixed_precision)[half]
    if grayscale:
        return skipped_data[:, 0], output[:, 0]
    return skipped_data, output


def patch_sizer(
    model: pl.LightningModule,
    patch_size: tuple[int, int, int, int],
    initial_patch_depth: int,
    min_overlap: int,
    mixed_precision: bool = False,
) -> int:
    """Find a temporal depth that fits, reducing the actual CZYX test shape."""
    if initial_patch_depth <= 0 or not 0 <= min_overlap < initial_patch_depth:
        message = "Require 0 <= min_overlap < initial_patch_depth"
        raise ValueError(message)
    depth = temporal_window_size(model, initial_patch_depth)
    minimum = temporal_window_size(model, 1)
    while depth > min_overlap and depth >= minimum:
        try:
            test_model_vram(model, (patch_size[0], depth, *patch_size[2:]), mixed_precision)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            LOGGER.info("Patch depth %d too large; reducing it", depth)
            depth = (depth // 2 // minimum) * minimum
        else:
            return depth
    raise PatchSizeTooLargeError()


def test_model_vram(
    model: pl.LightningModule,
    patch_size: tuple[int, int, int, int],
    mixed_precision=False,  # noqa: PT028 -- public inference helper, not a pytest test
) -> None:
    """Estimate model memory for batch-first CZYX input."""
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters(), torch.empty(0)).device
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    dummy_input = torch.rand(1, *patch_size, device=device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed_precision),
    ):
        output = model(dummy_input)
    del dummy_input, output
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _spatial_starts(size: int, patch_size: int, count: int) -> tuple[int, list[int]]:
    """Choose evenly spaced, overlapping spatial patches covering one dimension."""
    if patch_size <= 0 or count <= 0:
        message = "Spatial patch size and split counts must be positive"
        raise ValueError(message)
    patch_size = min(size, patch_size)
    if count == 1:
        if patch_size < size:
            message = "A single spatial patch must cover its dimension"
            raise ValueError(message)
        return patch_size, [0]
    starts = sorted(set(np.linspace(0, size - patch_size, count, dtype=int).tolist()))
    if any(b - a > patch_size for a, b in pairwise(starts)):
        message = "Spatial patches leave uncovered pixels; increase patch size or split"
        raise ValueError(message)
    return patch_size, starts


def grid_inference(
    data: np.ndarray,
    model: pl.LightningModule,
    device: torch.device,
    split: int | tuple = 3,
    patch_size: int = 512,
    min_overlap: int = 16,
    initial_patch_depth: int = 32,
) -> np.ndarray:
    """Run spatially tiled inference in common expected-count units."""
    volume, grayscale = _volume(data)
    split_x, split_y = split if isinstance(split, tuple) else (split, split)
    patch_x, starts_x = _spatial_starts(volume.shape[-2], patch_size, split_x)
    patch_y, starts_y = _spatial_starts(volume.shape[-1], patch_size, split_y)
    result = np.zeros_like(volume)
    coverage = np.zeros(volume.shape[-2:], dtype=np.int32)
    for x, y in tqdm(
        product(starts_x, starts_y), total=len(starts_x) * len(starts_y), desc="Grid inference"
    ):
        selection = (..., slice(x, x + patch_x), slice(y, y + patch_y))
        result[selection] += gpu_patch_inference(
            model,
            volume[selection],
            min_overlap=min_overlap,
            initial_patch_depth=initial_patch_depth,
            device=device,
        )
        coverage[x : x + patch_x, y : y + patch_y] += 1
    result /= coverage
    return result[:, 0] if grayscale else result


@dataclass
class PatchIdx:
    """Class to handle patch indices."""

    dim_size: int
    patch_size: int
    num_patches: int
    overlap: int
    start_idx: list = field(init=False, default_factory=list)
    end_idx: list = field(init=False, default_factory=list)
    local_start_idx: list = field(init=False, default_factory=list)
    local_end_idx: list = field(init=False, default_factory=list)
    start_non_overlap: list = field(init=False, default_factory=list)
    end_non_overlap: list = field(init=False, default_factory=list)

    def __post_init__(self):
        """Post initialization function."""
        self._calculate_idx()

    @classmethod
    def from_patch_size(cls, dim_size: int, patch_size: int, overlap: int = 0):
        """Create patch indices from patch size."""
        num_patches = math.ceil((dim_size - overlap) / (patch_size - overlap))
        return cls(dim_size, patch_size, num_patches, overlap)

    @classmethod
    def from_num_patches(cls, dim_size: int, num_patches: int, overlap: int = 0):
        """Create patch indices from number of patches."""
        patch_size = math.ceil((dim_size - overlap) / num_patches) + overlap
        return cls(dim_size, patch_size, num_patches, overlap)

    @classmethod
    def from_num_size(cls, dim_size: int, num_patches: int, patch_size: int):
        """Create patch indices from number of patches and patch size."""
        total_size = num_patches * patch_size
        if total_size < dim_size:
            raise InvalidPatchValuesError()
        overlap = (num_patches * patch_size - dim_size) // (num_patches - 1)
        return cls(dim_size, patch_size, num_patches, overlap)

    def _calculate_idx(self) -> None:
        """Calculate patch indices."""
        self._start_idx = []
        self._end_idx = []
        self._local_start_idx = []
        self._local_end_idx = []
        self._start_non_overlap = []
        self._end_non_overlap = []
        for i in range(self.num_patches):
            start = i * (self.patch_size - self.overlap)
            end = min(start + self.patch_size, self.dim_size)
            self._start_idx.append(start)
            self._end_idx.append(end)

    def __call__(self) -> tuple[list, list]:
        """Get start and end indices."""
        return self._start_idx, self._end_idx

    def __len__(self) -> int:
        """Get number of patches."""
        return self.num_patches

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Get start and end indices."""
        return self._start_idx[index], self._end_idx[index]

    def __iter__(self):
        """Iterate over start and end indices."""
        return zip(self._start_idx, self._end_idx, strict=False)
