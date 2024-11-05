"""GPU inference script."""

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pytorch_lightning as pl
import torch
from torch.cuda.amp.autocast_mode import autocast
from tqdm import tqdm

from ..ssunet.constants import LOGGER


class InvalidDataDimensionError(ValueError):
    """Exception raised when the input data has invalid dimensions."""

    def __init__(self):
        super().__init__("Data must be 3D or 4D")
        LOGGER.error("InvalidDataDimensionError: Data must be 3D or 4D")


class PatchSizeTooLargeError(RuntimeError):
    """Exception raised when the patch size is too large for available VRAM."""

    def __init__(self):
        super().__init__("Patch size too large for available VRAM")
        LOGGER.error("PatchSizeTooLargeError: Patch size too large for available VRAM")


class InvalidPatchValuesError(ValueError):
    """Exception raised when patch values are too small."""

    def __init__(self):
        super().__init__("Patch values are too small")
        LOGGER.error("InvalidPatchValuesError: Patch values are too small")


def gpu_inference(model: pl.LightningModule, data: np.ndarray, device_num: int = 0) -> np.ndarray:
    """Run inference on GPU."""
    device = torch.device(f"cuda:{device_num}")
    model.to(device)
    model.eval()
    with torch.no_grad():
        torch_data = torch.from_numpy(data)[None, None, ...].to(device)
        output = torch.exp(model(torch_data))[0, 0]
    return output.detach().cpu().numpy()


def gpu_patch_inference(
    model: pl.LightningModule,
    data: np.ndarray,
    min_overlap: int,
    device: int | str | torch.device = 0,
    initial_patch_depth: int = 64,
    test_vram=False,
    mixed_precision=False,
) -> np.ndarray:
    """Run patch inference on GPU."""
    if isinstance(device, int):
        device = f"cuda:{device}"
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    model.eval()

    num_dims = len(data.shape)
    if num_dims == 3:
        data = data[:, None, ...]
    elif num_dims == 4:
        pass
    else:
        raise InvalidDataDimensionError()

    z, c, x, y = data.shape
    patch_depth = initial_patch_depth
    patch_size = (patch_depth, c, x, y)

    patch_depth = (
        patch_sizer(model, patch_size, initial_patch_depth, min_overlap, mixed_precision)
        if test_vram
        else patch_depth
    )
    num_patches = (z + patch_depth - min_overlap - 1) // (patch_depth - min_overlap)
    overlap = (num_patches * patch_depth - z) // (num_patches - 1) if num_patches > 1 else 0

    output = np.zeros_like(data, dtype=np.float32)
    stream_for_data = torch.cuda.Stream(device=device)
    stream_for_inference = torch.cuda.Stream(device=device)

    data = data.swapaxes(0, 1).astype(np.float32)

    with torch.no_grad(), tqdm(total=num_patches, desc="Inference #") as pbar:
        for i in range(num_patches):
            start_depth = min(i * (patch_depth - overlap), z - patch_depth)
            end_depth = min(start_depth + patch_depth, z)

            with torch.cuda.stream(stream_for_data):  # type: ignore
                patch = data[:, start_depth:end_depth]
                torch_patch = torch.from_numpy(patch)[None, ...].to(device, non_blocking=True)
            stream_for_data.synchronize()
            with torch.cuda.stream(stream_for_inference):  # type: ignore
                if mixed_precision:
                    with autocast():
                        patch_output = torch.exp(model(torch_patch))[0]
                else:
                    patch_output = torch.exp(model(torch_patch))[0]
                output_cpu = patch_output.detach().cpu().numpy().swapaxes(0, 1)
            stream_for_inference.synchronize()

            pbar.set_postfix(
                vram_usage=f"{torch.cuda.max_memory_reserved(device) / (1024 ** 3):.1f} GB"
            )

            if i == 0:
                output[start_depth:end_depth] = output_cpu
            elif i == num_patches - 1:
                output[start_depth + overlap // 2 :] = output_cpu[overlap // 2 :]
            else:
                prev_end_depth = start_depth + overlap // 2
                output[start_depth:prev_end_depth] = output[start_depth:prev_end_depth]
                output[prev_end_depth:end_depth] = output_cpu[overlap // 2 :]

            pbar.update(1)

    return output if num_dims == 4 else output[:, 0]


def gpu_skip_inference(
    model: pl.LightningModule,
    data: np.ndarray,
    skip: int,
    device: int | str | torch.device = 0,
    patch_depth: int = 32,
    mixed_precision=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run skip inference on GPU.

    Some frames are skipped to reduce the memory usage and data size.
    """
    if isinstance(device, int):
        device = f"cuda:{device}"
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    model.eval()

    num_dims = len(data.shape)
    if num_dims == 3:
        data = data[:, None, ...]
    elif num_dims == 4:
        pass
    else:
        raise InvalidDataDimensionError()

    z, c, x, y = data.shape
    skipped_start_idx = patch_depth // 2
    skipped_end_idx = z - patch_depth // 2
    frame_idx = list(range(skipped_start_idx, skipped_end_idx, skip))
    skipped_data = data[skipped_start_idx:skipped_end_idx:skip, ...]
    output = np.zeros_like(skipped_data, dtype=np.float32)
    num_patches = len(skipped_data)

    stream_for_data = torch.cuda.Stream(device=device)
    stream_for_inference = torch.cuda.Stream(device=device)

    data = data.swapaxes(0, 1)

    with torch.no_grad(), tqdm(total=num_patches, desc="Inference #") as pbar:
        for i in range(num_patches):
            start_depth = frame_idx[i] - (patch_depth // 2)
            end_depth = frame_idx[i] + (patch_depth // 2)

            with torch.cuda.stream(stream_for_data):  # type: ignore
                patch = data[:, start_depth:end_depth].astype(np.float32)
                torch_patch = torch.from_numpy(patch)[None, ...].to(device, non_blocking=True)
            stream_for_data.synchronize()
            with torch.cuda.stream(stream_for_inference):  # type: ignore
                if mixed_precision:
                    with autocast():
                        patch_output = torch.exp(model(torch_patch))[0]
                else:
                    patch_output = torch.exp(model(torch_patch))[0]
                output_cpu = patch_output.detach().cpu().numpy().swapaxes(0, 1)
            stream_for_inference.synchronize()
            output[i] = output_cpu[patch_depth // 2]
            pbar.update()
            pbar.set_postfix(
                vram_usage=f"{torch.cuda.max_memory_reserved(device) / (1024 ** 3):.1f} GB"
            )
    return (skipped_data, output) if num_dims == 4 else (skipped_data[:, 0], output[:, 0])


def patch_sizer(
    model: pl.LightningModule,
    patch_size: tuple[int, int, int, int],
    initial_patch_depth: int,
    min_overlap: int,
    mixed_precision: bool = False,
) -> int:
    """Determine the patch depth based on the VRAM capacity."""
    patch_depth = initial_patch_depth
    while patch_depth > min_overlap * 2 + 1:
        try:
            test_model_vram(model, patch_size, mixed_precision)
            break
        except RuntimeError:
            LOGGER.info(f"Patch depth {patch_size} too large. Reducing patch depth")
            patch_depth = patch_depth // 2
        if patch_depth < min_overlap:
            LOGGER.error("Patch size too large. Exiting")
            raise PatchSizeTooLargeError()
    LOGGER.info(f"Patch depth: {patch_depth}")
    return patch_depth


def test_model_vram(
    model: pl.LightningModule,
    patch_size: tuple[int, int, int, int],
    mixed_precision=False,
) -> None:
    """Estimate the VRAM usage of the model."""
    dummy_input = torch.rand(1, *patch_size).to(model.device)
    with torch.no_grad():
        if mixed_precision:
            with autocast():
                _ = model(dummy_input)
        _ = model(dummy_input)
    del dummy_input, _
    torch.cuda.empty_cache()


def grid_inference(
    data: np.ndarray,
    model: pl.LightningModule,
    device: torch.device,
    split: int | tuple = 3,
    patch_size: int = 512,
    min_overlap: int = 16,
    initial_patch_depth: int = 32,
) -> np.ndarray:
    """Run grid inference on GPU."""
    if len(data.shape) == 3:
        _, max_x, max_y = data.shape
    else:
        _, __, max_x, max_y = data.shape
    split_x, split_y = split if isinstance(split, tuple) else (split, split)

    result = np.zeros_like(data, dtype=np.float32)

    # Calculate overlaps ensuring they're even numbers
    overlap_x = (patch_size * split_x - max_x + 1) // (split_x - 1)
    overlap_y = (patch_size * split_y - max_y + 1) // (split_y - 1)

    total_patches = split_x * split_y
    for i, j in tqdm(
        product(range(split_x), range(split_y)), total=total_patches, desc="Grid inference"
    ):
        # Calculate patch boundaries from the input data
        if i < (split_x - 1):
            patch_x_start = i * (patch_size - overlap_x)
            patch_x_end = patch_x_start + patch_size
        else:  # last row
            patch_x_end = max_x
            patch_x_start = patch_x_end - patch_size

        if j < (split_y - 1):
            patch_y_start = j * (patch_size - overlap_y)
            patch_y_end = patch_y_start + patch_size
        else:  # last column
            patch_y_end = max_y
            patch_y_start = patch_y_end - patch_size

        if i == 0:
            result_x_start = 0
            result_x_end = patch_x_end - overlap_x // 2
            output_x_start = 0
            output_x_end = patch_size - overlap_x // 2
        elif i == split_x - 1:
            result_x_start = patch_x_start + overlap_x // 2 - 1
            result_x_end = max_x
            output_x_start = overlap_x // 2 - 1
            output_x_end = patch_size
        else:
            result_x_start = patch_x_start + overlap_x // 2
            result_x_end = patch_x_end - overlap_x // 2
            output_x_start = overlap_x // 2
            output_x_end = patch_size - overlap_x // 2

        if j == 0:
            result_y_start = 0
            result_y_end = patch_y_end - overlap_y // 2
            output_y_start = 0
            output_y_end = patch_size - overlap_y // 2
        elif j == split_y - 1:
            result_y_start = patch_y_start + overlap_y // 2 - 1
            result_y_end = max_y
            output_y_start = overlap_y // 2 - 1
            output_y_end = patch_size
        else:
            result_y_start = patch_y_start + overlap_y // 2
            result_y_end = patch_y_end - overlap_y // 2
            output_y_start = overlap_y // 2
            output_y_end = patch_size - overlap_y // 2

        data_patch = data[..., patch_x_start:patch_x_end, patch_y_start:patch_y_end]
        output = gpu_patch_inference(
            model,
            data_patch.astype(np.float32),
            min_overlap=min_overlap,
            initial_patch_depth=initial_patch_depth,
            device=device,
        )

        input_patch = data[..., result_x_start:result_x_end, result_y_start:result_y_end]
        output_patch = output[..., output_x_start:output_x_end, output_y_start:output_y_end]
        output_patch = (
            output_patch
            / np.mean(output_patch, axis=(1, 2), keepdims=True)
            * np.mean(input_patch, axis=(1, 2), keepdims=True)
        ) if len(data.shape) == 3 else (output_patch
            / np.mean(output_patch, axis=(1, 2, 3), keepdims=True)
            * np.mean(input_patch, axis=(1, 2, 3), keepdims=True)
        )
        result[..., result_x_start:result_x_end, result_y_start:result_y_end] = output_patch
    return result


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
