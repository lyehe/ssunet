import numpy as np
import pytorch_lightning as pl
import torch
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from torch.cuda.amp.autocast_mode import autocast
import math

logger = logging.getLogger(__name__)


def gpu_inference(
    model: pl.LightningModule, data: np.ndarray, device_num: int = 0
) -> np.ndarray:
    """Run inference on GPU"""
    device = torch.device(f"cuda:{device_num}")
    model.to(device)
    model.eval()
    with torch.no_grad():
        troch_data = torch.from_numpy(data)[None, None, ...].to(device)
        output = torch.exp(model(troch_data))[0, 0]
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
        raise ValueError("Data must be 3D or 4D")

    z, c, x, y = data.shape
    patch_depth = initial_patch_depth
    patch_size = (patch_depth, c, x, y)

    patch_depth = (
        patch_sizer(
            model, patch_size, patch_depth, min_overlap, device, mixed_precision
        )
        if test_vram
        else patch_depth
    )
    num_patches = (z + patch_depth - min_overlap - 1) // (patch_depth - min_overlap)
    overlap = (
        (num_patches * patch_depth - z) // (num_patches - 1) if num_patches > 1 else 0
    )

    output = np.zeros_like(data, dtype=np.float32)
    stream_for_data = torch.cuda.Stream(device=device)
    stream_for_inference = torch.cuda.Stream(device=device)

    data = data.swapaxes(0, 1)

    with torch.no_grad(), tqdm(total=num_patches, desc="Inference #") as pbar:
        for i in range(num_patches):
            start_depth = min(i * (patch_depth - overlap), z - patch_depth)
            end_depth = min(start_depth + patch_depth, z)

            with torch.cuda.stream(stream_for_data):  # type: ignore
                patch = data[:, start_depth:end_depth]
                torch_patch = torch.from_numpy(patch)[None, ...].to(
                    device, non_blocking=True
                )
            stream_for_data.synchronize()
            with torch.cuda.stream(stream_for_inference):  # type: ignore
                if mixed_precision:
                    with autocast():
                        patch_output = torch.exp(model(torch_patch))[0]
                else:
                    patch_output = torch.exp(model(torch_patch))[0]
                output_cpu = patch_output.detach().cpu().numpy().swapaxes(0, 1)
            stream_for_inference.synchronize()
            if i == 0:
                output[start_depth:end_depth] = output_cpu
            elif i == num_patches - 1:
                output[start_depth + overlap // 2 :] = output_cpu[overlap // 2 :]
            else:
                prev_end_depth = start_depth + overlap // 2
                output[start_depth:prev_end_depth] = output[start_depth:prev_end_depth]
                output[prev_end_depth:end_depth] = output_cpu[overlap // 2 :]

            pbar.update(1)
            pbar.set_postfix(
                vram_usage=f"{torch.cuda.memory_allocated(device) / (1024 ^ 3):.1f} GB"
            )
    return output if num_dims == 4 else output[:, 0]


def gpu_skip_inference(
    model: pl.LightningModule,
    data: np.ndarray,
    skip: int,
    device: int | str | torch.device = 0,
    patch_depth: int = 32,
    mixed_precision=False,
) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError("Data must be 3D or 4D")

    z, c, x, y = data.shape
    skiped_start_idx = patch_depth // 2
    skiped_end_idx = z - patch_depth // 2
    frame_idx = list(range(skiped_start_idx, skiped_end_idx, skip))
    skipped_data = data[skiped_start_idx:skiped_end_idx:skip, ...]
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
                torch_patch = torch.from_numpy(patch)[None, ...].to(
                    device, non_blocking=True
                )
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
                vram_usage=f"{torch.cuda.memory_allocated(device) / (1024 ^ 3):.1f} GB"
            )
    return (
        (skipped_data, output) if num_dims == 4 else (skipped_data[:, 0], output[:, 0])
    )


def patch_sizer(
    model: pl.LightningModule,
    patch_size: tuple[int, int, int, int],
    inital_patch_depth: int,
    min_overlap: int,
    device: torch.device,
    mixed_precision: bool = False,
) -> int:
    """Test VRAM capacity"""
    patch_depth = inital_patch_depth
    while patch_depth > min_overlap * 2 + 1:
        try:
            test_model_vram(model, patch_size, mixed_precision)
            break
        except RuntimeError:
            logger.info(f"Patch depth {patch_size} too large. Reducing patch depth")
            patch_depth = patch_depth // 2
        if patch_depth < min_overlap:
            logger.error("Patch size too large. Exiting")
            raise RuntimeError("Patch size too large")
    logger.info(f"Patch depth: {patch_depth}")
    return patch_depth


def test_model_vram(
    model: pl.LightningModule,
    patch_size: tuple[int, int, int, int],
    mixed_precision=False,
) -> None:
    """Estimate the VRAM usage of the model"""
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
    patch: int = 512,
    min_overlap: int = 8,
    initial_patch_depth: int = 16,
) -> np.ndarray:
    dz, dx, dy = data.shape
    split_x, split_y = split if isinstance(split, tuple) else (split, split)
    processed_data = np.zeros((dz, dx, dy))
    overlap_x = (patch * split_x - dx + 1) // (split_x - 1)
    overlap_y = (patch * split_y - dy + 1) // (split_y - 1)
    for i in range(split_x):
        for j in range(split_y):
            x0 = i * (patch - overlap_x)
            x1 = x0 + patch
            y0 = j * (patch - overlap_y)
            y1 = y0 + patch

            if i == 0:
                x0b = 0
                x1b = x1 - overlap_x // 2
                x0c = 0
                x1c = patch - overlap_x // 2
            elif i == split_x - 1:
                x0b = x0 + overlap_x // 2
                x1b = dx
                x0c = overlap_x // 2
                x1c = patch
            else:
                x0b = x0 + overlap_x // 2
                x1b = x1 - overlap_x // 2
                x0c = overlap_x // 2
                x1c = patch - overlap_x // 2

            if j == 0:
                y0b = 0
                y1b = y1 - overlap_y // 2
                y0c = 0
                y1c = patch - overlap_y // 2
            elif j == split_y - 1:
                y0b = y0 + overlap_y // 2
                y1b = dy
                y0c = overlap_y // 2
                y1c = patch
            else:
                y0b = y0 + overlap_y // 2
                y1b = y1 - overlap_y // 2
                y0c = overlap_y // 2
                y1c = patch - overlap_y // 2

            data_patch = data[:, x0:x1, y0:y1]
            output = gpu_patch_inference(
                model,
                data_patch.astype(np.float32),
                min_overlap=min_overlap,
                initial_patch_depth=initial_patch_depth,
                device=device,
            )
            processed_data[:, x0b:x1b, y0b:y1b] = output[:, x0c:x1c, y0c:y1c]
    return processed_data


@dataclass
class PatchIdx:
    _dim_size: int
    _patch_size: int
    _num_patches: int
    _overlap: int
    _start_idx: list = field(init=False, default_factory=list)
    _end_idx: list = field(init=False, default_factory=list)
    _local_start_idx: list = field(init=False, default_factory=list)
    _local_end_idx: list = field(init=False, default_factory=list)
    _start_non_overlap: list = field(init=False, default_factory=list)
    _end_non_overlap: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self._calculate_idx()

    @classmethod
    def from_patch_size(cls, dim_size: int, patch_size: int, overlap: int = 0):
        num_patches = math.ceil((dim_size - overlap) / (patch_size - overlap))
        return cls(dim_size, patch_size, num_patches, overlap)

    @classmethod
    def from_num_patches(cls, dim_size: int, num_patches: int, overlap: int = 0):
        patch_size = math.ceil((dim_size - overlap) / num_patches) + overlap
        return cls(dim_size, patch_size, num_patches, overlap)

    @classmethod
    def from_num_size(cls, dim_size: int, num_patches: int, patch_size: int):
        total_size = num_patches * patch_size
        if total_size < dim_size:
            raise ValueError("values too small")
        overlap = (num_patches * patch_size - dim_size) // (num_patches - 1)
        return cls(dim_size, patch_size, num_patches, overlap)

    def _calculate_idx(self):
        self._start_idx = []
        self._end_idx = []
        self._local_start_idx = []
        self._local_end_idx = []
        self._start_non_overlap = []
        self._end_non_overlap = []
        for i in range(self._num_patches):
            start = i * (self._patch_size - self._overlap)
            end = min(start + self._patch_size, self._dim_size)
            self._start_idx.append(start)
            self._end_idx.append(end)

    @property
    def start_idx(self) -> list:
        return self._start_idx

    @property
    def end_idx(self) -> list:
        return self._end_idx

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def overlap(self) -> int:
        return self._overlap

    @property
    def dim_size(self) -> int:
        return self._dim_size

    def __call__(self) -> tuple[list, list]:
        return self._start_idx, self._end_idx

    def __len__(self) -> int:
        return self._num_patches

    def __get_item__(self, index: int) -> tuple[int, int]:
        return self._start_idx[index], self._end_idx[index]

    def __iter__(self):
        return zip(self._start_idx, self._end_idx)
