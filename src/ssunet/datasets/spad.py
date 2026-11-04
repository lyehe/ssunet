"""SPAD single-photon dataset with GPU bit unpacking."""

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.random import choice, rand, randint
from PIL import Image
from torch.utils.data import Dataset

from ..configs.data_config import DataConfig
from ..configs.file_config import SplitParams
from ..configs.train_config import LoaderConfig
from ..constants import EPSILON, LOGGER


class NpyFileCache:
    """Thread-safe LRU cache for numpy files loaded into RAM.

    Caches full npy arrays keyed by file path. Evicts least-recently-used
    entries when the total cached size exceeds max_bytes.

    Args:
        max_bytes: maximum cache size in bytes. None = unlimited.
    """

    def __init__(self, max_bytes: int | None = None) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._current_bytes = 0
        self._max_bytes = max_bytes
        self._hits = 0
        self._misses = 0

    def load(self, path: Path | str) -> np.ndarray:
        """Load a numpy file, returning cached copy if available.

        Args:
            path: path to .npy file.

        Returns:
            Numpy array loaded into RAM.
        """
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]

        # Load outside lock to avoid blocking other threads during I/O
        arr = np.load(key)

        with self._lock:
            # Double-check after acquiring lock
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            nbytes = arr.nbytes
            # Evict LRU entries if over budget
            if self._max_bytes is not None:
                while self._current_bytes + nbytes > self._max_bytes and self._cache:
                    _, evicted = self._cache.popitem(last=False)
                    self._current_bytes -= evicted.nbytes

            self._cache[key] = arr
            self._current_bytes += nbytes
            self._misses += 1

        return arr

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "cached_files": len(self._cache),
                "cached_gb": self._current_bytes / (1024**3),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
            }

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0


# Module-level cache shared across all SPADScene instances in the same process.
# Each DataLoader worker process gets its own cache (filled on demand).
_FILE_CACHE = NpyFileCache()


def gpu_unpackbits(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack uint8 tensor to binary bits along specified dimension on GPU.

    Equivalent to np.unpackbits(arr, axis=dim). Each uint8 byte is expanded
    to 8 binary values (MSB first) along the given dimension.

    Args:
        packed: uint8 tensor to unpack.
        dim: dimension along which to unpack bits.

    Returns:
        Float32 tensor with the specified dimension expanded 8x.
    """
    if dim < 0:
        dim = packed.ndim + dim
    mask = (128 >> torch.arange(8, device=packed.device, dtype=torch.uint8)).reshape(
        [1] * packed.ndim + [8]
    )
    unpacked = (packed.unsqueeze(-1) & mask).ne(0).to(torch.float32)
    # Merge bit dimension (last) into the target dimension
    perm = list(range(unpacked.ndim))
    perm.remove(unpacked.ndim - 1)
    perm.insert(dim + 1, unpacked.ndim - 1)
    unpacked = unpacked.permute(perm).contiguous()
    shape = list(unpacked.shape)
    shape[dim] *= shape[dim + 1]
    del shape[dim + 1]
    return unpacked.reshape(shape)


def set_file_cache(max_gb: float | None = None) -> None:
    """Configure the global file cache size.

    Args:
        max_gb: maximum cache size in GB. None = unlimited (cache everything).
    """
    global _FILE_CACHE  # noqa: PLW0603
    _FILE_CACHE = NpyFileCache(max_bytes=int(max_gb * (1024**3)) if max_gb is not None else None)


class SPADScene:
    """Manages data for a single SPAD scene with LRU RAM caching.

    Files are loaded fully into RAM on first access via the module-level
    NpyFileCache, avoiding mmap page-fault overhead for random access.
    """

    def __init__(self, scene_dir: Path) -> None:
        """Load scene metadata.

        Args:
            scene_dir: path to scene directory containing .npy files.
        """
        self.scene_dir = scene_dir
        self.name = scene_dir.name
        self.npy_paths = sorted(scene_dir.glob("*.npy"))
        self.n_files = len(self.npy_paths)

        # Read dimensions from first file (quick mmap peek, not cached)
        first = np.load(str(self.npy_paths[0]), mmap_mode="r")
        self.frames_per_file = first.shape[0]  # T per file (1024)
        self.height = first.shape[1]  # H (800)
        self.w_packed = first.shape[2]  # W packed (100)
        self.channels = first.shape[3]  # C (3)
        self.width = self.w_packed * 8  # W unpacked (800)
        self.total_frames = self.n_files * self.frames_per_file

    def _load_file(self, file_idx: int) -> np.ndarray:
        """Load a file via the LRU cache.

        Args:
            file_idx: index into self.npy_paths.

        Returns:
            Numpy array of shape (frames_per_file, H, W_packed, C).
        """
        return _FILE_CACHE.load(self.npy_paths[file_idx])

    def get_packed_patch(
        self, t_start: int, t_size: int, h_start: int, h_size: int
    ) -> np.ndarray:
        """Extract a bit-packed patch, handling cross-file boundaries.

        Args:
            t_start: starting frame index.
            t_size: number of frames to extract.
            h_start: starting row index.
            h_size: number of rows to extract.

        Returns:
            Bit-packed numpy array of shape (t_size, h_size, w_packed, channels).
        """
        t_end = t_start + t_size
        file_start = t_start // self.frames_per_file
        file_end = (t_end - 1) // self.frames_per_file

        if file_start == file_end:
            data = self._load_file(file_start)
            offset = t_start % self.frames_per_file
            return data[offset : offset + t_size, h_start : h_start + h_size, :, :]

        parts = []
        for fi in range(file_start, file_end + 1):
            data = self._load_file(fi)
            local_start = t_start % self.frames_per_file if fi == file_start else 0
            local_end = t_end % self.frames_per_file if fi == file_end else self.frames_per_file
            if local_end == 0:
                local_end = self.frames_per_file
            parts.append(data[local_start:local_end, h_start : h_start + h_size, :, :])
        return np.concatenate(parts, axis=0)


class SPADPackedDataset(Dataset):
    """Dataset returning bit-packed SPAD patches from a single scene.

    Each sample is a uint8 tensor of shape (z_size, xy_size, w_packed, channels)
    with random temporal and spatial cropping. Bit unpacking, W-axis cropping,
    and binomial splitting happen on GPU via SPADDataModule.on_after_batch_transfer.
    """

    def __init__(self, scene_dir: str | Path, config: DataConfig) -> None:
        """Initialize the dataset for a single scene.

        Args:
            scene_dir: path to a scene directory containing .npy files.
            config: data configuration (xy_size, z_size, virtual_size, etc.).
        """
        self.config = config
        self.scene = SPADScene(Path(scene_dir))
        LOGGER.info(
            "Loaded scene '%s': %d files, %d frames",
            self.scene.name,
            self.scene.n_files,
            self.scene.total_frames,
        )

    def __len__(self) -> int:
        """Get dataset length."""
        if self.config.virtual_size > 0:
            return self.config.virtual_size
        return self.scene.total_frames - self.config.z_size

    def __getitem__(self, index: int) -> torch.Tensor:
        """Sample a random bit-packed patch.

        Args:
            index: dataset index (used for length, actual sampling is random).

        Returns:
            Bit-packed uint8 tensor of shape (z_size, xy_size, w_packed, channels).
        """
        t_max = self.scene.total_frames - self.config.z_size
        t_start = randint(0, max(1, t_max))

        h_max = self.scene.height - self.config.xy_size
        h_start = randint(0, max(1, h_max)) if h_max > 0 else 0

        packed = self.scene.get_packed_patch(
            t_start, self.config.z_size, h_start, self.config.xy_size
        )
        return torch.from_numpy(packed)


class SPADValDataset(Dataset):
    """Fixed packed validation windows with optional PNG image references."""

    def __init__(self, scene_dir: str | Path, z_size: int = 32) -> None:
        self.scene_dir = Path(scene_dir)
        self.npy_paths = sorted(self.scene_dir.glob("*.npy"))
        self.png_paths = sorted(self.scene_dir.glob("*.png"))
        # Photon validation does not require a clean image reference.
        self.pairs = []
        for npy_path in self.npy_paths:
            png_path = npy_path.with_suffix(".png")
            self.pairs.append((npy_path, png_path if png_path.exists() else None))
        self.z_size = z_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        npy_path, png_path = self.pairs[idx]
        # Load and accumulate npy frames → noisy image
        npy = _FILE_CACHE.load(npy_path)  # (T, H, W_packed, C)
        # Take z_size frames from the middle for model input (bit-packed)
        t_mid = npy.shape[0] // 2
        t_start = max(0, t_mid - self.z_size // 2)
        packed_patch = npy[t_start : t_start + self.z_size, :, :, :]  # (Z, H, W_packed, C)
        result = {
            "packed": torch.from_numpy(packed_patch),
            "sample_index": idx,
        }
        if png_path is not None:
            with Image.open(png_path) as image:
                png = np.array(image).astype(np.float32) / 255.0
            result["gt"] = torch.from_numpy(png)
        return result


def _shuffle_1x1x8(data: torch.Tensor) -> torch.Tensor:
    """Randomly permute groups of 8 consecutive frames per-pixel along Z.

    Groups Z into blocks of 8 and applies an independent random permutation
    at each (B, C, group, H, W) position. With period-4 correlation, each
    group of 8 contains 2 copies of each residue class — shuffling mixes
    them so the model can't learn the fixed temporal ordering.

    Args:
        data: tensor of shape (B, C, Z, H, W), Z must be divisible by 8.

    Returns:
        Shuffled tensor of same shape.
    """
    B, C, Z, H, W = data.shape
    G = Z // 8

    # Reshape to (B, C, G, 8, H, W)
    data = data.reshape(B, C, G, 8, H, W)

    # Per-pixel random permutation via argsort of uniform random
    rand_keys = torch.rand(B, C, G, 8, H, W, device=data.device)
    perm = rand_keys.argsort(dim=3)

    data = data.gather(3, perm)

    return data.reshape(B, C, Z, H, W)


def _cross_group_split(data: torch.Tensor, num_workers: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Split data by t%4 RNG groups to guarantee independent noise/target pairs.

    Due to the RNG bug, DataLoader workers 0-3 each used the same threshold
    map for all their frames.  Within a 32-frame window, frame positions
    {0,4,8,...,28}, {1,5,9,...,29}, {2,6,10,...,30}, {3,7,11,...,31} each form
    a "group" of 8 identical frames (same threshold).  There are only 4
    independent threshold samples across the whole window.

    Strategy:
      1. Randomly partition the 4 residue groups into 2 input groups and
         2 target groups (there are C(4,2)=6 possible splits, chosen uniformly).
      2. For each partition, average the 8 frames within each group to get
         the group's representative value (this is just the value itself for
         binary data, but averaging is correct in general and handles any
         minor variations).
      3. Sum the 2 input-group averages -> input signal (rescaled to match
         expected photon count). Sum the 2 target-group averages -> target.
      4. Tile each 2-frame summary back to 32 frames so the 3D U-Net receives
         its expected (B, C, 32, H, W) shape.

    Because input and target come from DIFFERENT threshold maps, they are
    statistically independent conditioned on the true scene intensity p.
    This is exactly the independence requirement for self-supervised denoising.

    Args:
        data: tensor of shape (B, C, Z, H, W) where Z=32 (or any multiple of
              num_workers). Binary float32 values (0 or 1).
        num_workers: number of DataLoader workers that caused the RNG bug (4).

    Returns:
        (target, noise) each of shape (B, C, Z, H, W).
    """
    B, C, Z, H, W = data.shape
    frames_per_group = Z // num_workers  # 32 // 4 = 8

    # Reshape to separate the 4 residue groups:
    # data[:, :, i, :, :] has residue (i % 4). Reshape to (B, C, 8, 4, H, W)
    # by deinterleaving: group g gets frames [g, g+4, g+8, ..., g+28]
    data_groups = data.reshape(B, C, frames_per_group, num_workers, H, W)
    # data_groups[:, :, :, g, :, :] = all frames from residue group g

    # Average across the 8 identical frames per group -> (B, C, 1, 4, H, W)
    # For truly identical binary frames this is the same value, but averaging
    # is robust to minor deviations and generalizes correctly.
    group_means = data_groups.mean(dim=2, keepdim=True)  # (B, C, 1, 4, H, W)

    # Randomly select 2 of 4 groups for input, remaining 2 for target.
    # Generate a random permutation of [0,1,2,3] and take first 2 / last 2.
    perm = torch.randperm(num_workers, device=data.device)
    input_groups = perm[:2]
    target_groups = perm[2:]

    # Extract and sum the group means for input and target
    # group_means shape: (B, C, 1, 4, H, W)
    input_signal = group_means[:, :, :, input_groups, :, :].sum(dim=3)   # (B, C, 1, H, W)
    target_signal = group_means[:, :, :, target_groups, :, :].sum(dim=3)  # (B, C, 1, H, W)

    # Tile back to Z=32 frames for the 3D U-Net
    noise = input_signal.expand(B, C, Z, H, W)
    target = target_signal.expand(B, C, Z, H, W)

    return target, noise


def _cross_group_split_interleaved(
    data: torch.Tensor, num_workers: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-group split preserving temporal structure for 3D convolutions.

    Like _cross_group_split, but instead of collapsing to a single frame and
    tiling, this version keeps the interleaved temporal structure.  Input and
    target each get 16 of the 32 frames (the 2 assigned groups x 8 frames
    each), placed at their ORIGINAL temporal positions.  The other 16 positions
    are filled by repeating the nearest same-group frame (zero-order hold)
    so the 3D convolutions see smooth temporal transitions.

    This preserves more temporal information for the 3D U-Net while still
    guaranteeing independence between input and target.

    Args:
        data: (B, C, Z, H, W), Z must be divisible by num_workers.
        num_workers: number of RNG-correlated worker groups (4).

    Returns:
        (target, noise) each of shape (B, C, Z, H, W).
    """
    B, C, Z, H, W = data.shape

    # Random 2-vs-2 split of the 4 groups
    perm = torch.randperm(num_workers, device=data.device)
    input_groups = perm[:2]
    target_groups = perm[2:]

    # Build group assignment for each frame position: frame i belongs to group (i % 4)
    frame_groups = torch.arange(Z, device=data.device) % num_workers  # (Z,)

    # Build masks: which frames belong to input vs target
    input_mask = (frame_groups.unsqueeze(0) == input_groups.unsqueeze(1)).any(dim=0)  # (Z,)
    target_mask = ~input_mask  # (Z,)

    # Reshape mask to broadcast: (1, 1, Z, 1, 1)
    input_mask_5d = input_mask.reshape(1, 1, Z, 1, 1).float()
    target_mask_5d = target_mask.reshape(1, 1, Z, 1, 1).float()

    # For the input tensor: keep input-group frames as-is, fill target-group
    # positions with the average of adjacent input-group frames (interpolation).
    # For binary SPAD data, simply repeating the nearest input-group frame works.
    # Use gather-based nearest-neighbor fill along Z.
    input_indices = torch.where(input_mask)[0]   # positions of input frames
    target_indices = torch.where(target_mask)[0]  # positions of target frames

    # Build nearest-neighbor lookup: for each of 32 positions, find the
    # nearest frame index that belongs to the desired group set.
    def _nearest_fill_indices(owned_positions: torch.Tensor, total: int) -> torch.Tensor:
        """For each position 0..total-1, find the nearest owned position."""
        all_pos = torch.arange(total, device=owned_positions.device)
        # Distances from each position to each owned position
        dists = (all_pos.unsqueeze(1) - owned_positions.unsqueeze(0)).abs()  # (total, n_owned)
        nearest_owned_idx = dists.argmin(dim=1)  # index into owned_positions
        return owned_positions[nearest_owned_idx]  # actual frame index

    noise_fill = _nearest_fill_indices(input_indices, Z)   # (Z,) frame indices
    target_fill = _nearest_fill_indices(target_indices, Z)  # (Z,) frame indices

    # Gather along Z dimension
    noise_fill_5d = noise_fill.reshape(1, 1, Z, 1, 1).expand(B, C, Z, H, W)
    target_fill_5d = target_fill.reshape(1, 1, Z, 1, 1).expand(B, C, Z, H, W)

    noise = data.gather(2, noise_fill_5d)
    target = data.gather(2, target_fill_5d)

    return target, noise


class SPADUniformDataset(Dataset):
    """Dataset for preprocessed uniform-sampled SPAD data (single .npy file).

    The .npy contains N_chunks * frames_per_chunk frames, bit-packed.
    Each group of `frames_per_chunk` consecutive frames is one chunk.
    Samples z_size consecutive frames from a random chunk with random H crop.
    """

    def __init__(
        self, npy_path: str | Path, config: DataConfig, frames_per_chunk: int = 41,
    ) -> None:
        self.config = config
        self.frames_per_chunk = frames_per_chunk
        self.data = np.load(str(npy_path))  # (N, H, W_packed, C)
        self.n_chunks = self.data.shape[0] // frames_per_chunk
        LOGGER.info(
            "Loaded uniform data '%s': %d chunks x %d frames, shape %s",
            Path(npy_path).name, self.n_chunks, frames_per_chunk, self.data.shape,
        )

    def __len__(self) -> int:
        if self.config.virtual_size > 0:
            return self.config.virtual_size
        return self.n_chunks

    def __getitem__(self, index: int) -> torch.Tensor:
        chunk_idx = randint(0, self.n_chunks)
        chunk_start = chunk_idx * self.frames_per_chunk

        # Random z_size window within the chunk's frames
        z_max = self.frames_per_chunk - self.config.z_size
        z_start = randint(0, max(1, z_max))

        # Random H crop
        h_max = self.data.shape[1] - self.config.xy_size
        h_start = randint(0, max(1, h_max)) if h_max > 0 else 0

        t_start = chunk_start + z_start
        patch = self.data[
            t_start : t_start + self.config.z_size,
            h_start : h_start + self.config.xy_size,
            :, :,
        ]
        return torch.from_numpy(patch)


class SPADMultiOffsetDataset(Dataset):
    """Dataset that randomly selects from all 25 offset .npy files each sample.

    Each offset file contains the same chunks sampled at a different phase
    of the step-25 grid. Randomly picking an offset before each sample
    provides 25x data augmentation and ensures the model sees all frames.
    """

    def __init__(
        self, offset_dir: str | Path, config: DataConfig, frames_per_chunk: int = 41,
        n_offsets: int = 25,
    ) -> None:
        self.config = config
        self.frames_per_chunk = frames_per_chunk
        self.n_offsets = n_offsets

        offset_dir = Path(offset_dir)
        self.all_data = []
        for off in range(n_offsets):
            path = offset_dir / f"off{off:02d}.npy"
            data = np.load(str(path))
            self.all_data.append(data)
            if off == 0:
                LOGGER.info(
                    "Loaded offset dir '%s': %d offsets, shape %s each",
                    offset_dir.name, n_offsets, data.shape,
                )

        self.n_chunks = self.all_data[0].shape[0] // frames_per_chunk

    def __len__(self) -> int:
        if self.config.virtual_size > 0:
            return self.config.virtual_size
        return self.n_chunks * self.n_offsets

    def __getitem__(self, index: int) -> torch.Tensor:
        # Randomly pick an offset
        off_idx = randint(0, self.n_offsets)
        data = self.all_data[off_idx]

        # Random chunk
        chunk_idx = randint(0, self.n_chunks)
        chunk_start = chunk_idx * self.frames_per_chunk

        # Random z_size window within the chunk
        z_max = self.frames_per_chunk - self.config.z_size
        z_start = randint(0, max(1, z_max))

        # Random H crop
        h_max = data.shape[1] - self.config.xy_size
        h_start = randint(0, max(1, h_max)) if h_max > 0 else 0

        t_start = chunk_start + z_start
        patch = data[
            t_start : t_start + self.config.z_size,
            h_start : h_start + self.config.xy_size,
            :, :,
        ]
        return torch.from_numpy(patch)


class SPADUniformDataModule(pl.LightningDataModule):
    """DataModule for preprocessed uniform-sampled SPAD data.

    Uses standard binomial splitting (not cross-group) since frames are
    already independent after uniform step-25 sampling.

    Args:
        npy_path: path to a single .npy file OR a directory containing off00..off24.npy.
            If a directory, uses SPADMultiOffsetDataset (randomly selects offset each sample).
            If a file, uses SPADUniformDataset (single offset).
    """

    def __init__(
        self,
        npy_path: str | Path,
        data_config: DataConfig,
        split_params: SplitParams,
        loader_config: LoaderConfig,
        frames_per_chunk: int = 41,
    ) -> None:
        super().__init__()
        self.npy_path = Path(npy_path)
        self.data_config = data_config
        self.split_params = split_params
        self.loader_config = loader_config
        self.frames_per_chunk = frames_per_chunk

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            if self.npy_path.is_dir():
                self.train_dataset = SPADMultiOffsetDataset(
                    self.npy_path, self.data_config, self.frames_per_chunk,
                )
            else:
                self.train_dataset = SPADUniformDataset(
                    self.npy_path, self.data_config, self.frames_per_chunk,
                )
            self._has_val = False

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.loader_config.loader(self.train_dataset)

    def val_dataloader(self) -> list[torch.utils.data.DataLoader]:
        return []

    def on_after_batch_transfer(self, batch: torch.Tensor, dataloader_idx: int) -> list:
        packed = batch  # (B, Z, H, W_packed, C) uint8

        # Unpack bits along W dimension
        unpacked = gpu_unpackbits(packed, dim=3)  # (B, Z, H, 800, C)

        # Random W crop
        w_size = self.data_config.xy_size
        w_max = unpacked.shape[3] - w_size
        if w_max > 0:
            w_start = torch.randint(0, w_max + 1, (1,), device=unpacked.device).item()
            unpacked = unpacked[:, :, :, w_start : w_start + w_size, :]

        data = unpacked.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, Z, H, W)

        # Augmentation
        if dataloader_idx == 0 and self.data_config.augments:
            if torch.rand(1).item() > 0.5:
                data = data.transpose(-1, -2)
            if torch.rand(1).item() > 0.5:
                data = data.flip(-1)
            if torch.rand(1).item() > 0.5:
                data = data.flip(-2)

        # Standard binomial splitting (frames are already independent)
        p = self._sample_p(data)
        noise = data * (torch.rand_like(data) < p).float()
        target = data - noise

        if self.split_params.normalize_target:
            target = target / (target.mean() + EPSILON)

        return [target, noise]

    def _sample_p(self, data: torch.Tensor | None = None) -> float:
        method = self.split_params.method
        min_p = min(self.split_params.min_p, self.split_params.max_p)
        max_p = max(self.split_params.min_p, self.split_params.max_p)

        if method == "signal":
            return float(rand() * (max_p - min_p) + min_p)
        if method == "fixed":
            return float(self.split_params.min_p)
        if method == "list" and self.split_params.p_list:
            return float(choice(self.split_params.p_list))
        if method == "db" and data is not None:
            uniform = rand() * (max_p - min_p) + min_p
            p = (10 ** (uniform / 10.0)) / (data.float().mean().item() + EPSILON)
            return float(max(EPSILON, min(1 - EPSILON, p)))
        return float(rand() * (max_p - min_p) + min_p)


class SPADDataModule(pl.LightningDataModule):
    """Lightning DataModule for SPAD data with GPU bit unpacking and binomial splitting.

    Pipeline per batch:
    1. DataLoader yields bit-packed uint8 tensors from CPU/RAM
    2. Lightning transfers batch to GPU
    3. on_after_batch_transfer: unpack bits, crop W, augment, binomial split
    4. Model receives [target, noise] tensors on GPU
    """

    def __init__(
        self,
        scene_dir: str | Path,
        data_config: DataConfig,
        split_params: SplitParams,
        loader_config: LoaderConfig,
    ) -> None:
        """Initialize the SPAD data module for a single scene.

        Args:
            scene_dir: path to scene directory containing .npy files.
            data_config: data configuration.
            split_params: binomial splitting parameters.
            loader_config: dataloader configuration.
        """
        super().__init__()
        self.scene_dir = scene_dir
        self.data_config = data_config
        self.split_params = split_params
        self.loader_config = loader_config

    def setup(self, stage: str | None = None) -> None:
        """Set up dataset."""
        if stage in ("fit", None):
            self.train_dataset = SPADPackedDataset(self.scene_dir, self.data_config)
        if stage in ("fit", "validate", None):
            self.val_dataset = SPADValDataset(
                self.scene_dir, z_size=self.data_config.z_size
            )
            self._has_val = bool(len(self.val_dataset))

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create training dataloader."""
        return self.loader_config.loader(self.train_dataset)

    def val_dataloader(self) -> list[torch.utils.data.DataLoader]:
        """Create validation dataloader without requiring PNG references."""
        if not getattr(self, "_has_val", False):
            return []
        return [torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=0,
        )]

    def on_after_batch_transfer(self, batch: torch.Tensor, dataloader_idx: int) -> list:
        """Unpack bits on GPU, crop W, augment, and apply binomial split.

        Args:
            batch: bit-packed uint8 tensor (B, Z, H, W_packed, C) on GPU.
            dataloader_idx: 0 for training, 1+ for validation.

        Returns:
            [target, noise] tensors of shape (B, C, Z, H, W) on GPU.
        """
        # Validation batches have packed photons and stable sample identifiers.
        if isinstance(batch, dict):
            packed = batch["packed"]  # (B, Z, H, W_packed, C)
            gt = batch.get("gt")  # Optional (B, H, W, C) image reference.
            unpacked = gpu_unpackbits(packed, dim=3)  # (B, Z, H, 800, C)
            unpacked = unpacked[:, :, :, :800, :]
            # Center crop to xy_size for model compatibility
            s = self.data_config.xy_size
            if s > unpacked.shape[2] or s > unpacked.shape[3]:
                message = "SPAD validation crop must fit within the observation"
                raise ValueError(message)
            h0 = (unpacked.shape[2] - s) // 2
            w0 = (unpacked.shape[3] - s) // 2
            unpacked = unpacked[:, :, h0 : h0 + s, w0 : w0 + s, :]
            data = unpacked.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, Z, H, W)
            # Fixed held-out splitting keeps validation repeatable without
            # altering the training generators or its experimental group split.
            indices = batch.get("sample_index", range(len(data)))
            seed = self.split_params.seed if self.split_params.seed is not None else 0
            noise_parts = []
            for index, sample in zip(indices, data, strict=True):
                generator = torch.Generator(device=sample.device)
                generator.manual_seed((int(seed) + int(index)) % (2**63))
                split = torch.rand(sample.shape, generator=generator, device=sample.device) < 0.5
                noise_parts.append(sample * split)
            noise = torch.stack(noise_parts)
            target = data - noise
            if self.split_params.normalize_target:
                target = target / (target.mean(dim=(1, 2, 3, 4), keepdim=True) + EPSILON)
            result = [target, noise]
            if gt is not None:
                gt_crop = gt[:, h0 : h0 + s, w0 : w0 + s, :]
                # A single image reference is optional and separate from the target.
                result.append(gt_crop.permute(0, 3, 1, 2).unsqueeze(2))
            return result

        packed = batch  # (B, Z, H, W_packed, C) uint8

        # Unpack bits along W dimension (dim=3)
        unpacked = gpu_unpackbits(packed, dim=3)  # (B, Z, H, 800, C) float32

        # Random W crop (uniform across batch for GPU efficiency)
        w_size = self.data_config.xy_size
        w_max = unpacked.shape[3] - w_size
        if w_max > 0:
            w_start = torch.randint(0, w_max + 1, (1,), device=unpacked.device).item()
            unpacked = unpacked[:, :, :, w_start : w_start + w_size, :]

        # Permute to (B, C, Z, H, W)
        data = unpacked.permute(0, 4, 1, 2, 3).contiguous()

        # GPU augmentation (training only)
        if dataloader_idx == 0 and self.data_config.augments:
            if torch.rand(1).item() > 0.5:
                data = data.transpose(-1, -2)
            if torch.rand(1).item() > 0.5:
                data = data.flip(-1)
            if torch.rand(1).item() > 0.5:
                data = data.flip(-2)

        # Cross-group split: assign 2 of 4 RNG-correlated worker groups to
        # input and 2 to target. Guarantees statistical independence between
        # noise and target despite the period-4 threshold corruption.
        target, noise = _cross_group_split_interleaved(data, num_workers=4)

        if self.split_params.normalize_target:
            target = target / (target.mean() + EPSILON)

        return [target, noise]

    def _sample_p(self, data: torch.Tensor | None = None) -> float:
        """Sample a p-value for binomial splitting.

        Args:
            data: input tensor (used for db method).

        Returns:
            p-value in (0, 1).
        """
        method = self.split_params.method
        min_p = min(self.split_params.min_p, self.split_params.max_p)
        max_p = max(self.split_params.min_p, self.split_params.max_p)

        if method == "signal":
            return float(rand() * (max_p - min_p) + min_p)
        if method == "fixed":
            return float(self.split_params.min_p)
        if method == "list" and self.split_params.p_list:
            return float(choice(self.split_params.p_list))
        if method == "db" and data is not None:
            uniform = rand() * (max_p - min_p) + min_p
            p = (10 ** (uniform / 10.0)) / (data.float().mean().item() + EPSILON)
            return float(max(EPSILON, min(1 - EPSILON, p)))

        return float(rand() * (max_p - min_p) + min_p)
