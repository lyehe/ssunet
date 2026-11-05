"""Count-calibrated, overlapping inference for large confocal XYZT acquisitions."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .datasets.confocal import _metadata
from .softmax_fix import fixed_log_softmax


def write_json(path, value):
    """Publish a complete progress document through an atomic rename."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    """Hash a file with bounded host memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tile_starts(size, patch, overlap):
    """Cover an axis completely, anchoring the last tile at its real boundary."""
    if size < 1 or patch < 1 or not 0 <= overlap < patch:
        message = "Require positive sizes and 0 <= overlap < patch"
        raise ValueError(message)
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, patch - overlap))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def spatial_weights(shape):
    """Use a separable Hann taper with strictly positive weights at every voxel."""
    z, y, x = [np.hanning(n + 2)[1:-1].astype(np.float32) for n in shape]
    return z[:, None, None] * y[None, :, None] * x[None, None, :]


def photon_half(counts, *, first_time, seed):
    """Produce one acquisition-wide thinning, independent of temporal chunk grouping."""
    counts = np.asarray(counts)
    if counts.ndim != 4 or counts.dtype != np.uint8 or first_time < 0:
        message = "Expected unmodified uint8 TZYX photon counts"
        raise ValueError(message)
    retained = np.empty_like(counts)
    for t, frame in enumerate(counts):
        generator = np.random.default_rng(np.random.SeedSequence([seed, first_time + t]))
        retained[t] = generator.binomial(frame, 0.5).astype(np.uint8)
    return retained


def _raw_layout(image):
    shape = (image.dims.t, image.dims.z, image.dims.y, image.dims.x)
    length = int(np.prod(shape)) * image.channels
    if (
        image.display_dims != (1, 2)
        or image.n_mosaic != 1
        or tuple(image.bit_depth) != (8,) * image.channels
        or image.offsets[1] != length
    ):
        message = "Bulk reading requires an unpacked 8-bit, non-mosaic XYZT LIF series"
        raise ValueError(message)
    return shape


def _volume_from_bytes(block, image, channel):
    z, y, x = image.dims.z, image.dims.y, image.dims.x
    raw = np.frombuffer(block, dtype=np.uint8)
    if image.channel_as_second_dim:
        raw = raw.reshape(image.channels, z, y, x)[channel]
    else:
        raw = raw.reshape(z, image.channels, y, x)[:, channel]
    return np.ascontiguousarray(raw)


def describe_source(source):
    """Read physical spacing and metadata without altering the source acquisition."""
    image, identity = _metadata(source)
    shape = _raw_layout(image)
    spacing = [1 / float(image.scale[i]) for i in (2, 1, 0)]
    return image, {
        "identity": identity,
        "shape_tzyx": list(shape),
        "voxel_spacing_zyx_um": [abs(value) for value in spacing],
        "reverse_z_for_display": spacing[0] < 0,
        "volume_interval_seconds": 1 / float(image.scale[3]),
    }


def prepare_full_counts(source, output, *, on_progress=None):
    """Stream full single-channel volumes to a local cache; verify against readlif.

    Bulk reads include both detector channels in one sequential network read,
    retaining only the requested detector. Three distributed planes are checked
    against readlif before copying. Source metadata and complete cache hashes
    prevent reuse of another acquisition or an incomplete cache.
    """
    output = Path(output)
    image, description = describe_source(source)
    shape = tuple(description["shape_tzyx"])
    path = output / "counts.npy"
    cache_path = output / "counts_manifest.json"
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved["description"] != description:
            message = "Count cache belongs to a different source acquisition"
            raise ValueError(message)
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            cached.shape != shape
            or cached.dtype != np.uint8
            or sha256_file(path) != saved["sha256"]
        ):
            message = "Full count cache has changed"
            raise ValueError(message)
        return cached, description
    output.mkdir(parents=True, exist_ok=True)
    volume_bytes = int(np.prod(shape[1:])) * image.channels
    channel = source["channel"]
    with Path(source["path"]).open("rb") as handle:
        for t, z in ((0, 0), (shape[0] // 2, shape[1] // 2), (shape[0] - 1, shape[1] - 1)):
            handle.seek(image.offsets[0] + t * volume_bytes)
            block = handle.read(volume_bytes)
            values = _volume_from_bytes(block, image, channel)
            if not np.array_equal(values[z], np.asarray(image.get_frame(t=t, z=z, c=channel))):
                message = "Bulk LIF layout disagrees with readlif"
                raise ValueError(message)
        partial = path.with_suffix(".partial.npy")
        counts = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=shape)
        histogram = np.zeros(256, dtype=np.int64)
        for t in range(shape[0]):
            handle.seek(image.offsets[0] + t * volume_bytes)
            block = handle.read(volume_bytes)
            if len(block) != volume_bytes:
                message = "Incomplete LIF read"
                raise OSError(message)
            values = _volume_from_bytes(block, image, channel)
            counts[t] = values
            histogram += np.bincount(values.ravel(), minlength=256)
            if on_progress is not None and (t % 16 == 0 or t + 1 == shape[0]):
                on_progress(t + 1, shape[0])
        counts.flush()
        del counts
    partial.replace(path)
    write_json(
        cache_path,
        {
            "description": description,
            "sha256": sha256_file(path),
            "histogram": histogram.tolist(),
            "readlif_planes_verified": 3,
        },
    )
    return np.load(path, mmap_mode="r", allow_pickle=False), description


@torch.inference_mode()
def _predict_tile(model, counts, retained, core_slice, device, precision, microbatch_size):
    """Average complementary conditional distributions at the full count scale."""
    mass = int(counts.sum(dtype=np.uint64))
    shape = (len(np.arange(counts.shape[0])[core_slice]), *counts.shape[1:])
    expected = np.zeros(shape, dtype=np.float32)
    if not mass:
        return expected
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    halves = (retained, counts - retained)
    for first in range(0, 2, microbatch_size):
        values = np.stack(halves[first : first + microbatch_size])[:, None]
        inputs = torch.from_numpy(values).to(device)
        context = torch.autocast(device_type="cuda", dtype=dtype) if dtype else nullcontext()
        with context:
            logits = model(inputs)
        q = fixed_log_softmax(logits.float().flatten(1)).exp().reshape_as(logits)
        expected += (q[:, 0, core_slice].sum(0) * (mass / 2)).cpu().numpy()
    if not np.isfinite(expected).all() or np.any(expected < 0):
        message = "Nonfinite or negative reconstructed intensity"
        raise ValueError(message)
    return expected


def _validate_reconstruction(observed, retained, precision, microbatch_size):
    """Reject altered count units and unsupported precision or batching."""
    if observed.shape != retained.shape or observed.ndim != 4:
        message = "Expected matching TZYX full counts and retained photons"
        raise ValueError(message)
    if observed.dtype != np.uint8 or retained.dtype != np.uint8 or np.any(retained > observed):
        message = "Retained counts must be a valid thinning of the observation"
        raise ValueError(message)
    if precision not in {"fp32", "fp16", "bf16"} or microbatch_size not in (1, 2):
        message = "Unsupported inference precision or complementary-half microbatch size"
        raise ValueError(message)


@torch.inference_mode()
def reconstruct_core(
    model,
    observed,
    retained,
    *,
    core_slice,
    patch_zyx,
    overlap_zyx,
    device,
    precision="fp32",
    microbatch_size=1,
    on_tile=None,
):
    """Blend count-calibrated predictions from complementary photon halves.

    Each half has the same expected count level as the training inputs. Both
    halves contribute equally to the predicted location distribution; multiplying
    by the observed patch total estimates its full-exposure count scale. The
    final whole-acquisition distribution is normalized separately after blending.
    No independent detector, temporal average or clean target enters inference.
    """
    observed, retained = np.asarray(observed), np.asarray(retained)
    _validate_reconstruction(observed, retained, precision, microbatch_size)
    temporal = np.arange(observed.shape[0])[core_slice]
    if temporal.size == 0:
        message = "The output core must contain at least one real time point"
        raise ValueError(message)
    spatial = observed.shape[1:]
    starts = [tile_starts(n, p, o) for n, p, o in zip(spatial, patch_zyx, overlap_zyx, strict=True)]
    result = np.zeros((temporal.size, *spatial), dtype=np.float32)
    coverage = np.zeros(spatial, dtype=np.float32)
    total_tiles = int(np.prod([len(axis) for axis in starts]))
    tile_index = 0
    model.eval()
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                slices = tuple(
                    slice(a, min(a + p, n))
                    for a, p, n in zip((z, y, x), patch_zyx, spatial, strict=True)
                )
                counts = observed[(slice(None), *slices)]
                a = retained[(slice(None), *slices)]
                weight = spatial_weights(counts.shape[1:])
                expected = _predict_tile(
                    model, counts, a, core_slice, device, precision, microbatch_size
                )
                result[(slice(None), *slices)] += expected * weight
                coverage[slices] += weight
                tile_index += 1
                if on_tile is not None:
                    on_tile(tile_index, total_tiles)
    if np.any(coverage <= 0):
        message = "Inference tiling left uncovered voxels"
        raise ValueError(message)
    result /= coverage
    return result
