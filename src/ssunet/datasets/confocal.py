"""Cache selected Leica count planes locally and sample reproducible GAP patches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _metadata(source):
    """Read acquisition metadata without loading the full network-shared LIF."""
    from readlif.reader import LifFile

    path = Path(source["path"])
    lif = LifFile(path)
    image = lif.get_image(source["series_index"])
    node = next(
        element.find("./Data/Image")
        for element in lif.xml_root.iter("Element")
        if element.get("Name") == image.name and element.find("./Data/Image") is not None
    )
    detectors = [d.attrib for d in node.findall(".//Detector") if d.get("IsActive") == "1"]
    if (
        len(detectors) != image.channels
        or any(d.get("AcquisitionModeName") != "PhotonCounting" for d in detectors)
        or any(
            image.settings.get(key) != "1"
            for key in ("FrameAverage", "LineAverage", "FrameAccumulation", "Line_Accumulation")
        )
        or any(bits != 8 for bits in image.bit_depth)
    ):
        message = "This pilot requires unaveraged 8-bit PhotonCounting acquisitions"
        raise ValueError(message)
    identity = {
        "source": source,
        "series_name": image.name,
        "file_size_bytes": path.stat().st_size,
        "file_mtime_ns": path.stat().st_mtime_ns,
        "xml_header_sha256": hashlib.sha256(lif.xml_header.encode("utf-8")).hexdigest(),
        "active_detectors": detectors,
        "conversion": "identity_to_detected_photon_counts",
    }
    return image, identity


def prepare_counts(source: dict, destination: Path) -> tuple[Path, dict]:
    """Copy only the configured one-channel TZYX crop, preserving integer values."""
    image, identity = _metadata(source)
    crop = source["crop_tzyx"]
    bounds = (image.dims.t, image.dims.z, image.dims.y, image.dims.x)
    if (
        len(crop) != 4
        or any(len(pair) != 2 for pair in crop)
        or any(not 0 <= lo < hi <= size for (lo, hi), size in zip(crop, bounds, strict=True))
        or not 0 <= source["channel"] < image.channels
    ):
        message = "Source crop/channel is outside the acquisition"
        raise ValueError(message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = destination.with_suffix(".json")
    shape = tuple(hi - lo for lo, hi in crop)
    if destination.exists() or manifest.exists():
        if not destination.exists() or not manifest.exists():
            message = "Incomplete count cache; choose a fresh cache directory"
            raise FileExistsError(message)
        saved = json.loads(manifest.read_text())
        cached = np.load(destination, mmap_mode="r", allow_pickle=False)
        if saved["identity"] != identity or cached.shape != shape or cached.dtype != np.uint8:
            message = "Existing count cache differs from the requested source/crop"
            raise ValueError(message)
        if _sha256(destination) != saved["cache_sha256"]:
            message = "Cached counts have changed since preparation"
            raise ValueError(message)
        return destination, saved
    temporary = destination.with_suffix(".partial.npy")
    counts = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
    (t_begin, t_end), (z_begin, z_end), (y_begin, y_end), (x_begin, x_end) = crop
    histogram = np.zeros(256, dtype=np.int64)
    for t in range(t_begin, t_end):
        for z in range(z_begin, z_end):
            plane = np.asarray(image.get_frame(t=t, z=z, c=source["channel"]))
            selected = plane[y_begin:y_end, x_begin:x_end]
            counts[t - t_begin, z - z_begin] = selected
            histogram += np.bincount(selected.ravel(), minlength=256)
    counts.flush()
    del counts
    temporary.replace(destination)
    saved = {
        "identity": identity,
        "shape_tzyx": shape,
        "dtype": "uint8",
        "histogram": histogram.tolist(),
        "cache_sha256": _sha256(destination),
    }
    manifest.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return destination, saved


def _sha256(path):
    """Hash a small local crop rather than reading the complete source acquisition."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_patch_shape(patch_shape):
    """Require four positive TZYX extents, without silently rounding dimensions."""
    if len(patch_shape) != 4 or any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in patch_shape
    ):
        message = "patch_tzyx must contain four positive integers"
        raise ValueError(message)


def sample_split(counts, patch_shape, *, batch_size, p, seed, index, stream):
    """Draw uniform patches and thin each occurrence, with separate RNG streams.

    Validation uses fixed indices and its own stream. Training uses fresh indices.
    Every architecture receives identical locations and A/B draws at a given
    training index, irrespective of model initialization or global Torch RNG.
    """
    validate_patch_shape(patch_shape)
    if (
        counts.ndim != 4
        or counts.dtype.kind not in "ui"
        or any(want > have for want, have in zip(patch_shape, counts.shape, strict=True))
        or not 0 < p < 1
        or batch_size < 1
        or index < 0
        or stream < 0
        or (counts.dtype.kind == "i" and np.any(counts < 0))
    ):
        message = "Expected nonnegative TZYX counts, a fitting patch and valid sampling options"
        raise ValueError(message)
    generator = np.random.default_rng(np.random.SeedSequence([seed, stream, index]))
    patches = []
    for _ in range(batch_size):
        start = [
            int(generator.integers(have - want + 1))
            for have, want in zip(counts.shape, patch_shape, strict=True)
        ]
        slices = tuple(slice(lo, lo + size) for lo, size in zip(start, patch_shape, strict=True))
        patches.append(np.asarray(counts[slices], dtype=np.int64))
    full = np.stack(patches)[:, None]
    retained = generator.binomial(full, p)
    return torch.from_numpy(retained), torch.from_numpy(full - retained)
