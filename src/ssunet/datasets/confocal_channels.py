"""Cache co-registered raw photon counts from every detector in a Leica series."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..confocal_inference import sha256_file, write_json
from .confocal import _metadata


def _copy_crop(image, source, shape, counts):
    """Copy every detector together and byte-check the order against readlif."""
    crop = source["crop_tzyx"]
    (t0, t1), (z0, z1), (y0, y1), (x0, x1) = crop
    volume_bytes = int(np.prod(shape[1:])) * image.channels
    histograms = np.zeros((image.channels, 256), dtype=np.int64)
    with Path(source["path"]).open("rb") as handle:
        for t in range(t0, t1):
            handle.seek(image.offsets[0] + t * volume_bytes)
            block = handle.read(volume_bytes)
            if len(block) != volume_bytes:
                message = "Truncated Leica volume while caching all detector channels"
                raise OSError(message)
            raw = np.frombuffer(block, dtype=np.uint8)
            if image.channel_as_second_dim:
                raw = raw.reshape(image.channels, *shape[1:])
            else:
                raw = raw.reshape(shape[1], image.channels, *shape[2:]).swapaxes(0, 1)
            selected = raw[:, z0:z1, y0:y1, x0:x1]
            counts[:, t - t0] = selected
            for channel in range(image.channels):
                histograms[channel] += np.bincount(selected[channel].ravel(), minlength=256)
                if t in {t0, (t0 + t1) // 2, t1 - 1}:
                    public = np.asarray(image.get_frame(t=t, z=z0, c=channel))[y0:y1, x0:x1]
                    if not np.array_equal(selected[channel, 0], public):
                        message = "Bulk channel order disagrees with readlif"
                        raise ValueError(message)
    return histograms


def prepare_all_channels(source, destination):
    """Copy one bounded CTZYX crop, retaining separate detector values and identities."""
    destination = Path(destination)
    image, identity = _metadata(source)
    shape = (image.dims.t, image.dims.z, image.dims.y, image.dims.x)
    crop = source["crop_tzyx"]
    if (
        image.display_dims != (1, 2)
        or image.n_mosaic != 1
        or image.offsets[1] != int(np.prod(shape)) * image.channels
        or len(crop) != 4
        or any(not 0 <= a < b <= size for (a, b), size in zip(crop, shape, strict=True))
    ):
        message = "All-channel caching requires a fitting uncompressed 8-bit XYZT crop"
        raise ValueError(message)
    target_shape = (image.channels, *(b - a for a, b in crop))
    manifest_path = destination.with_suffix(".json")
    if destination.exists() or manifest_path.exists():
        if not destination.exists() or not manifest_path.exists():
            message = "Incomplete all-channel cache; choose a fresh output directory"
            raise FileExistsError(message)
        saved = json.loads(manifest_path.read_text())
        values = np.load(destination, mmap_mode="r", allow_pickle=False)
        if (
            saved["identity"] != identity
            or list(values.shape) != list(target_shape)
            or values.dtype != np.uint8
            or saved["sha256"] != sha256_file(destination)
        ):
            message = "All-channel cache identity or content changed"
            raise ValueError(message)
        return values, saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.npy")
    counts = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=target_shape)
    histograms = _copy_crop(image, source, shape, counts)
    counts.flush()
    del counts
    partial.replace(destination)
    manifest = {
        "identity": identity,
        "shape_ctzyx": list(target_shape),
        "channel_indices": list(range(image.channels)),
        "channel_names": [item["Name"] for item in identity["active_detectors"]],
        "histograms": histograms.tolist(),
        "sha256": sha256_file(destination),
        "axis_order": "C,T,Z,Y,X",
        "all_channels": True,
    }
    write_json(manifest_path, manifest)
    return np.load(destination, mmap_mode="r", allow_pickle=False), manifest
