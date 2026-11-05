"""Apply the same two-channel 4D checkpoint to continuous clips from both recordings."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from ..ssunet.confocal_inference import describe_source, sha256_file, write_json
from ..ssunet.confocal_motion import (
    normalized_motion_probabilities,
    split_motion_pool,
    temporal_control,
    temporal_logits,
)
from ..ssunet.confocal_trial import prepare_training_pool
from ..ssunet.datasets.confocal_channels import prepare_all_channels
from ..ssunet.models.confocal_multichannel import MultichannelConfocalGAP
from .confocal_joint_train import source_hashes

LOGGER = logging.getLogger(__name__)


def prepare_recording(name, record, root):
    """Keep both detectors registered and retain the original permanent photon reserve."""
    if "cached_raw" in record:
        raw = np.load(record["cached_raw"], mmap_mode="r", allow_pickle=False)
        pool = np.load(record["cached_pool"], mmap_mode="r", allow_pickle=False)
    else:
        source = {key: record[key] for key in ("path", "series_index", "crop_tzyx")}
        source["channels"] = "all"
        cache = root / "cache" / name
        raw, manifest = prepare_all_channels(source, cache / "counts.npy")
        pool, _ = prepare_training_pool(
            raw,
            cache / "training_counts.npy",
            {"training_probability": 0.8, "seed": 8801},
            manifest["sha256"],
        )
    source = {"path": record["path"], "series_index": record["series_index"], "channels": "all"}
    _, geometry = describe_source(source)
    names = [d["Name"] for d in geometry["identity"]["active_detectors"]]
    if names != ["HyD 2", "HyD 4"]:
        message = "Motion inference requires the trained [HyD 2, HyD 4] detector order"
        raise ValueError(message)
    return raw, pool, geometry


def select_region(values, record, region, config):
    """Extract real temporal context at absolute source coordinates, without padding time."""
    first = config["core_time_start"] - config["temporal_halo"]
    starts = [first, *region["start_zyx"]]
    sizes = [config["core_frames"] + 2 * config["temporal_halo"], *region["shape_zyx"]]
    local = [a - b for a, b in zip(starts, record["cache_offset_tzyx"], strict=True)]
    if any(
        a < 0 or a + n > have for a, n, have in zip(local, sizes, values.shape[1:], strict=True)
    ):
        message = "Motion region or temporal halo lies outside the verified source cache"
        raise ValueError(message)
    selection = (slice(None), *(slice(a, a + n) for a, n in zip(local, sizes, strict=True)))
    return np.array(values[selection], dtype=np.uint8, copy=True)


def infer_region(model, raw, pool, config, *, seed, device):
    """Average complementary predictions at the trained input count level."""
    if np.any(pool > raw):
        message = "The retained photon pool is not a subset of observations"
        raise ValueError(message)
    first_time = config["core_time_start"] - config["temporal_halo"]
    a, b = split_motion_pool(pool, seed=seed, first_time=first_time)
    options = {"halo": config["temporal_halo"], "core": config["temporal_core"], "device": device}
    control = temporal_control(model, a, **options)
    qa = normalized_motion_probabilities(
        temporal_logits(model, a, frames=config["core_frames"], **options), device
    )
    qb = normalized_motion_probabilities(
        temporal_logits(model, b, frames=config["core_frames"], **options), device
    )
    q = (qa + qb) * np.float32(0.5)
    q /= q.sum(axis=(1, 2, 3, 4), dtype=np.float64)[:, None, None, None, None]
    halo = config["temporal_halo"]
    end = halo + config["core_frames"]
    observed = np.ascontiguousarray(raw[:, halo:end])
    heldout = np.ascontiguousarray((raw - pool)[:, halo:end])
    totals = observed.sum(axis=(1, 2, 3, 4), dtype=np.int64)
    expected = (q * totals[:, None, None, None, None]).astype(np.float32)
    return {
        "observed": observed,
        "heldout": heldout,
        "gap4d_probability": q,
        "gap4d_expected_counts": expected,
    }, control


def _export(root, name, record, region, config, arrays, control, geometry, checkpoint_sha):
    directory = f"{name}_{region['name']}"
    destination = root / "examples" / directory
    destination.mkdir(parents=True, exist_ok=True)
    for field, values in arrays.items():
        np.save(destination / f"{field}.npy", values, allow_pickle=False)
    q = arrays["gap4d_probability"]
    metadata = {
        "status": "complete",
        "recording": name,
        "recording_name": record["title"],
        "phase": region["name"],
        "source_start_tzyx": [config["core_time_start"], *region["start_zyx"]],
        "shape_ctzyx": list(q.shape),
        "axis_order": "C,T,Z,Y,X",
        "channel_names": ["HyD 2", "HyD 4"],
        "voxel_spacing_zyx_um": geometry["voxel_spacing_zyx_um"],
        "volume_interval_seconds": geometry["volume_interval_seconds"],
        "reverse_z_for_display": geometry["reverse_z_for_display"],
        "temporal_halo": config["temporal_halo"],
        "temporal_core": config["temporal_core"],
        "checkpoint_sha256": checkpoint_sha,
        "temporal_control": control,
        "probability_mass": q.sum(axis=(1, 2, 3, 4), dtype=np.float64).tolist(),
        "input_count_level": 0.4,
        "inference": "Real temporal halos; valid-center logits; global per-detector ensemble PDF",
        "model_training_recording": "microtubule3",
        "transfer_test": name == "microtubule4",
        "scope": "Exploratory continuous motion review; primary short-patch NLL does not apply",
        "clean_ground_truth": False,
    }
    write_json(destination / "metadata.json", metadata)
    return {"directory": directory, **metadata}


def main():
    """Prepare bounded clips, validate temporal context, and reconstruct both detectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/confocal_twochannel_motion.yml")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    training_root = Path(config["model_study"])
    result = json.loads((training_root / "training/result.json").read_text())
    if source_hashes() != result["identity"]["source_hashes"]:
        message = "The trained two-channel model sources changed"
        raise ValueError(message)
    checkpoint_path = training_root / "training/last.pt"
    checkpoint_sha = sha256_file(checkpoint_path)
    primary = json.loads((training_root / "comparison.json").read_text())
    if checkpoint_sha != primary["checkpoint_sha256"]:
        message = "Use the independently audited final two-channel checkpoint"
        raise ValueError(message)
    prepared = {
        name: prepare_recording(name, record, root) for name, record in config["recordings"].items()
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    device = torch.device("cuda:0")
    model = MultichannelConfocalGAP(channels=2, **result["identity"]["config"]["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    entries = []
    write_json(root / "config.json", config)
    for data_index, (name, record) in enumerate(config["recordings"].items()):
        raw, pool, geometry = prepared[name]
        for roi_index, region in enumerate(record["regions"]):
            LOGGER.info("Reconstructing both channels: %s / %s", name, region["name"])
            selected_raw = select_region(raw, record, region, config)
            selected_pool = select_region(pool, record, region, config)
            arrays, control = infer_region(
                model,
                selected_raw,
                selected_pool,
                config,
                seed=config["split_seed"] + 1000 * data_index + 10 * roi_index,
                device=device,
            )
            entry = _export(
                root, name, record, region, config, arrays, control, geometry, checkpoint_sha
            )
            entries.append(entry)
            write_json(
                root / "index.json",
                {
                    "status": "running",
                    "clips": entries,
                    "model_study": str(training_root),
                    "checkpoint_sha256": checkpoint_sha,
                },
            )
            LOGGER.info("Finished %s / %s; halo/static checks passed", name, region["name"])
            del arrays, selected_raw, selected_pool
    write_json(
        root / "index.json",
        {
            "status": "complete",
            "clips": entries,
            "model_study": str(training_root),
            "checkpoint_sha256": checkpoint_sha,
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
    )
    write_json(
        root / "inference_sources.json",
        {
            name: sha256_file(Path(name))
            for name in ("src/ssunet/confocal_motion.py", "src/tools/confocal_motion_review.py")
        },
    )


if __name__ == "__main__":
    main()
