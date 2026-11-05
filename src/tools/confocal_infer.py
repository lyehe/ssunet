"""Reconstruct a complete confocal acquisition with the saved 4D SSL model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from filelock import FileLock

from ..ssunet.confocal_inference import (
    describe_source,
    photon_half,
    prepare_full_counts,
    reconstruct_core,
    sha256_file,
    write_json,
)
from ..ssunet.models.confocal_gap import ConfocalGAPUNet
from ..ssunet.softmax_fix import fixed_log_softmax
from .confocal_evaluate import _validate_external_package, _validate_sources

LOGGER = logging.getLogger(__name__)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("data/confocal_gap4d_features32_batch4_v1")
    )
    parser.add_argument("--source", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("data/confocal_full_series013_v1"))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--core-time", type=int, default=96)
    parser.add_argument("--first-core", type=int, default=16)
    parser.add_argument("--halo-time", type=int, default=14)
    parser.add_argument("--patch-zyx", type=int, nargs=3, default=(16, 128, 128))
    parser.add_argument("--overlap-zyx", type=int, nargs=3, default=(4, 32, 32))
    parser.add_argument("--microbatch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=4202)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def _model(study, settings):
    saved = torch.load(
        study / "training/pilot/conv4d/last.pt", map_location="cpu", weights_only=True
    )
    if saved["step"] != 500:
        message = "Whole-acquisition inference requires the completed fixed 500-update checkpoint"
        raise ValueError(message)
    model = ConfocalGAPUNet(**{k: v for k, v in settings.items() if k != "name"})
    model.load_state_dict(saved["model_state_dict"])
    return model.to("cuda:0").eval()


@torch.inference_mode()
def precision_audit(model, study, precision):
    """Check three predeclared real examples against FP32 before mixed-precision export."""
    examples = np.load(study / "prediction_examples.npz", allow_pickle=False)
    results = []
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    for index in range(3):
        raw = torch.from_numpy(examples["input"][index][None, None]).to("cuda:0")
        target = torch.from_numpy(examples["target"][index]).to("cuda:0").flatten().double()
        reference = fixed_log_softmax(model(raw).float().flatten(1))
        if dtype is None:
            actual = reference
        else:
            with torch.autocast("cuda", dtype=dtype):
                logits = model(raw)
            actual = fixed_log_softmax(logits.float().flatten(1))
        l1 = float((reference.exp() - actual.exp()).abs().sum())
        delta = float(((reference[0] - actual[0]).double() * target).sum() / target.sum())
        if not np.isfinite([l1, delta]).all() or l1 > 0.005 or abs(delta) > 0.001:
            message = "Inference precision differs excessively from FP32; use --precision fp32"
            raise ValueError(message)
        results.append(
            {
                "example": int(examples["example_indices"][index]),
                "probability_l1": l1,
                "photon_nll_delta": delta,
            }
        )
    return {
        "precision": precision,
        "normalization": "FP32 Barron equation 5",
        "status": "passed",
        "examples": results,
    }


def _inference_sources():
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root / "src/ssunet/confocal_inference.py"]
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def _manifest(args, config, source, description):
    settings = next(item for item in config["models"] if item["name"] == "conv4d")
    minimum_halo = 4 * settings["depth"] - 2
    if args.halo_time < minimum_halo or min(args.core_time, args.first_core) < 1:
        message = "Temporal halo must cover the network's full temporal receptive field"
        raise ValueError(message)
    checkpoint = sha256_file(args.study / "training/pilot/conv4d/last.pt")
    evidence = json.loads((args.study / "evidence_audit.json").read_text())
    if (
        evidence["status"] != "passed"
        or evidence["models"]["conv4d"]["checkpoint_sha256"] != checkpoint
    ):
        message = "The saved checkpoint differs from the successfully replayed study"
        raise ValueError(message)
    return {
        **description,
        "source": source,
        "model": settings,
        "checkpoint_sha256": checkpoint,
        "inference_sources": _inference_sources(),
        "precision": args.precision,
        "core_time": args.core_time,
        "first_core": args.first_core,
        "halo_time": args.halo_time,
        "patch_zyx": list(args.patch_zyx),
        "overlap_zyx": list(args.overlap_zyx),
        "microbatch_size": args.microbatch_size,
        "seed": args.seed,
        "method": "Complementary half-count conditional PDFs; Hann spatial blending",
        "probability_support": "Joint full-acquisition TZYX photon-arrival distribution",
        "counts_calibration": "Observed patch totals and full-acquisition mass calibration",
        "scope": "SSL reconstruction; training acquisitions are valid inference inputs",
        "clean_ground_truth": False,
    }


def _completed_chunks(args, manifest):
    manifest_path = args.output / "manifest.json"
    progress_path = args.output / "progress.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
        return [], "new"
    if json.loads(manifest_path.read_text()) != manifest:
        message = "Inference configuration, source or checkpoint changed; choose a new output"
        raise ValueError(message)
    if not args.resume and progress_path.exists():
        message = "This output already has inference progress; use --resume"
        raise FileExistsError(message)
    if not progress_path.exists():
        return [], "new"
    progress = json.loads(progress_path.read_text())
    return progress["chunks"], progress["status"]


def _output_array(args, shape, chunks):
    path = args.output / "expected_counts.npy"
    if chunks:
        values = np.load(path, mmap_mode="r+", allow_pickle=False)
        if values.shape != shape or values.dtype != np.float32:
            message = "Reconstruction shape/dtype differs from the source"
            raise ValueError(message)
        normalizing = (args.output / "normalization.json").exists()
        for chunk in [] if normalizing else chunks:
            part = np.ascontiguousarray(values[chunk["begin"] : chunk["end"]])
            if hashlib.sha256(memoryview(part)).hexdigest() != chunk["sha256"]:
                message = "A completed inference chunk has changed"
                raise ValueError(message)
        return values
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _progress(args, chunks, status, **extra):
    return {
        "status": status,
        "pid": os.getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_frames": chunks[-1]["end"] if chunks else 0,
        "chunks": chunks,
        **extra,
    }


def _reconstruct(args, model, counts, output, chunks):
    first = chunks[-1]["end"] if chunks else 0
    while first < counts.shape[0]:
        end = min(counts.shape[0], first + (args.first_core if first == 0 else args.core_time))
        read_begin, read_end = (
            max(0, first - args.halo_time),
            min(counts.shape[0], end + args.halo_time),
        )
        observed = np.asarray(counts[read_begin:read_end])
        retained = photon_half(observed, first_time=read_begin, seed=args.seed)
        started = time.perf_counter()

        def on_tile(done, total, first=first, end=end, started=started):
            if done % 10 == 0 or done == total:
                write_json(
                    args.output / "progress.json",
                    _progress(
                        args,
                        chunks,
                        "running",
                        active_core=[first, end],
                        tile=done,
                        tiles=total,
                        elapsed_core_seconds=time.perf_counter() - started,
                    ),
                )
                LOGGER.info("Volumes %d:%d, tile %d/%d", first, end, done, total)

        values = reconstruct_core(
            model,
            observed,
            retained,
            core_slice=slice(first - read_begin, end - read_begin),
            patch_zyx=args.patch_zyx,
            overlap_zyx=args.overlap_zyx,
            precision=args.precision,
            microbatch_size=args.microbatch_size,
            device=torch.device("cuda:0"),
            on_tile=on_tile,
        )
        output[first:end] = values
        output.flush()
        chunks.append(
            {
                "begin": first,
                "end": end,
                "observed_photons": int(counts[first:end].sum(dtype=np.uint64)),
                "predicted_mass": float(values.sum(dtype=np.float64)),
                "sha256": hashlib.sha256(memoryview(values)).hexdigest(),
                "seconds": time.perf_counter() - started,
            }
        )
        write_json(args.output / "progress.json", _progress(args, chunks, "running"))
        LOGGER.info("Completed %d/%d volumes in %.1fs", end, counts.shape[0], chunks[-1]["seconds"])
        first = end
        del values, retained


def finalize(output, values, chunks):
    """Export a normalized acquisition PDF without mistaking its count scale for GT."""
    photons = sum(chunk["observed_photons"] for chunk in chunks)
    mass = sum(chunk["predicted_mass"] for chunk in chunks)
    normalization_path = output / "normalization.json"
    if normalization_path.exists():
        saved = json.loads(normalization_path.read_text())
        if (
            saved["raw_mass"] != mass
            or saved["photons"] != photons
            or saved["probability_sha256"] != sha256_file(output / "probability.npy")
        ):
            message = "Saved normalization checkpoint differs from the inference output"
            raise ValueError(message)
        q = np.load(output / "probability.npy", mmap_mode="r", allow_pickle=False)
    else:
        partial = output / "probability.partial.npy"
        q = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=values.shape)
        for first in range(values.shape[0]):
            q[first] = values[first] / mass if mass > 0 else 1 / values.size
        q.flush()
        del q
        partial.replace(output / "probability.npy")
        write_json(
            normalization_path,
            {
                "raw_mass": mass,
                "photons": photons,
                "probability_sha256": sha256_file(output / "probability.npy"),
            },
        )
        q = np.load(output / "probability.npy", mmap_mode="r", allow_pickle=False)
    probability_sum = sum(float(frame.sum(dtype=np.float64)) for frame in q)
    if not np.isfinite(probability_sum) or abs(probability_sum - 1) > 1e-6:
        message = "Full-acquisition probability normalization failed"
        raise ValueError(message)
    # Rewrite from q, so a resumed finalization cannot multiply a scale twice.
    for first in range(values.shape[0]):
        values[first] = q[first] * photons
    values.flush()
    return {
        "probability_sum_float64": probability_sum,
        "observed_photons": photons,
        "expected_count_sum": sum(float(frame.sum(dtype=np.float64)) for frame in values),
        "raw_tiled_prediction_mass": mass,
        "calibration_note": "Total expected counts is estimated from the observed acquisition",
        "probability_sha256": sha256_file(output / "probability.npy"),
    }


def run(args):
    """Verify provenance, reconstruct all real voxels, and publish resumable output."""
    config = yaml.safe_load((args.study / "config.yml").read_text())
    source = dict(config["sources"][args.source])
    _, description = describe_source(source)
    source["crop_tzyx"] = [[0, size] for size in description["shape_tzyx"]]
    counts, description = prepare_full_counts(
        source,
        args.output,
        on_progress=lambda n, total: LOGGER.info("Copying %d/%d full volumes", n, total),
    )
    if args.prepare_only:
        return
    hashes = json.loads((args.study / "source_snapshot.json").read_text())
    _validate_sources(args.study, hashes)
    _validate_external_package(args.study)
    manifest = _manifest(args, config, source, description)
    chunks, status = _completed_chunks(args, manifest)
    if status == "complete":
        LOGGER.info("The complete acquisition is already reconstructed")
        return
    for name in manifest["inference_sources"]:
        snapshot = args.output / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(Path(name).read_bytes())
    model = _model(args.study, manifest["model"])
    write_json(
        args.output / "precision_audit.json", precision_audit(model, args.study, args.precision)
    )
    output = _output_array(args, counts.shape, chunks)
    write_json(args.output / "progress.json", _progress(args, chunks, "running"))
    _reconstruct(args, model, counts, output, chunks)
    if _inference_sources() != manifest["inference_sources"]:
        message = "The inference implementation changed during reconstruction"
        raise ValueError(message)
    _validate_sources(args.study, hashes)
    write_json(args.output / "progress.json", _progress(args, chunks, "normalizing"))
    result = finalize(args.output, output, chunks)
    write_json(
        args.output / "result.json",
        {
            "status": "passed",
            **result,
            "shape_tzyx": list(counts.shape),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "source_verified": True,
            "precision": args.precision,
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "inference_seconds": sum(item["seconds"] for item in chunks),
        },
    )
    write_json(args.output / "progress.json", _progress(args, chunks, "complete"))
    LOGGER.info(
        "Full-acquisition reconstruction complete; probability sum %.10f",
        result["probability_sum_float64"],
    )


def main():
    """Run full-volume SSL reconstruction on physical GPU 0 only."""
    args = _arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.prepare_only:
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(args.output / "inference.lock", timeout=0):
        run(args)


if __name__ == "__main__":
    main()
