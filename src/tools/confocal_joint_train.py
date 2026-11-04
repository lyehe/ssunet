"""Train a joint-channel 4D GAP pilot with permanent photon holdout and checkpoints."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import time
from pathlib import Path

import torch
import yaml
from filelock import FileLock

from ..ssunet.confocal_inference import sha256_file, write_json
from ..ssunet.confocal_trial import (
    evaluate_baselines,
    evaluate_model,
    prepare_training_pool,
    training_batch,
    validation_cases,
)
from ..ssunet.datasets.confocal import validate_patch_shape
from ..ssunet.datasets.confocal_channels import prepare_all_channels
from ..ssunet.models.confocal_multichannel import MultichannelConfocalGAP, multichannel_gap_loss

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("config/confocal_microtubule3_joint4d.yml")
SOURCE_FILES = [
    "src/tools/confocal_joint_train.py",
    "src/tools/confocal_joint_report.py",
    "src/ssunet/confocal_trial.py",
    "src/ssunet/models/confocal_multichannel.py",
    "src/ssunet/datasets/confocal_channels.py",
    "src/ssunet/datasets/confocal.py",
    "src/ssunet/models/confocal_gap.py",
    "src/ssunet/photon_methods.py",
    "src/ssunet/distributions.py",
    "src/ssunet/photon_reference.py",
    "src/ssunet/softmax_fix.py",
    "src/ssunet/confocal_inference.py",
]


def load_config(path):
    """Require raw multi-count GAP supervision and the tested FP32 execution mode."""
    config = yaml.safe_load(Path(path).read_text())
    train = config["training"]
    validate_patch_shape(train["patch_tzyx"])
    if (
        config["observation_kind"] != "photon_counts"
        or config["method"] != "gap"
        or config["source"]["channels"] != "all"
        or train["precision"] != "fp32"
        or train["logit_normalization"] != "barron2026_per_detector"
        or not 0 < config["photon_pool"]["training_probability"] < 1
        or not 0 < train["split_probability"] < 1
        or any(
            train[key] < 1
            for key in ("steps", "batch_size", "accumulation_steps", "checkpoint_every")
        )
        or min(train["learning_rate"], train["gradient_clip"]) <= 0
        or config["evaluation"]["every_steps"] < 1
    ):
        message = (
            "Expected all-channel FP32 GAP with positive settings and independent photon holdout"
        )
        raise ValueError(message)
    return config


def prepare_data(config):
    """Verify the raw local cache and the permanent training/evaluation partition."""
    root = Path(config["output"])
    raw, raw_manifest = prepare_all_channels(config["source"], root / "counts.npy")
    pool, pool_manifest = prepare_training_pool(
        raw, root / "training_counts.npy", config["photon_pool"], raw_manifest["sha256"]
    )
    cases = validation_cases(raw, pool, config)
    return pool, cases, {"raw": raw_manifest, "training_pool": pool_manifest}


def source_hashes():
    """Identify every local implementation used in fitting and scoring."""
    project = Path(__file__).resolve().parents[2]
    return {name: sha256_file(project / name) for name in SOURCE_FILES}


def convolution_hashes():
    """Identify the installed convolution package and its Python/C++/CUDA sources."""
    import torch_convnd

    package = Path(torch_convnd.__file__).parent
    suffixes = {".py", ".cpp", ".cu", ".h", ".cuh", ".hpp"}
    return {
        "version": importlib.metadata.version("torch-convnd"),
        "sources": {
            str(path.relative_to(package)): sha256_file(path)
            for path in sorted(package.rglob("*"))
            if path.is_file() and path.suffix in suffixes
        },
    }


def snapshot_sources(root, expected):
    """Archive source bytes, rejecting edits that could invalidate a resumed trial."""
    project = Path(__file__).resolve().parents[2]
    if source_hashes() != expected:
        message = "A trial source changed during fitting"
        raise ValueError(message)
    for name, digest in expected.items():
        destination = root / "source_snapshot" / name
        if destination.exists():
            if sha256_file(destination) != digest:
                message = "An archived trial source changed"
                raise ValueError(message)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((project / name).read_bytes())
    write_json(root / "source_snapshot.json", expected)


def optimizer_step(model, optimizer, pool, config, device, step):
    """Average canonical raw-count patch losses before clipping and one Adam update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    options = config["training"]
    for microstep in range(options["accumulation_steps"]):
        a, b = training_batch(pool, config, step, microstep)
        a, b = a.to(device), b.to(device)
        loss = multichannel_gap_loss(model(a), b, a) / options["accumulation_steps"]
        if not bool(torch.isfinite(loss)):
            message = "Nonfinite joint-channel GAP loss"
            raise RuntimeError(message)
        loss.backward()
        loss_value += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), options["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return loss_value, float(norm)


def _checkpoint(path, model, optimizer, state, identity, device):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "state": state,
        "identity": identity,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore(path, model, optimizer, identity, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["identity"] != identity:
        message = "Checkpoint source/config/data identity differs from this trial"
        raise ValueError(message)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), device)
    return payload["state"]


def run_training(model, pool, cases, config, directory, identity, device, *, resume=False):
    """Run the fixed endpoint, restoring the exact optimizer/example cursor on resume."""
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "last.pt"
    options = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    if checkpoint.exists() and not resume:
        message = "A checkpoint already exists; use --resume or a new output directory"
        raise FileExistsError(message)
    if resume:
        state = _restore(checkpoint, model, optimizer, identity, device)
    else:
        state = {
            "step": 0,
            "training_seconds": 0.0,
            "losses": [],
            "evaluations": [{"step": 0, **evaluate_model(model, cases, device)}],
            "baselines": evaluate_baselines(cases, config),
            "peak_gpu_allocated_gib": 0.0,
            "peak_gpu_reserved_gib": 0.0,
        }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_started, prior_seconds = time.perf_counter(), state["training_seconds"]
    write_json(directory / "progress.json", {**state, "phase": "training", "pid": os.getpid()})
    for step in range(state["step"] + 1, options["steps"] + 1):
        loss, norm = optimizer_step(model, optimizer, pool, config, device, step)
        state["step"] = step
        state["losses"].append({"step": step, "loss": loss, "gradient_norm": norm})
        if step % config["evaluation"]["every_steps"] == 0 or step == options["steps"]:
            score = evaluate_model(model, cases, device)
            state["evaluations"].append({"step": step, **score})
            LOGGER.info(
                "Step %d: held-out NLL %.6f; detector mean %.6f",
                step,
                score["photon_nll"],
                score["channel_mean_nll"],
            )
        state["training_seconds"] = prior_seconds + time.perf_counter() - run_started
        if device.type == "cuda":
            state["peak_gpu_allocated_gib"] = max(
                state["peak_gpu_allocated_gib"], torch.cuda.max_memory_allocated(device) / 2**30
            )
            state["peak_gpu_reserved_gib"] = max(
                state["peak_gpu_reserved_gib"], torch.cuda.max_memory_reserved(device) / 2**30
            )
        if step % 5 == 0 or step == options["steps"]:
            write_json(
                directory / "progress.json", {**state, "phase": "training", "pid": os.getpid()}
            )
            LOGGER.info(
                "Training %d/%d loss %.6f, elapsed %.1f min",
                step,
                options["steps"],
                loss,
                state["training_seconds"] / 60,
            )
        if step % options["checkpoint_every"] == 0 or step == options["steps"]:
            if source_hashes() != identity["source_hashes"]:
                message = "Source changed during fitting; refusing an unauditable checkpoint"
                raise ValueError(message)
            _checkpoint(checkpoint, model, optimizer, state, identity, device)
    write_json(directory / "progress.json", {**state, "phase": "complete", "pid": os.getpid()})
    return state


def main():
    """Prepare, capacity-test or fit both detectors on physical GPU zero only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(args.config)
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    pool, cases, manifests = prepare_data(config)
    if args.prepare_only:
        LOGGER.info("Verified both detector caches and %d fixed evaluation cases", len(cases))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    device = torch.device("cuda:0")
    torch.manual_seed(config["training"]["seed"])
    if args.smoke_test:
        config["training"]["steps"] = 3
    model = MultichannelConfocalGAP(channels=pool.shape[0], **config["model"]).to(device)
    identity = {
        "config": config,
        "source_hashes": source_hashes(),
        "raw_sha256": manifests["raw"]["sha256"],
        "training_pool_sha256": manifests["training_pool"]["sha256"],
        "channel_names": manifests["raw"]["channel_names"],
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "convolution": convolution_hashes(),
        "device_name": torch.cuda.get_device_name(0),
    }
    directory = root / ("smoke" if args.smoke_test else "training")
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "train.lock"), timeout=0):
        snapshot_sources(root, identity["source_hashes"])
        write_json(directory / "process.json", {"pid": os.getpid(), "phase": "running"})
        state = run_training(
            model, pool, cases, config, directory, identity, device, resume=args.resume
        )
        result = {
            "status": "complete",
            "scope": "readiness" if args.smoke_test else config["evaluation"]["scope"],
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "identity": identity,
            "state": state,
        }
        write_json(directory / "result.json", result)
        write_json(directory / "process.json", {"pid": os.getpid(), "phase": "complete"})
        LOGGER.info(
            "Completed %d updates; peak allocated %.2f GiB",
            state["step"],
            state["peak_gpu_allocated_gib"],
        )


if __name__ == "__main__":
    main()
