"""Measure a sealed follow-up stage without changing its frozen implementation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.neutrino_training import _peak_rss
from .neutrino_ablations import test_mechanisms
from .neutrino_precision import resolve_config, test_precision
from .neutrino_precision_continuation import _configuration, test_continuation
from .neutrino_prepare import write_json
from .neutrino_probe import run_probe
from .neutrino_pulse_study import test_pulses


def measure_stage(action, config_path, device):
    """Record process/GPU peaks around a single already-authorized evaluation."""
    config = yaml.safe_load(Path(config_path).read_text())
    functions = {
        "pulse-test": test_pulses,
        "probe": run_probe,
        "mechanisms-test": test_mechanisms,
        "precision-test": test_precision,
        "continuation-test": test_continuation,
    }
    if action == "precision-test":
        config = resolve_config(config)
        root = Path(config["precision_followup"]["output"])
    elif action == "continuation-test":
        config = _configuration(config)
        root = Path(config["continuation"]["output"])
    else:
        root = Path(config["data"]["output"])
    destination = root / f"performance_{action}.json"
    if destination.exists():
        message = "This measured stage already has a performance record"
        raise FileExistsError(message)
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        torch.cuda.set_device(selected_device)
        torch.cuda.reset_peak_memory_stats(selected_device)
    started, succeeded = time.perf_counter(), False
    try:
        functions[action](config, selected_device)
        succeeded = True
    finally:
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        write_json(
            destination,
            {
                "action": action,
                "completed": succeeded,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_process_rss_gib": _peak_rss(),
                "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated(selected_device) / 2**30
                if selected_device.type == "cuda"
                else None,
                "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved(selected_device) / 2**30
                if selected_device.type == "cuda"
                else None,
                "device": str(selected_device),
                "measurement_code_sha256": shard_sha256(Path(__file__)),
                "frozen_stage_source_unchanged": True,
            },
        )


def main():
    """Run one externally instrumented stage with the original test guards intact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("pulse-test", "probe", "mechanisms-test", "precision-test", "continuation-test"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    measure_stage(args.action, args.config, args.device)


if __name__ == "__main__":
    main()
