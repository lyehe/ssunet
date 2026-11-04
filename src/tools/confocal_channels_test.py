"""Verify real-data training and prediction with every confocal detector channel."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import yaml

from ..ssunet.confocal_inference import write_json
from ..ssunet.datasets.confocal_channels import prepare_all_channels
from ..ssunet.models.confocal_multichannel import (
    MultichannelConfocalGAP,
    channel_log_probabilities,
    multichannel_gap_loss,
    split_all_channels,
)

LOGGER = logging.getLogger(__name__)


def train_steps(model, counts, options, device):
    """Train a bounded readiness test with raw per-detector targets and joint inputs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    losses = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(options["smoke_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for micro in range(options["accumulation_steps"]):
            index = step * options["accumulation_steps"] + micro
            a, b = split_all_channels(
                counts,
                options["patch_tzyx"],
                seed=options["photon_split_seed"],
                index=index,
                batch_size=options["batch_size"],
            )
            a, b = a.to(device), b.to(device)
            loss = multichannel_gap_loss(model(a), b, a) / options["accumulation_steps"]
            loss.backward()
            loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"])
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(loss_value)
        LOGGER.info(
            "All-channel training step %d/%d loss %.6f",
            step + 1,
            options["smoke_steps"],
            loss_value,
        )
    model.eval()
    with torch.inference_mode():
        logits = model(a[:1])
        q = channel_log_probabilities(logits).exp()
        mass = q.flatten(2).sum(-1)
    if not torch.isfinite(mass).all() or float((mass - 1).abs().max()) > 1e-5:
        message = "Each detector output must have finite unit probability mass"
        raise ValueError(message)
    return {
        "training_losses": losses,
        "seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "probability_mass_per_detector": mass[0].tolist(),
        "batch_size": options["batch_size"],
        "accumulation_steps": options["accumulation_steps"],
        "effective_batch_size": options["batch_size"] * options["accumulation_steps"],
        "steps": options["smoke_steps"],
    }


def main():
    """Prepare all channels and run the full-size 4D forward/backward preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/confocal_all_channels_test.yml")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    counts, manifest = prepare_all_channels(config["source"], root / "counts.npy")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    torch.manual_seed(config["training"]["seed"])
    model = MultichannelConfocalGAP(channels=counts.shape[0], **config["model"]).to("cuda:0")
    result = train_steps(model, counts, config["training"], torch.device("cuda:0"))
    result.update(
        status="passed",
        scope="Three-update readiness check, not a trained quality result",
        model_parameters=sum(p.numel() for p in model.parameters()),
        detector_channels=counts.shape[0],
        channel_names=manifest["channel_names"],
        config=config,
    )
    write_json(root / "readiness.json", result)
    LOGGER.info(
        "Every detector output normalized; peak allocated %.2f GiB",
        result["peak_gpu_allocated_gib"],
    )


if __name__ == "__main__":
    main()
