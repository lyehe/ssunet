"""Sliding window inference to reconstruct ALL frames.

Uses z_size=32 windows, keeps center 16 frames from each, stride=8.
Each frame gets ~2 estimates (from overlapping windows) which are averaged.

Usage:
    python infer_all_frames.py --scene-dir D:/Yehe/PC/reconstruction/test/basement \
        --checkpoint models_spad/basement.ckpt --device 0 --output output/basement-allframes
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from visionsim.emulate.spc import spc_avg_to_rgb
from visionsim.utils.color import linearrgb_to_srgb

from src.ssunet.configs.model_config import ModelConfig
from src.ssunet.datasets.spad import SPADScene, gpu_unpackbits

torch.serialization.add_safe_globals([ModelConfig])
from src.ssunet.models import Bit2Bit


def frame_to_png(frame_np: np.ndarray, factor: float) -> np.ndarray:
    frame_rgb = spc_avg_to_rgb(frame_np, factor=factor)
    frame_srgb = linearrgb_to_srgb(frame_rgb)
    return (np.clip(frame_srgb, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config_spad.yml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--z-size", type=int, default=32)
    parser.add_argument("--keep", type=int, default=16, help="Keep center N frames per window")
    parser.add_argument("--stride", type=int, default=8, help="Stride between windows")
    parser.add_argument("--save-png", action="store_true", default=True)
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)
    model_config = ModelConfig(**cfg["MODEL"])

    device = torch.device(f"cuda:{args.device}")
    model = Bit2Bit.load_from_checkpoint(args.checkpoint, config=model_config, map_location=device)
    model.to(device)
    model.eval()

    scene = SPADScene(Path(args.scene_dir))
    total_frames = scene.total_frames
    z_size = args.z_size
    keep = args.keep
    stride = args.stride
    margin = (z_size - keep) // 2  # frames to discard on each side

    output_dir = Path(args.output) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Scene: %s, total_frames=%d, z_size=%d, keep=%d, stride=%d, margin=%d",
        scene.name, total_frames, z_size, keep, stride, margin,
    )

    # Compute all window positions
    windows = []
    t = 0
    while t + z_size <= total_frames:
        windows.append(t)
        t += stride
    # Handle last window
    if windows[-1] + z_size < total_frames:
        windows.append(total_frames - z_size)

    logger.info("Total windows: %d", len(windows))

    # Accumulator: for each global frame, accumulate (sum, count)
    # To save memory, process and save in chunks
    # Each window produces frames [t+margin, t+margin+keep)
    # We process windows in order and flush frames once all contributions are in

    # Track which frames have been fully accumulated
    accum = {}  # frame_idx -> (sum_array, count)

    n_saved = 0
    with torch.no_grad():
        for wi, t_start in enumerate(windows):
            packed = scene.get_packed_patch(t_start, z_size, 0, scene.height)
            packed_t = torch.from_numpy(packed).to(device)
            unpacked = gpu_unpackbits(packed_t, dim=2)[:, :, :800, :]
            data = unpacked.permute(3, 0, 1, 2).unsqueeze(0).float()

            output = torch.exp(model(data))
            input_mean = data.mean()
            vol = output[0]
            vol = vol / vol.mean() * input_mean
            volume_np = vol.cpu().numpy().transpose(1, 2, 3, 0)  # (Z, H, W, C)
            volume_np = np.clip(volume_np, 0, 1 - 1e-6)

            # Keep center `keep` frames
            for k in range(margin, margin + keep):
                global_idx = t_start + k
                frame = volume_np[k]

                if global_idx not in accum:
                    accum[global_idx] = (frame.astype(np.float64), 1)
                else:
                    s, c = accum[global_idx]
                    accum[global_idx] = (s + frame.astype(np.float64), c + 1)

            del packed_t, unpacked, data, output
            torch.cuda.empty_cache()

            # Flush frames that won't get more contributions
            # A frame at global_idx is complete if the next window can't contribute to it
            next_t = windows[wi + 1] if wi + 1 < len(windows) else total_frames
            flush_up_to = next_t + margin  # next window's first kept frame

            frames_to_flush = sorted(
                idx for idx in accum if idx < flush_up_to
            )

            for idx in frames_to_flush:
                s, c = accum.pop(idx)
                avg_frame = (s / c).astype(np.float32)

                if args.save_png:
                    png = frame_to_png(avg_frame, args.factor)
                    Image.fromarray(png).save(output_dir / f"{idx:06d}.png")

                n_saved += 1

            if (wi + 1) % 100 == 0 or wi == len(windows) - 1:
                logger.info(
                    "Window %d/%d, saved %d frames, accum buffer: %d",
                    wi + 1, len(windows), n_saved, len(accum),
                )

    # Flush remaining
    for idx in sorted(accum):
        s, c = accum.pop(idx)
        avg_frame = (s / c).astype(np.float32)
        if args.save_png:
            png = frame_to_png(avg_frame, args.factor)
            Image.fromarray(png).save(output_dir / f"{idx:06d}.png")
        n_saved += 1

    logger.info("Done. %d frames saved to %s", n_saved, output_dir)


if __name__ == "__main__":
    main()
