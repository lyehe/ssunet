"""Sliding-window inference on binned SPAD data.

Loads z_size*bin_factor binary frames per window, bins them on GPU,
runs the model, and saves denoised output as PNGs.

The last 1024 binary frames (last file) are binned by bin_factor to produce
128 reference target frames saved alongside the denoised output.

Usage:
    python infer_binned.py \
        --scene-dir D:/Yehe/PC/reconstruction/test/basement \
        --checkpoint models_spad/basement-bin8.ckpt \
        --device 0 --output output/basement-bin8
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
from src.ssunet.models import Bit2Bit

torch.serialization.add_safe_globals([ModelConfig])


def frame_to_png(frame_np: np.ndarray, factor: float) -> np.ndarray:
    """Convert a linear RGB frame to sRGB uint8."""
    frame_rgb = spc_avg_to_rgb(frame_np, factor=factor)
    frame_srgb = linearrgb_to_srgb(frame_rgb)
    return (np.clip(frame_srgb, 0, 1) * 255).astype(np.uint8)


def save_target_frames(
    scene: SPADScene,
    bin_factor: int,
    factor: float,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Bin the last 1024 frames and save as target PNGs."""
    logger = logging.getLogger(__name__)
    target_dir = output_dir / "target"
    target_dir.mkdir(parents=True, exist_ok=True)

    last_file_start = (scene.n_files - 1) * scene.frames_per_file
    n_frames = scene.frames_per_file  # 1024
    n_binned = n_frames // bin_factor  # 128

    logger.info(
        "Saving %d target frames (last %d binary frames binned by %d)",
        n_binned,
        n_frames,
        bin_factor,
    )

    # Process in chunks to avoid OOM
    chunk_size = 64 * bin_factor  # 64 binned frames at a time
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        actual = end - start
        n_out = actual // bin_factor

        packed = scene.get_packed_patch(
            last_file_start + start,
            actual,
            0,
            scene.height,
        )
        packed_t = torch.from_numpy(packed).to(device)
        unpacked = gpu_unpackbits(packed_t, dim=2)[:, :, :800, :]  # (Z, H, W, C)

        # Bin: reshape to n_out x bin_factor x H x W x C, then sum each bin
        binned = unpacked.reshape(n_out, bin_factor, scene.height, 800, scene.channels)
        binned = binned.sum(dim=1).cpu().numpy().astype(np.float32) / bin_factor

        for i in range(n_out):
            global_idx = start // bin_factor + i
            png = frame_to_png(binned[i], factor)
            Image.fromarray(png).save(target_dir / f"{global_idx:06d}.png")

        del packed_t, unpacked, binned
        torch.cuda.empty_cache()

    logger.info("Saved %d target frames to %s", n_binned, target_dir)


def main() -> None:
    """Run sliding-window inference on binned SPAD data."""
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
    parser.add_argument("--bin-factor", type=int, default=8)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--z-size", type=int, default=32, help="Model z_size (binned frames)")
    parser.add_argument(
        "--keep", type=int, default=16, help="Keep center N binned frames per window"
    )
    parser.add_argument("--stride", type=int, default=8, help="Stride in binned frames")
    parser.add_argument(
        "--save-target",
        action="store_true",
        default=True,
        help="Save last-file binned frames as target PNGs",
    )
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)
    model_config = ModelConfig(**cfg["MODEL"])
    model_config.loss_function = "photon"

    device = torch.device(f"cuda:{args.device}")
    model = Bit2Bit.load_from_checkpoint(args.checkpoint, config=model_config, map_location=device)
    model.to(device)
    model.eval()

    scene = SPADScene(Path(args.scene_dir))
    bin_factor = args.bin_factor
    z_size = args.z_size
    z_raw = z_size * bin_factor  # binary frames per window
    keep = args.keep
    stride = args.stride
    margin = (z_size - keep) // 2

    # Total binned frames across entire scene
    total_binned = scene.total_frames // bin_factor

    output_dir = Path(args.output) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Scene: %s, total_frames=%d, total_binned=%d, bin_factor=%d",
        scene.name,
        scene.total_frames,
        total_binned,
        bin_factor,
    )
    logger.info(
        "z_size=%d, z_raw=%d, keep=%d, stride=%d, margin=%d",
        z_size,
        z_raw,
        keep,
        stride,
        margin,
    )

    # Save target frames from last file
    if args.save_target:
        save_target_frames(scene, bin_factor, args.factor, Path(args.output), device)

    # Compute window positions in binned-frame space
    windows = []
    t = 0
    while t + z_size <= total_binned:
        windows.append(t)
        t += stride
    if windows[-1] + z_size < total_binned:
        windows.append(total_binned - z_size)

    logger.info("Total windows: %d", len(windows))

    accum = {}  # binned_frame_idx -> (sum_array, count)
    n_saved = 0

    with torch.no_grad():
        for wi, t_binned in enumerate(windows):
            # Load z_raw binary frames starting at t_binned * bin_factor
            t_binary = t_binned * bin_factor
            packed = scene.get_packed_patch(t_binary, z_raw, 0, scene.height)
            packed_t = torch.from_numpy(packed).to(device)
            unpacked = gpu_unpackbits(packed_t, dim=2)[:, :, :800, :]  # (z_raw, H, W, C)

            # Bin: (z_raw, H, W, C) → (z_size, bin_factor, H, W, C) → sum → (z_size, H, W, C)
            binned = unpacked.reshape(z_size, bin_factor, scene.height, 800, scene.channels)
            binned = binned.sum(dim=1).float()  # (z_size, H, W, C)

            # To model format: (1, C, Z, H, W)
            data = binned.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, Z, H, W)

            output = torch.exp(model(data))
            input_mean = data.mean()
            vol = output[0]
            vol = vol / vol.mean() * input_mean
            # Normalize to per-frame average (divide by bin_factor to get per-binary-frame rate)
            volume_np = vol.cpu().numpy().transpose(1, 2, 3, 0) / bin_factor  # (Z, H, W, C)
            volume_np = np.clip(volume_np, 0, 1 - 1e-6)

            for k in range(margin, margin + keep):
                global_idx = t_binned + k
                frame = volume_np[k]

                if global_idx not in accum:
                    accum[global_idx] = (frame.astype(np.float64), 1)
                else:
                    s, c = accum[global_idx]
                    accum[global_idx] = (s + frame.astype(np.float64), c + 1)

            del packed_t, unpacked, binned, data, output
            torch.cuda.empty_cache()

            # Flush completed frames
            next_t = windows[wi + 1] if wi + 1 < len(windows) else total_binned
            flush_up_to = next_t + margin

            for idx in sorted(idx for idx in accum if idx < flush_up_to):
                s, c = accum.pop(idx)
                avg_frame = (s / c).astype(np.float32)
                png = frame_to_png(avg_frame, args.factor)
                Image.fromarray(png).save(output_dir / f"{idx:06d}.png")
                n_saved += 1

            if (wi + 1) % 50 == 0 or wi == len(windows) - 1:
                logger.info(
                    "Window %d/%d, saved %d frames, accum buffer: %d",
                    wi + 1,
                    len(windows),
                    n_saved,
                    len(accum),
                )

    # Flush remaining
    for idx in sorted(accum):
        s, c = accum.pop(idx)
        avg_frame = (s / c).astype(np.float32)
        png = frame_to_png(avg_frame, args.factor)
        Image.fromarray(png).save(output_dir / f"{idx:06d}.png")
        n_saved += 1

    logger.info("Done. %d denoised frames saved to %s", n_saved, output_dir)


if __name__ == "__main__":
    main()
