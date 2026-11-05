"""Inference: denoise the last binned frame of each 1024-frame chunk.

For each .npy file (chunk of 1024 binary frames):
  - Loads the last z_size*bin_factor binary frames ending at the chunk boundary
  - Bins by bin_factor → z_size frames for the model
  - Runs inference and saves the LAST output frame (= the chunk's last binned frame)

Handles edge cases:
  - First chunk: if not enough preceding frames, pads from the start of the scene
  - Cross-file boundaries: get_packed_patch handles seamlessly
  - Very last chunk: always included

Saves:
  - target/: raw binned last frame (last bin_factor binary frames summed / bin_factor)
  - denoised/: model output for the last frame

Usage:
    python infer_binned_last.py \
        --scene-dir D:/Yehe/PC/reconstruction/test/basement \
        --checkpoint models_spad/basement-bin8.ckpt \
        --device 0 --output output/basement-bin8-last
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
    """Convert an averaged photon frame to an 8-bit sRGB image."""
    frame_rgb = spc_avg_to_rgb(frame_np, factor=factor)
    frame_srgb = linearrgb_to_srgb(frame_rgb)
    return (np.clip(frame_srgb, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    """Denoise the last binned frame of each 1024-frame chunk."""
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
    parser.add_argument("--z-size", type=int, default=32)
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
    z_raw = z_size * bin_factor  # binary frames per model input (256)

    target_dir = Path(args.output) / "target"
    denoised_dir = Path(args.output) / "denoised"
    target_dir.mkdir(parents=True, exist_ok=True)
    denoised_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Scene: %s, %d chunks (%d frames each), bin_factor=%d, z_size=%d, z_raw=%d",
        scene.name,
        scene.n_files,
        scene.frames_per_file,
        bin_factor,
        z_size,
        z_raw,
    )

    with torch.no_grad():
        for fi in range(scene.n_files):
            # The last binary frame of this chunk
            chunk_end = (fi + 1) * scene.frames_per_file
            # We want z_raw binary frames ending at chunk_end
            t_start = chunk_end - z_raw

            # Handle first chunk: clamp to start of scene
            if t_start < 0:
                t_start = 0
                actual_raw = chunk_end
                # Pad: load what we can, then pad the front
                pad_needed = z_raw - actual_raw
            else:
                pad_needed = 0

            # Handle if total frames don't align (shouldn't happen but be safe)
            t_start = min(t_start, scene.total_frames - z_raw)
            t_start = max(t_start, 0)
            actual_load = min(z_raw, scene.total_frames - t_start)

            packed = scene.get_packed_patch(t_start, actual_load, 0, scene.height)
            packed_t = torch.from_numpy(packed).to(device)
            unpacked = gpu_unpackbits(packed_t, dim=2)[:, :, :800, :]  # (actual_load, H, W, C)

            # Pad front if needed (repeat first frame)
            if pad_needed > 0:
                pad = unpacked[:1].expand(pad_needed, -1, -1, -1)
                unpacked = torch.cat([pad, unpacked], dim=0)

            # Bin → (z_size, bin_factor, H, W, C) → sum → (z_size, H, W, C)
            binned = unpacked.reshape(z_size, bin_factor, scene.height, 800, scene.channels)
            binned = binned.sum(dim=1).float()

            # --- Target: last binned frame (raw, normalized to rate) ---
            target_frame = binned[-1].cpu().numpy() / bin_factor
            target_frame = np.clip(target_frame, 0, 1 - 1e-6).astype(np.float32)
            png = frame_to_png(target_frame, args.factor)
            Image.fromarray(png).save(target_dir / f"{fi:04d}.png")

            # --- Denoised: run model on all z_size binned frames, take last ---
            data = binned.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, Z, H, W)
            output = torch.exp(model(data))
            input_mean = data.mean()
            vol = output[0]
            vol = vol / vol.mean() * input_mean

            # Last frame, normalize to per-binary-frame rate
            last_frame = vol[:, -1, :, :].cpu().numpy().transpose(1, 2, 0) / bin_factor
            last_frame = np.clip(last_frame, 0, 1 - 1e-6).astype(np.float32)
            png = frame_to_png(last_frame, args.factor)
            Image.fromarray(png).save(denoised_dir / f"{fi:04d}.png")

            del packed_t, unpacked, binned, data, output
            torch.cuda.empty_cache()

            if (fi + 1) % 10 == 0 or fi == scene.n_files - 1:
                logger.info("Chunk %d/%d done", fi + 1, scene.n_files)

    logger.info("Done. %d frames saved to %s", scene.n_files, args.output)


if __name__ == "__main__":
    main()
