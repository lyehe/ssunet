"""Inference for models trained on preprocessed uniform-sampled SPAD data.

Loads the preprocessed off24.npy (41 frames/chunk, step-25 sampled),
feeds z_size=32 frames ending at the last frame of each chunk (the target),
and saves the output PNG per chunk.

Usage:
    python infer_uniform.py --scene blue-bathroom --device 0
    python infer_uniform.py --scene train-passby --device 1
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
from src.ssunet.datasets.spad import gpu_unpackbits
from src.ssunet.models import Bit2Bit

torch.serialization.add_safe_globals([ModelConfig])


def frame_to_png(frame_np: np.ndarray, factor: float) -> np.ndarray:
    """Convert an averaged photon frame to an 8-bit sRGB image."""
    frame_rgb = spc_avg_to_rgb(frame_np, factor=factor)
    frame_srgb = linearrgb_to_srgb(frame_rgb)
    return (np.clip(frame_srgb, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    """Run inference with a model trained on uniform-sampled SPAD data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Inference for uniform-trained SPAD models")
    parser.add_argument(
        "--scene", type=str, required=True, help="Scene name (blue-bathroom or train-passby)"
    )
    parser.add_argument("--config", type=str, default="config_spad.yml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: models_spad/<scene>_uniform.ckpt)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output dir (default: output/<scene>-uniform-trained)",
    )
    parser.add_argument(
        "--offset", type=int, default=24, help="Sampling offset (default: 24 = primary)"
    )
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--frames-per-chunk", type=int, default=41)
    args = parser.parse_args()

    # Load config
    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    model_config = ModelConfig(**cfg["MODEL"])
    z_size = cfg["DATA"]["z_size"]  # 32

    # Paths
    ckpt = args.checkpoint or f"models_spad/{args.scene}_uniform.ckpt"
    data_path = Path(f"D:/Yehe/PC/reconstruction/test/{args.scene}_step25/off{args.offset:02d}.npy")
    output_dir = Path(args.output or f"output/{args.scene}-uniform-trained")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device(f"cuda:{args.device}")
    model = Bit2Bit.load_from_checkpoint(ckpt, config=model_config, map_location=device)
    model.to(device)
    model.eval()

    # Load preprocessed data
    data = np.load(str(data_path))  # (N_total, H, W_packed, C)
    fpc = args.frames_per_chunk
    n_chunks = data.shape[0] // fpc

    logger.info(
        "Scene: %s, data: %s, shape: %s, %d chunks x %d frames",
        args.scene,
        data_path.name,
        data.shape,
        n_chunks,
        fpc,
    )

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * fpc
            chunk_end = chunk_start + fpc  # 41 frames

            # Take last z_size frames from the chunk (frames 9..40 for z=32, fpc=41)
            # Target = last frame (index fpc-1 = 40)
            t_start = chunk_end - z_size
            packed = data[t_start:chunk_end]  # (z_size, H, W_packed, C)

            packed_t = torch.from_numpy(packed).to(device)
            unpacked = gpu_unpackbits(packed_t, dim=2)[:, :, :800, :]  # (Z, H, W, C)
            batch = unpacked.permute(3, 0, 1, 2).unsqueeze(0).float()  # (1, C, Z, H, W)

            output = torch.exp(model(batch))

            input_mean = batch.mean()
            vol = output[0]
            vol = vol / vol.mean() * input_mean
            volume_np = vol.cpu().numpy().transpose(1, 2, 3, 0)  # (Z, H, W, C)
            volume_np = np.clip(volume_np, 0, 1 - 1e-6)

            # Target = last frame in the volume
            target_frame = volume_np[-1]
            png = frame_to_png(target_frame, args.factor)
            Image.fromarray(png).save(output_dir / f"{chunk_idx:06d}.png")

            # Also save raw npy
            np.save(str(output_dir / f"{chunk_idx:06d}.npy"), target_frame)

            logger.info("Saved %06d.png (chunk %d/%d)", chunk_idx, chunk_idx + 1, n_chunks)

            del packed_t, unpacked, batch, output
            torch.cuda.empty_cache()

    logger.info("Done. %d chunks saved to %s", n_chunks, output_dir)


if __name__ == "__main__":
    main()
