"""Sliding-window inference with temporal offset averaging for SPAD data.

For the target chunk, runs inference at 16 temporal offsets (-8 to +7),
caching each full volume. This gives 16 aligned predictions per frame
from different z-window contexts, which can be averaged for smoother results.

Usage:
    # GPU (fast):
    python infer_spad_offsets.py --scene-dir D:/Yehe/PC/reconstruction/train/attic \
        --ckpt models_spad/attic.ckpt --device 0

    # CPU:
    python infer_spad_offsets.py --scene-dir D:/Yehe/PC/reconstruction/train/attic \
        --ckpt models_spad/attic.ckpt --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.ssunet.configs.model_config import ModelConfig
from src.ssunet.inference import (
    average_temporal_offsets,
    logits_to_expected_counts,
    model_distribution_dims,
    reflected_window,
    temporal_window_size,
)
from src.ssunet.models import Bit2Bit

torch.serialization.add_safe_globals([ModelConfig])

Z_SIZE = 32
N_OFFSETS = 16  # offsets 0..15, mapped to -8..+7


def load_and_unpack(npy_path: str) -> np.ndarray:
    """Load bitpacked chunk and unpack to float32 binary.

    Returns: (T, H, W, C) float32 with values in {0.0, 1.0}.
    """
    packed = np.load(npy_path)
    unpacked = np.unpackbits(packed, axis=2)[:, :, :800, :]
    return unpacked.astype(np.float32)


def infer_zwindow(model: torch.nn.Module, data: np.ndarray, device: torch.device) -> np.ndarray:
    """Run inference on a single z-window.

    Args:
        model: Bit2Bit model.
        data: (Z, H, W, C) float32 binary input.
        device: torch device.

    Returns:
        (Z, H, W, C) float32 expected counts, scaled to the input count.
    """
    if data.ndim != 4 or any(size == 0 for size in data.shape):
        message = "Expected a nonempty (time, height, width, channel) volume"
        raise ValueError(message)
    original_depth = data.shape[0]
    depth = temporal_window_size(model, original_depth)
    if depth != original_depth:
        data = reflected_window(data, 0, depth)
    # (Z, H, W, C) -> (1, C, Z, H, W)
    x = torch.from_numpy(np.ascontiguousarray(data.transpose(3, 0, 1, 2)[None])).to(
        device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        logits = model(x)[:, :, :original_depth]
        out = (
            logits_to_expected_counts(
                logits, x[:, :, :original_depth], dim=model_distribution_dims(model)
            )[0]
            .float()
            .cpu()
        )
    return out.numpy().transpose(1, 2, 3, 0)  # (Z, H, W, C)


def main():
    """Denoise a chunk with aligned temporal contexts and export its target frame."""
    parser = argparse.ArgumentParser(description="Offset-averaged inference for SPAD")
    parser.add_argument("--scene-dir", type=str, required=True)
    parser.add_argument("--chunk", type=int, default=0, help="Chunk index to process")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/<scene_name>/offsets)",
    )
    parser.add_argument(
        "--device", type=str, default="0", help="Device: GPU index (0, 1, ...) or 'cpu'"
    )
    parser.add_argument(
        "--factor", type=float, default=0.5, help="SPC response function factor (default: 0.5)"
    )
    parser.add_argument(
        "--target-frame",
        type=str,
        default="last",
        help="Which frame to save as PNG: 'last' (default) or frame index",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    # Resolve output dir
    scene_dir = Path(args.scene_dir)
    scene_name = scene_dir.name
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("output") / scene_name / "offsets"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = scene_dir / f"{args.chunk:06d}.npy"

    def log(msg, **kwargs):
        print(msg, **kwargs)
        sys.stdout.flush()

    log(f"Scene: {scene_name}, chunk: {args.chunk}, device: {device}")
    log(f"Loading chunk: {chunk_path}")

    # Load model
    log("Loading model...")
    model = Bit2Bit.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    model.eval()

    # Load and unpack target chunk
    data = load_and_unpack(str(chunk_path))  # (1024, 800, 800, 3)
    length = data.shape[0]
    window_size = temporal_window_size(model, Z_SIZE)
    log(f"Chunk shape: {data.shape}, context depth: {window_size} frames")

    pad_size = N_OFFSETS // 2  # 8
    offsets = tuple(range(-pad_size, N_OFFSETS - pad_size))
    total_calls = sum(
        len(range(offset + ((-offset) // window_size) * window_size, length, window_size))
        for offset in offsets
    )
    call_idx = 0
    t_start = time.time()

    def window_done(offset: int, start: int) -> None:
        nonlocal call_idx
        call_idx += 1
        elapsed = time.time() - t_start
        eta = elapsed / call_idx * (total_calls - call_idx)
        log(
            f"  offset {offset:+d}, start {start:4d} | {call_idx}/{total_calls} | "
            f"ETA: {eta / 60:.0f}min"
        )

    def save_offset(offset: int, volume: np.ndarray) -> None:
        offset_path = output_dir / f"offset_{offset:+03d}.npy"
        np.save(str(offset_path), volume)
        log(f"  Saved: {offset_path} ({volume.nbytes / 1e9:.1f} GB)")

    avg, coverage = average_temporal_offsets(
        data,
        lambda window: infer_zwindow(model, window, device),
        window_size=window_size,
        offsets=offsets,
        on_offset=save_offset,
        on_window=window_done,
    )
    log(f"Per-frame prediction coverage: {coverage.min()}..{coverage.max()}")

    # Save raw averaged volume
    avg_path = output_dir / "averaged.npy"
    np.save(str(avg_path), avg)

    # Determine target frame index for PNG output
    if args.target_frame == "last":
        target_idx = length - 1
    else:
        target_idx = int(args.target_frame)

    if not 0 <= target_idx < length:
        message = f"Target frame must be between 0 and {length - 1}"
        raise ValueError(message)
    # Each window was already scaled to expected counts. Preserve temporal
    # intensity variation instead of forcing every selected frame to the mean.
    frame = avg[target_idx]  # (H, W, C)
    frame = np.clip(frame, 0, 1 - 1e-6)

    # The simulator is only needed for color export, not inference/alignment.
    from visionsim.emulate.spc import spc_avg_to_rgb
    from visionsim.utils.color import linearrgb_to_srgb

    # Invert SPC response function: rgb = -log(1 - p) / factor
    frame_rgb = spc_avg_to_rgb(frame, factor=args.factor)

    # Apply sRGB gamma correction
    frame_srgb = linearrgb_to_srgb(frame_rgb)

    # Save float32 npy (linear RGB)
    np.save(str(output_dir / f"{args.chunk:06d}.npy"), frame_rgb.astype(np.float32))

    # Save submission-ready PNG
    frame_uint8 = (np.clip(frame_srgb, 0, 1) * 255).astype(np.uint8)
    png_path = output_dir / f"{args.chunk:06d}.png"
    Image.fromarray(frame_uint8).save(png_path)
    log(f"Saved PNG: {png_path} (frame {target_idx})")

    elapsed_total = time.time() - t_start
    log(f"\nDone! Total time: {elapsed_total / 3600:.1f} hours")
    log(f"Cached {N_OFFSETS} offset volumes in: {output_dir}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
