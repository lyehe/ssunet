"""Inference with multi-offset SIFT averaging.

For each chunk:
1. Run model on all 25 offsets (off00-off24)
2. Each offset's target = a different original frame (999-1023)
3. SIFT-register all 25 denoised frames to the target (off24 = original 1023)
4. Weighted average → final output

Usage:
    python infer_uniform_merge.py --scene blue-bathroom --device 0
    python infer_uniform_merge.py --scene train-passby --device 1
"""

import argparse
import logging
from pathlib import Path

import cv2
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


def find_translation_ecc(
    source: np.ndarray,
    target: np.ndarray,
    max_shift: float = 5.0,
) -> np.ndarray | None:
    """ECC translation aligning source to target, or None if ECC fails or exceeds max_shift."""
    src_u8 = (np.clip(source, 0, 1) * 255).astype(np.uint8)
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)
    src_gray = cv2.cvtColor(cv2.GaussianBlur(src_u8, (0, 0), 5.0), cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(cv2.GaussianBlur(tgt_u8, (0, 0), 5.0), cv2.COLOR_RGB2GRAY)

    src_f = src_gray.astype(np.float32) / 255.0
    tgt_f = tgt_gray.astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(
            tgt_f,
            src_f,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None

    tx, ty = warp[0, 2], warp[1, 2]
    if abs(tx) > max_shift or abs(ty) > max_shift:
        return None
    return warp


def warp_frame(source: np.ndarray, warp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp each channel by an affine matrix and return the mask of covered pixels."""
    height, width, channels = source.shape
    warped = np.zeros_like(source)
    for c in range(channels):
        warped[:, :, c] = cv2.warpAffine(
            source[:, :, c],
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    mask = cv2.warpAffine(
        np.ones((height, width), dtype=np.float32),
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = (mask > 0.5).astype(np.float32)
    return warped, mask


def main() -> None:
    """Run inference and average predictions registered across offsets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--config", type=str, default="config_spad.yml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--frames-per-chunk", type=int, default=41)
    parser.add_argument("--n-offsets", type=int, default=25)
    parser.add_argument(
        "--n-merge", type=int, default=16, help="Number of frames to merge (from off24 backwards)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.85,
        help="Weight decay per frame distance from target",
    )
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    model_config = ModelConfig(**cfg["MODEL"])
    z_size = cfg["DATA"]["z_size"]

    ckpt = args.checkpoint or f"models_spad/{args.scene}_uniform_all.ckpt"
    data_dir = Path(f"D:/Yehe/PC/reconstruction/test/{args.scene}_step25")
    output_dir = Path(args.output or f"output/{args.scene}-uniform-merge")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device(f"cuda:{args.device}")
    model = Bit2Bit.load_from_checkpoint(ckpt, config=model_config, map_location=device)
    model.to(device)
    model.eval()

    # Load all offset data files
    all_data = []
    for off in range(args.n_offsets):
        path = data_dir / f"off{off:02d}.npy"
        all_data.append(np.load(str(path)))
    logger.info("Loaded %d offset files, shape %s each", len(all_data), all_data[0].shape)

    fpc = args.frames_per_chunk
    n_chunks = all_data[0].shape[0] // fpc

    # Which offsets to merge: last n_merge offsets (closest to target frame 1023)
    # off24 → orig 1023 (target), off23 → orig 1022, ..., off(25-n_merge) → orig(1024-n_merge)
    merge_offsets = list(range(args.n_offsets - args.n_merge, args.n_offsets))
    target_offset = args.n_offsets - 1  # off24
    logger.info(
        "Merging %d offsets: off%02d-off%02d (orig frames %d-%d), target=off%02d (orig 1023)",
        args.n_merge,
        merge_offsets[0],
        merge_offsets[-1],
        1024 - args.n_merge,
        1023,
        target_offset,
    )

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * fpc

            # Run inference on each offset and collect target frames
            denoised_frames = {}  # offset -> (H, W, C) float32
            for off in merge_offsets:
                data = all_data[off]
                t_start = chunk_start + fpc - z_size
                packed = data[t_start : chunk_start + fpc]

                packed_t = torch.from_numpy(packed).to(device)
                unpacked = gpu_unpackbits(packed_t, dim=2)[:, :, :800, :]
                batch = unpacked.permute(3, 0, 1, 2).unsqueeze(0).float()

                output = torch.exp(model(batch))
                input_mean = batch.mean()
                vol = output[0]
                vol = vol / vol.mean() * input_mean
                volume_np = vol.cpu().numpy().transpose(1, 2, 3, 0)
                volume_np = np.clip(volume_np, 0, 1 - 1e-6)

                denoised_frames[off] = volume_np[-1]  # last frame = target

                del packed_t, unpacked, batch, output

            # SIFT-register all frames to target (off24)
            target = denoised_frames[target_offset]
            height, width, _channels = target.shape

            accum = target.astype(np.float64).copy()
            weight_sum = np.ones((height, width, 1), dtype=np.float64)

            for off in merge_offsets:
                if off == target_offset:
                    continue

                source = denoised_frames[off]
                dist = target_offset - off  # distance in original frames
                w = args.weight_decay**dist

                warp = find_translation_ecc(source, target)
                if warp is not None:
                    warped, mask = warp_frame(source, warp)
                else:
                    warped, mask = source.copy(), np.ones((height, width), dtype=np.float32)

                # Skip if warped is much brighter (registration failure)
                tgt_mean = target.mean()
                warp_mean = warped[mask > 0.5].mean() if (mask > 0.5).any() else 0
                if tgt_mean > 0 and warp_mean / tgt_mean > 1.5:
                    continue

                mask_3d = mask[:, :, None].astype(np.float64)
                accum += warped.astype(np.float64) * mask_3d * w
                weight_sum += mask_3d * w

            merged = (accum / weight_sum).astype(np.float32)

            # Save
            png = frame_to_png(merged, args.factor)
            Image.fromarray(png).save(output_dir / f"{chunk_idx:06d}.png")
            np.save(str(output_dir / f"{chunk_idx:06d}.npy"), merged)

            logger.info(
                "Saved %06d.png (chunk %d/%d, merged %d frames)",
                chunk_idx,
                chunk_idx + 1,
                n_chunks,
                args.n_merge,
            )

            torch.cuda.empty_cache()

    logger.info("Done. %d chunks saved to %s", n_chunks, output_dir)


if __name__ == "__main__":
    main()
