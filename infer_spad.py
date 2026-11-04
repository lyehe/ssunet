"""Inference script for SPAD scenes.

For each chunk (npy file), runs the model on a z_size window centered at the
last frame. Nearby frames in the output volume are registered to the target
and averaged for noise reduction.

Registration methods:
  - sift: SIFT feature matching + homography warp (best for static scenes
    with small camera motion)
  - raft: RAFT deep optical flow (best for scenes with local deformation)
  - none: no registration, single target frame only

Usage:
    python infer_spad.py --scene-dir D:/Yehe/PC/reconstruction/test/basement \
        --checkpoint models_spad/basement.ckpt --device 0 --output output/basement \
        --method sift raft
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
from src.ssunet.datasets.spad import SPADScene, gpu_unpackbits

torch.serialization.add_safe_globals([ModelConfig])
from src.ssunet.models import Bit2Bit


def _find_ecc_transform(
    source: np.ndarray, target: np.ndarray,
    max_shift: float = 10.0, max_rotation_deg: float = 2.0,
) -> np.ndarray | None:
    """Find transform from source to target using cascade ECC.

    Tries Euclidean (3 DOF: rotation + translation) first, then falls back
    to translation-only (2 DOF) if Euclidean fails or produces large rotation.

    Returns 2x3 affine matrix or None.
    """
    # Preprocess: blur and convert to grayscale
    src_u8 = (np.clip(source, 0, 1) * 255).astype(np.uint8)
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)
    src_gray = cv2.cvtColor(cv2.GaussianBlur(src_u8, (0, 0), 5.0), cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(cv2.GaussianBlur(tgt_u8, (0, 0), 5.0), cv2.COLOR_RGB2GRAY)

    src_f = src_gray.astype(np.float32) / 255.0
    tgt_f = tgt_gray.astype(np.float32) / 255.0

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

    # Try Euclidean first (3 DOF: rotation + translation)
    M_euc = np.eye(2, 3, dtype=np.float32)
    try:
        _, M_euc = cv2.findTransformECC(
            tgt_f, src_f, M_euc, cv2.MOTION_EUCLIDEAN, criteria,
            inputMask=None, gaussFiltSize=5,
        )
        angle = np.degrees(np.arctan2(M_euc[1, 0], M_euc[0, 0]))
        tx, ty = M_euc[0, 2], M_euc[1, 2]
        if abs(angle) <= max_rotation_deg and abs(tx) <= max_shift and abs(ty) <= max_shift:
            return M_euc
    except cv2.error:
        pass

    # Fall back to translation-only (2 DOF)
    M_tr = np.eye(2, 3, dtype=np.float32)
    try:
        _, M_tr = cv2.findTransformECC(
            tgt_f, src_f, M_tr, cv2.MOTION_TRANSLATION, criteria,
            inputMask=None, gaussFiltSize=5,
        )
        tx, ty = M_tr[0, 2], M_tr[1, 2]
        if abs(tx) <= max_shift and abs(ty) <= max_shift:
            return M_tr
    except cv2.error:
        pass

    return None


def warp_with_ecc(
    source: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp source to target using cascade ECC (Euclidean → translation).

    Args:
        source: (H, W, C) float32 source frame.
        target: (H, W, C) float32 target frame.

    Returns:
        (warped, mask) or (None, None) if registration failed.
    """
    H, W, C = target.shape

    M = _find_ecc_transform(source, target)
    if M is None:
        return None, None

    warped = np.zeros_like(source)
    for c in range(C):
        warped[:, :, c] = cv2.warpAffine(
            source[:, :, c], M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    mask = cv2.warpAffine(
        np.ones((H, W), dtype=np.float32), M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    mask = (mask > 0.5).astype(np.float32)

    return warped, mask


# Keep old name as alias for compatibility
warp_with_homography = warp_with_ecc


def warp_with_raft(
    source: np.ndarray, target: np.ndarray,
    raft_model: torch.nn.Module, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp source to target using RAFT optical flow.

    Args:
        source: (H, W, C) float32 source frame.
        target: (H, W, C) float32 target frame.
        raft_model: RAFT model on device.
        device: torch device.

    Returns:
        (warped, mask): warped source and validity mask.
    """
    H, W, C = target.shape

    # RAFT expects (B, 3, H, W) float32 in [0, 1], images must be
    # divisible by 8 — pad if needed
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    src_t = torch.from_numpy(source.transpose(2, 0, 1)).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(target.transpose(2, 0, 1)).unsqueeze(0).to(device)

    if pad_h > 0 or pad_w > 0:
        src_t = torch.nn.functional.pad(src_t, (0, pad_w, 0, pad_h), mode="replicate")
        tgt_t = torch.nn.functional.pad(tgt_t, (0, pad_w, 0, pad_h), mode="replicate")

    # RAFT expects uint8-range [0,255] input
    with torch.no_grad():
        flow_list = raft_model(src_t * 255, tgt_t * 255)
    flow = flow_list[-1][0].cpu().numpy()  # (2, H_pad, W_pad)

    # Crop padding
    flow = flow[:, :H, :W]  # (2, H, W)
    flow = flow.transpose(1, 2, 0)  # (H, W, 2)

    # Build remap coordinates
    coords_x, coords_y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    map_x = coords_x + flow[:, :, 0]
    map_y = coords_y + flow[:, :, 1]

    # Warp float32 source per channel
    warped = np.zeros_like(source)
    for c in range(C):
        warped[:, :, c] = cv2.remap(
            source[:, :, c], map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    mask = ((map_x >= 0) & (map_x < W) & (map_y >= 0) & (map_y < H)).astype(np.float32)
    return warped, mask


def register_and_average(
    volume: np.ndarray,
    target_idx: int,
    method: str = "sift",
    before: int = 8,
    after: int = 7,
    weight_decay: float = 0.7,
    outlier_sigma: float = 2.0,
    raft_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Register nearby frames to target with per-pixel outlier rejection.

    Uses cascade ECC (Euclidean → translation) for registration.
    Rejects individual pixels that deviate from the median across frames,
    rather than discarding entire frames.

    Args:
        volume: (Z, H, W, C) float32 denoised volume.
        target_idx: index of the target frame.
        method: "sift" or "raft".
        before: frames before target to include.
        after: frames after target to include.
        weight_decay: weight decay per frame distance.
        outlier_sigma: per-pixel rejection threshold (MAD multiples).
        raft_model: RAFT model (required if method="raft").
        device: torch device (required if method="raft").

    Returns:
        (H, W, C) float32 averaged frame.
    """
    Z, H, W, C = volume.shape
    target = volume[target_idx]

    lo = max(0, target_idx - before)
    hi = min(Z, target_idx + after + 1)

    # Pass 1: register all frames
    registered = []  # list of (fi, warped, mask)
    for fi in range(lo, hi):
        if fi == target_idx:
            continue

        source = volume[fi]

        if method == "sift":
            warped, mask = warp_with_ecc(source, target)
        elif method == "raft":
            warped, mask = warp_with_raft(source, target, raft_model, device)
        else:
            warped, mask = source.copy(), np.ones((H, W), dtype=np.float32)

        if warped is None:
            continue

        registered.append((fi, warped, mask))

    if not registered:
        return target

    # Pass 2: per-pixel outlier rejection
    # Stack all registered frames + target
    n = len(registered)
    stack = np.zeros((n + 1, H, W, C), dtype=np.float32)
    stack[0] = target
    for i, (fi, warped, mask) in enumerate(registered):
        stack[i + 1] = warped

    # Per-pixel median and MAD across frames
    median_img = np.median(stack, axis=0)  # (H, W, C)
    abs_dev = np.abs(stack - median_img)   # (N+1, H, W, C)
    mad = np.median(abs_dev, axis=0)       # (H, W, C)
    threshold = outlier_sigma * np.maximum(mad, 1e-4)  # (H, W, C)

    # Pass 3: weighted average with per-pixel rejection
    accum = target.astype(np.float64).copy()
    weight_sum = np.ones((H, W, C), dtype=np.float64)

    for i, (fi, warped, mask) in enumerate(registered):
        dist = abs(fi - target_idx)
        w = weight_decay ** dist

        # Per-pixel validity: spatial mask AND within outlier threshold
        pixel_ok = (abs_dev[i + 1] < threshold).astype(np.float64)  # (H, W, C)
        mask_3c = np.broadcast_to(mask[:, :, None], (H, W, C)).astype(np.float64)
        combined = pixel_ok * mask_3c * w

        accum += warped.astype(np.float64) * combined
        weight_sum += combined

    return (accum / np.maximum(weight_sum, 1e-8)).astype(np.float32)


def frame_to_png(frame_np: np.ndarray, factor: float) -> np.ndarray:
    """Convert denoised photon rate to sRGB uint8.

    Args:
        frame_np: (H, W, C) float32 photon probability in [0, 1).
        factor: SPC response function factor.

    Returns:
        (H, W, C) uint8 sRGB image.
    """
    frame_rgb = spc_avg_to_rgb(frame_np, factor=factor)
    frame_srgb = linearrgb_to_srgb(frame_rgb)
    return (np.clip(frame_srgb, 0, 1) * 255).astype(np.uint8)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="SPAD inference")
    parser.add_argument("--scene-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config_spad.yml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--factor", type=float, default=0.5,
                        help="SPC response function factor (default: 0.5)")
    parser.add_argument("--reg-before", type=int, default=8,
                        help="Register N frames before target (default: 8)")
    parser.add_argument("--reg-after", type=int, default=7,
                        help="Register N frames after target (default: 7)")
    parser.add_argument("--method", type=str, nargs="+", default=["sift", "raft"],
                        choices=["none", "sift", "raft"],
                        help="Registration method(s) (default: sift raft)")
    parser.add_argument("--uniform-sample", action="store_true",
                        help="Uniformly sample z_size frames across each chunk "
                             "(for period-4 corrupted data)")
    args = parser.parse_args()

    # Load config for model settings
    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    model_config = ModelConfig(**cfg["MODEL"])
    z_size = cfg["DATA"]["z_size"]  # 32

    # Load denoising model
    device = torch.device(f"cuda:{args.device}")
    model = Bit2Bit.load_from_checkpoint(args.checkpoint, config=model_config, map_location=device)
    model.to(device)
    model.eval()

    # Load RAFT if needed
    raft_model = None
    if "raft" in args.method:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        logger.info("RAFT model loaded")

    # Create output dirs per method
    output_base = Path(args.output)
    output_dirs = {}
    for m in args.method:
        if m == "none":
            d = output_base
        else:
            d = output_base.parent / f"{output_base.name}-{m}"
        d.mkdir(parents=True, exist_ok=True)
        output_dirs[m] = d

    # Also always save the raw (no registration) output
    if "none" not in args.method:
        output_base.mkdir(parents=True, exist_ok=True)

    # Load scene
    scene_dir = Path(args.scene_dir)
    scene = SPADScene(scene_dir)
    n_chunks = scene.n_files
    fpf = scene.frames_per_file  # 1024

    logger.info(
        "Scene: %s (%d chunks, %d frames/chunk, z_size=%d, methods=%s, uniform=%s)",
        scene.name, n_chunks, fpf, z_size, args.method, args.uniform_sample,
    )

    half_z = z_size // 2

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            target_frame = (chunk_idx + 1) * fpf - 1
            chunk_start = chunk_idx * fpf

            if args.uniform_sample:
                # All chunks conceptually padded to 1025 = 25*41 and concatenated.
                # Step 25 is odd (coprime with 4) and 4*25=100 > 96 (block size),
                # guaranteeing independence for period-4 corrupted data.
                # Target = last frame of each padded chunk = chunk_idx*1025 + 1024.
                # Sample z_size frames ending at the target, step 25.
                padded_fpf = fpf + 1  # 1025
                target_padded = chunk_idx * padded_fpf + (padded_fpf - 1)

                # z_size frames ending at target, step 25
                padded_indices = np.arange(
                    target_padded, target_padded - z_size * 25, -25
                )[::-1]

                # Map padded indices back to real frame indices
                # padded pos p -> chunk = p // 1025, local = p % 1025
                # local 1024 = duplicate of local 1023
                global_indices = []
                for p in padded_indices:
                    c = p // padded_fpf
                    local = p % padded_fpf
                    if local >= fpf:
                        local = fpf - 1  # duplicate -> original last frame
                    global_indices.append(c * fpf + local)

                # Load each sampled frame and stack
                parts = []
                for gi in global_indices:
                    gi = max(0, min(gi, scene.total_frames - 1))
                    packed_frame = scene.get_packed_patch(gi, 1, 0, scene.height)
                    parts.append(packed_frame)
                packed = np.concatenate(parts, axis=0)

                frame_idx = len(global_indices) - 1  # target is last
                frame_desc = f"uniform step-25, {len(global_indices)} frames, target={global_indices[-1]}"
            else:
                t_start = target_frame - half_z
                t_end = target_frame + half_z

                if t_start < 0:
                    t_start = 0
                    t_end = z_size
                if t_end > scene.total_frames:
                    t_end = scene.total_frames
                    t_start = t_end - z_size

                packed = scene.get_packed_patch(t_start, z_size, 0, scene.height)
                frame_idx = target_frame - t_start
                frame_desc = f"frames {t_start}-{t_start + z_size - 1}"

            packed_t = torch.from_numpy(packed).to(device)
            unpacked = gpu_unpackbits(packed_t, dim=2)[:, :, :800, :]
            data = unpacked.permute(3, 0, 1, 2).unsqueeze(0).float()

            output = torch.exp(model(data))

            input_mean = data.mean()
            vol = output[0]
            vol = vol / vol.mean() * input_mean
            volume_np = vol.cpu().numpy().transpose(1, 2, 3, 0)  # (Z, H, W, C)
            volume_np = np.clip(volume_np, 0, 1 - 1e-6)

            # Save raw (no registration)
            raw_frame = volume_np[frame_idx]
            raw_uint8 = frame_to_png(raw_frame, args.factor)
            Image.fromarray(raw_uint8).save(output_base / f"{chunk_idx:06d}.png")

            # Save npy
            np.save(str(output_base / f"{chunk_idx:06d}.npy"), volume_np[frame_idx])

            # Save registered versions
            for m in args.method:
                if m == "none":
                    continue

                avg_frame = register_and_average(
                    volume_np, frame_idx,
                    method=m,
                    before=args.reg_before,
                    after=args.reg_after,
                    raft_model=raft_model,
                    device=device,
                )
                avg_uint8 = frame_to_png(avg_frame, args.factor)
                Image.fromarray(avg_uint8).save(output_dirs[m] / f"{chunk_idx:06d}.png")

            logger.info(
                "Saved %06d.png (%s, target=%d)",
                chunk_idx, frame_desc, target_frame,
            )

            del packed_t, unpacked, data, output
            torch.cuda.empty_cache()

    logger.info("Done. %d chunks saved to %s (+%s)", n_chunks, output_base, list(output_dirs.keys()))


if __name__ == "__main__":
    main()
