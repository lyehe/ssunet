"""Register nearby inference frames to the target frame and average.

For each target PNG, loads neighboring frames, estimates optical flow to
warp them onto the target, then averages in the overlapping region.

Usage:
    python register_average.py --input-dir output/basement --neighbors 2
    python register_average.py --input-dir output/blue-bathroom --neighbors 3 --output-dir output/blue-bathroom-avg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def register_and_warp(target: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Register source to target using optical flow, return warped image and valid mask.

    Uses Farneback dense optical flow to compute per-pixel displacement,
    then warps source onto target's coordinate system.

    Args:
        target: (H, W, 3) uint8 target image.
        source: (H, W, 3) uint8 source image.

    Returns:
        (warped, mask): warped source image and float32 validity mask [0,1].
    """
    H, W = target.shape[:2]

    # Convert to grayscale for flow estimation
    gray_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    gray_source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)

    # Dense optical flow: source -> target
    flow = cv2.calcOpticalFlowFarneback(
        gray_source, gray_target,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=15,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )

    # Build remap coordinates
    coords_x, coords_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                       np.arange(H, dtype=np.float32))
    map_x = coords_x + flow[:, :, 0]
    map_y = coords_y + flow[:, :, 1]

    # Warp source to target space
    warped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Validity mask: 1 where remapped coordinates are within bounds
    mask = ((map_x >= 0) & (map_x < W) & (map_y >= 0) & (map_y < H)).astype(np.float32)

    return warped, mask


def main():
    parser = argparse.ArgumentParser(description="Register and average nearby frames")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with inference PNGs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <input-dir>-registered)")
    parser.add_argument("--before", type=int, default=8,
                        help="Number of frames before target (default: 8)")
    parser.add_argument("--after", type=int, default=7,
                        help="Number of frames after target (default: 7)")
    parser.add_argument("--weight-decay", type=float, default=0.7,
                        help="Weight decay per frame distance (default: 0.7)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}-registered"
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    n = len(png_files)
    print(f"Scene: {input_dir.name}, {n} frames, -{args.before}/+{args.after} neighbors")

    for i, target_path in enumerate(png_files):
        target = cv2.imread(str(target_path))
        if target is None:
            print(f"  Skipping {target_path.name}: failed to read")
            continue

        H, W, C = target.shape
        accum = target.astype(np.float64)
        weight_sum = np.ones((H, W, 1), dtype=np.float64)

        # Register neighbors: -before to +after
        count = 0
        for offset in range(-args.before, args.after + 1):
            if offset == 0:
                continue
            j = i + offset
            if j < 0 or j >= n:
                continue

            source = cv2.imread(str(png_files[j]))
            if source is None:
                continue

            warped, mask = register_and_warp(target, source)

            # Weight decays with distance
            w = args.weight_decay ** abs(offset)
            mask_3d = mask[:, :, None]
            accum += warped.astype(np.float64) * mask_3d * w
            weight_sum += mask_3d * w
            count += 1

        # Average
        avg = (accum / weight_sum).clip(0, 255).astype(np.uint8)

        out_path = output_dir / target_path.name
        cv2.imwrite(str(out_path), avg)
        print(f"  {target_path.name}: averaged with {count} neighbors")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
