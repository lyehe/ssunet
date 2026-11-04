"""Napari viewer for raw SPAD data.

Usage:
    python view_spad.py D:/Yehe/PC/reconstruction/test/blue-bathroom
    python view_spad.py D:/Yehe/PC/reconstruction/train/attic --chunk 0 --frames 100
"""

import argparse
from pathlib import Path

import napari
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="View raw SPAD data in napari")
    parser.add_argument("scene_dir", type=str, help="Path to scene directory")
    parser.add_argument("--chunk", type=int, default=None, help="Chunk (npy file) index (default: all)")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to load per chunk (default: all 1024)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    npy_files = sorted(scene_dir.glob("*.npy"))
    scene_name = scene_dir.name

    if args.chunk is not None:
        npy_files = [npy_files[args.chunk]]

    # Load and concatenate all chunks
    all_frames = []
    for npy_path in npy_files:
        packed = np.load(str(npy_path))
        n_frames = args.frames or packed.shape[0]
        packed = packed[:n_frames]
        unpacked = np.unpackbits(packed, axis=2)[:, :, :800, :]
        all_frames.append(unpacked)
        print(f"Loaded {npy_path.name} ({n_frames} frames)")

    unpacked = np.concatenate(all_frames, axis=0)  # (total_T, 800, 800, 3)
    total_frames = unpacked.shape[0]
    print(f"Total: {total_frames} frames from {len(npy_files)} chunks")

    # Temporal sum for overview
    temporal_sum = unpacked.sum(axis=0).astype(np.float32)

    viewer = napari.Viewer(title=f"SPAD: {scene_name} ({len(npy_files)} chunks, {total_frames} frames)")

    # Add binary frames as image layer (T, H, W, C)
    viewer.add_image(
        unpacked,
        name=f"binary frames ({total_frames})",
        rgb=True,
        contrast_limits=[0, 1],
    )

    # Add temporal sum
    viewer.add_image(
        temporal_sum,
        name=f"sum ({total_frames} frames)",
        rgb=True,
        visible=False,
    )

    # Load ground truth PNG if it exists
    png_path = scene_dir / f"{args.chunk:06d}.png"
    if png_path.exists():
        from PIL import Image
        gt = np.array(Image.open(str(png_path)))
        viewer.add_image(gt, name="ground truth", rgb=True, visible=False)

    napari.run()


if __name__ == "__main__":
    main()
