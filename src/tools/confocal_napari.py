"""Play real confocal observations and reconstructed 3D volumes through time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _read_npy_frame(path, index, reverse_z):
    """Copy one display volume, closing its file mapping before returning."""
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        volume = mapped[index, ::-1] if reverse_z else mapped[index]
        return np.array(volume, copy=True, order="C")
    finally:
        mapped._mmap.close()


class LiveSeries:
    """Update lazy time-series layers as complete inference chunks are published."""

    def __init__(self, viewer, root):
        self.viewer, self.root = viewer, root
        self.layers = []
        self.completed, self.complete = -1, False

    def _data(self, filename, frames, reverse_z):
        import dask.array as da
        from dask import delayed

        path = self.root / filename
        header = np.load(path, mmap_mode="r", allow_pickle=False)
        shape, dtype = header.shape[1:], header.dtype
        header._mmap.close()
        # from_array copies NumPy/memmap inputs in current Dask. Explicit
        # delayed frame reads keep construction and slicing truly bounded.
        return da.stack(
            [
                da.from_delayed(
                    delayed(_read_npy_frame, pure=False)(path, t, reverse_z),
                    shape=shape,
                    dtype=dtype,
                )
                for t in range(frames)
            ]
        )

    def refresh(self):
        """Expose only fully written frames, preserving the user's camera and time."""
        manifest_path, progress_path = self.root / "manifest.json", self.root / "progress.json"
        if not manifest_path.exists() or not progress_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        progress = json.loads(progress_path.read_text())
        frames, complete = progress["completed_frames"], progress["status"] == "complete"
        if frames <= 0 or (frames == self.completed and complete == self.complete):
            return
        first_load = not self.layers
        viewer = self.viewer
        current = viewer.dims.current_step[0] if self.layers else 0
        for i, filename in enumerate(("counts.npy", "expected_counts.npy")):
            data = self._data(filename, frames, manifest["reverse_z_for_display"])
            if i < len(self.layers):
                self.layers[i].data = data
            else:
                label = "Observed | photon counts" if i == 0 else "4D SSL | expected photon counts"
                self.layers.append(
                    viewer.add_image(
                        data,
                        name=label,
                        scale=(1.0, *manifest["voxel_spacing_zyx_um"]),
                        contrast_limits=(0, 0.5 if i else 2),
                        colormap="magma" if i else "gray",
                        rendering="mip",
                        interpolation3d="linear",
                    )
                )
        viewer.dims.axis_labels = ("time point", "Z (um)", "Y (um)", "X (um)")
        viewer.dims.set_current_step(0, min(current, frames - 1))
        qt_dims = viewer.window._qt_viewer.dims
        if qt_dims.is_playing:
            # napari 0.6 keeps its animation bounds when the data shape grows.
            # Expand only the default full range, preserving custom subranges.
            selected = qt_dims.slider_widgets[0].frame_range
            if selected is None or selected == (0, self.completed - 1):
                qt_dims.play(axis=0, frame_range=(0, frames - 1))
        self.completed, self.complete = frames, complete
        viewer.grid.enabled = True
        viewer.grid.shape = (1, 2)
        viewer.grid.stride = -1
        total = manifest["shape_tzyx"][0]
        viewer.title = f"Confocal SSL | {frames}/{total} volumes | {progress['status']}"
        if first_load:
            viewer.reset_view()
            viewer.camera.angles = (25, 30, 45)


def main():
    """Open the saved patch comparison or stream a whole-acquisition reconstruction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("data/confocal_gap4d_features32_batch4_v1")
    )
    parser.add_argument("--inference", type=Path)
    parser.add_argument("--example", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--cache-gib", type=float, default=1)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import napari
    from napari.utils import resize_dask_cache
    from qtpy.QtCore import QTimer

    viewer = napari.Viewer(
        title="Confocal SSL | raw observations and 4D reconstruction", ndisplay=3
    )
    # Window construction applies napari's persisted cache preference first.
    # Apply this process's explicit bound after that initialization.
    image_cache = resize_dask_cache(nbytes=int(max(args.cache_gib, 0) * 2**30))
    state_root = args.inference or args.study
    state_root.mkdir(parents=True, exist_ok=True)
    layers = []

    if args.inference is None:
        archive = np.load(args.study / "prediction_examples.npz", allow_pickle=False)
        index = args.example
        scale = (1.0, 0.42390867346938775, 0.12638641878669276, 0.12638641878669276)
        raw = archive["input"][index] + archive["target"][index]
        fields = (
            (raw[:, ::-1], "Observed C | photon counts", (0, 2), "gray"),
            (archive["conv3d"][index, :, ::-1], "3D | arrival probability", (0, 6e-7), "magma"),
            (archive["conv4d"][index, :, ::-1], "4D | arrival probability", (0, 6e-7), "magma"),
        )
        for data, label, limits, color in fields:
            layers.append(
                viewer.add_image(
                    data,
                    name=label,
                    scale=scale,
                    contrast_limits=limits,
                    colormap=color,
                    rendering="mip",
                    interpolation3d="linear",
                )
            )
        viewer.title = (
            f"Confocal SSL preview | patch {archive['example_indices'][index]} | no clean GT"
        )
        viewer.grid.enabled = True
        viewer.grid.shape = (1, 3)
        viewer.grid.stride = -1
    else:
        live = LiveSeries(viewer, args.inference)
        live.refresh()
        layers = live.layers
        refresh_timer = QTimer()
        refresh_timer.timeout.connect(live.refresh)
        refresh_timer.start(10000)

    if layers:
        viewer.dims.axis_labels = ("time point", "Z (um)", "Y (um)", "X (um)")
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        "Raw observations and SSL reconstruction | temporal context is not ground truth"
    )
    viewer.reset_view()
    viewer.camera.angles = (25, 30, 45)

    def record_state():
        state = {
            "pid": os.getpid(),
            "ndisplay": viewer.dims.ndisplay,
            "current_step": list(viewer.dims.current_step),
            "playing": viewer.window._qt_viewer.dims.is_playing,
            "layers": [{"name": layer.name, "shape": list(layer.data.shape)} for layer in layers],
            "inference": str(args.inference),
            "cpu_cache_limit_bytes": image_cache.cache.available_bytes,
            "cpu_cache_used_bytes": image_cache.cache.total_bytes,
        }
        path = state_root / "napari_state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(path)

    def start_playback():
        if layers:
            viewer.screenshot(path=state_root / "napari_preview.png", canvas_only=False)
            viewer.window._qt_viewer.dims.play(axis=0, fps=args.fps, loop_mode="loop")
        record_state()

    QTimer.singleShot(2500, start_playback)
    state_timer = QTimer()
    state_timer.timeout.connect(record_state)
    state_timer.start(5000)
    napari.run()


if __name__ == "__main__":
    main()
