"""Compare both detectors' observed and reconstructed scored patches in napari."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ..ssunet.confocal_display import configure_detector_layout, detector_color


class PredictionBrowser:
    """Display observed counts and count-calibrated predictions at identical coordinates."""

    def __init__(self, viewer, root, result, geometry, phase, display, maximum, layout=None):
        self.viewer, self.root, self.result, self.geometry = viewer, root, result, geometry
        self.phase, self.display, self.maximum = phase, display, maximum
        self.layout = layout
        self.frames = 0
        self.metadata = {}

    def source_start(self):
        """Resolve absolute clip coordinates or offsets into the original training cache."""
        if "source_start_tzyx" in self.metadata:
            return self.metadata["source_start_tzyx"]
        crop = self.result["identity"]["config"]["source"]["crop_tzyx"]
        return [
            local + pair[0] for local, pair in zip(self.metadata["start_tzyx"], crop, strict=True)
        ]

    def refresh(self, *_):
        """Switch phases without retaining full movies or treating the observation as GT."""
        from qtpy.QtCore import QTimer

        viewer = self.viewer
        viewer.window._qt_viewer.dims.stop()
        if viewer.layers:
            viewer.window._qt_viewer.canvas._scene_canvas.set_current()
        viewer.layers.clear()
        selection = self.phase.currentData() or self.phase.currentText()
        location = self.root / "examples" / selection
        observed = np.load(location / "observed.npy", mmap_mode="r", allow_pickle=False)
        predicted = np.load(
            location / "gap4d_expected_counts.npy", mmap_mode="r", allow_pickle=False
        )
        metadata = json.loads((location / "metadata.json").read_text())
        self.metadata = metadata
        phase = metadata["phase"]
        if observed.shape != predicted.shape:
            message = "Observed and predicted patches must have matching CTZYX coordinates"
            raise ValueError(message)
        movie = self.display.currentIndex() == 0
        spacing = tuple(self.geometry["voxel_spacing_zyx_um"])
        names = self.result["identity"]["channel_names"]
        for values, kind in ((observed, "observed, noisy"), (predicted, "GAP expected counts")):
            for channel, name in enumerate(names):
                image = values[channel]
                if self.geometry["reverse_z_for_display"]:
                    image = image[:, ::-1]
                if not movie:
                    image = image.mean(axis=0, dtype=np.float32)
                viewer.add_image(
                    image,
                    name=f"{name} | {kind}",
                    scale=(1.0, *spacing) if movie else spacing,
                    contrast_limits=(0, self.maximum.value()),
                    colormap=detector_color(name),
                    blending="additive",
                    rendering="mip",
                    interpolation3d="linear",
                )
        overlay = self.layout is None or self.layout.currentIndex() == 0
        configure_detector_layout(viewer, len(names), comparisons=2, overlay=overlay)
        viewer.dims.axis_labels = (
            ("time point", "Z (um)", "Y (um)", "X (um)")
            if movie
            else ("Z (um)", "Y (um)", "X (um)")
        )
        start = self.source_start()
        self.frames = observed.shape[1]
        recording = metadata.get("recording_name", "Microtubule 3")
        viewer.title = f"{recording} | {phase} | both detectors | observations and joint GAP4D"
        viewer.text_overlay.text = (
            f"Scored patch: T={start[0]}..{start[0] + self.frames - 1}, Z={start[1]}, "
            f"Y={start[2]}, X={start[3]} | common counts/voxel scale | no clean GT\n"
            + (
                "Left: observed; right: GAP | HyD2 blue + HyD4 green"
                if overlay
                else "Top: observed; bottom: GAP | HyD2 blue; HyD4 green"
            )
        )
        viewer.reset_view()
        viewer.camera.angles = (25, 30, 45)
        if movie:
            QTimer.singleShot(500, self.play)
        self._write_state()

    def adjust_contrast(self, *_):
        """Apply one physical count scale to every observation and prediction panel."""
        for layer in self.viewer.layers:
            layer.contrast_limits = (0, self.maximum.value())

    def play(self):
        """Animate the selected 4D patch only when raw-movie display remains selected."""
        if self.display.currentIndex() == 0 and self.frames > 1:
            self.viewer.window._qt_viewer.dims.play(
                axis=0, fps=3, loop_mode="loop", frame_range=(0, self.frames - 1)
            )

    def _write_state(self):
        state = {
            "pid": os.getpid(),
            "phase": self.phase.currentText(),
            "ndisplay": self.viewer.dims.ndisplay,
            "layer_names": [layer.name for layer in self.viewer.layers],
            "colormaps": [layer.colormap.name for layer in self.viewer.layers],
            "blending": [layer.blending for layer in self.viewer.layers],
            "visible": [layer.visible for layer in self.viewer.layers],
            "grid_shape": list(self.viewer.grid.shape),
            "grid_stride": self.viewer.grid.stride,
            "channel_layout": (
                self.layout.currentText() if self.layout is not None else "Overlay channels"
            ),
            "mode": self.display.currentText(),
            "clean_ground_truth": False,
            "scope": self.metadata.get(
                "scope", "Scored example patches, not the whole acquisition"
            ),
        }
        (self.root / "napari_prediction_state.json").write_text(json.dumps(state, indent=2))


def main():
    """Open the completed trial's co-registered raw/predicted 3D/4D panels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=0, help="Initial scored-patch selector index")
    parser.add_argument("--study", type=Path, default=Path("data/confocal_microtubule3_joint4d_v1"))
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path("data/confocal_all_channels_survey_v2/lif_03_01_all_channels.json"),
    )
    args = parser.parse_args()
    result = json.loads((args.study / "training/result.json").read_text())
    audit = json.loads((args.study / "comparison.json").read_text())
    if audit["status"] != "passed":
        message = "Replay the completed checkpoint successfully before displaying its predictions"
        raise ValueError(message)
    geometry = json.loads(args.geometry.read_text())
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import napari
    from napari.utils import resize_dask_cache
    from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QWidget

    viewer = napari.Viewer(ndisplay=3)
    resize_dask_cache(nbytes=2**30)
    phase, display, maximum, layout = QComboBox(), QComboBox(), QDoubleSpinBox(), QComboBox()
    layout.addItems(["Overlay channels", "Separate channels"])
    index_path = args.study / "examples/index.json"
    if index_path.exists():
        for record in json.loads(index_path.read_text())["cases"]:
            t, z, y, x = record["start_tzyx"]
            label = f"{record['phase']} | T={t}, Z={z}, Y={y}, X={x} (crop)"
            phase.addItem(label, record["directory"])
    else:
        phase.addItems(list(result["identity"]["config"]["evaluation"]["phases"]))
    if not 0 <= args.case < phase.count():
        parser.error("The initial case lies outside the scored-patch selector")
    phase.setCurrentIndex(args.case)
    display.addItems(["Raw movie", "Temporal mean (noisy context)"])
    maximum.setDecimals(3)
    maximum.setRange(0.001, 255)
    maximum.setSingleStep(0.05)
    maximum.setValue(0.3)
    panel = QWidget()
    form = QFormLayout(panel)
    form.addRow("Scored patch", phase)
    form.addRow("Display", display)
    form.addRow("Channels", layout)
    form.addRow("Max counts/voxel", maximum)
    viewer.window.add_dock_widget(panel, name="Observed / joint GAP4D", area="right")
    viewer.text_overlay.visible = True
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    browser = PredictionBrowser(
        viewer, args.study, result, geometry, phase, display, maximum, layout
    )
    phase.currentIndexChanged.connect(browser.refresh)
    display.currentIndexChanged.connect(browser.refresh)
    layout.currentIndexChanged.connect(browser.refresh)
    maximum.valueChanged.connect(browser.adjust_contrast)
    browser.refresh()
    napari.run()


if __name__ == "__main__":
    main()
