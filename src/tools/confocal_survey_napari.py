"""Browse every channel and early/middle/late clips from the 4D data survey."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ..ssunet.confocal_display import configure_detector_layout, detector_color


class SurveyBrowser:
    """Keep co-registered detectors visible while switching datasets and time ranges."""

    def __init__(self, viewer, root, records, dataset, phase, display, layout=None):
        self.viewer, self.root, self.records = viewer, root, records
        self.dataset, self.phase, self.display = dataset, phase, display
        self.layout = layout
        self.counts = None
        self.metadata = None

    def refresh(self, *_):
        """Load all physical channels from a bounded real-data clip."""
        from qtpy.QtCore import QTimer

        viewer = self.viewer
        viewer.window._qt_viewer.dims.stop()
        record = self.records[self.dataset.currentIndex()]
        phase = self.phase.currentText()
        filename = (
            f"{record['id']}_all_channels.npy"
            if phase == "early"
            else f"{record['id']}_{phase}.npy"
        )
        if viewer.layers:
            viewer.window._qt_viewer.canvas._scene_canvas.set_current()
        viewer.layers.clear()
        self.counts = np.load(self.root / filename, mmap_mode="r", allow_pickle=False)
        self.metadata = json.loads((self.root / f"{record['id']}_all_channels.json").read_text())
        if self.counts.shape[0] != record["channels"]:
            message = "The preview does not contain every detector channel"
            raise ValueError(message)
        movie = self.display.currentIndex() == 0
        spacing = self.metadata["voxel_spacing_zyx_um"]
        for channel in range(record["channels"]):
            values = self.counts[channel]
            if self.metadata["reverse_z_for_display"]:
                values = values[:, ::-1]
            if not movie:
                values = values.mean(axis=0, dtype=np.float32)
            label = record["active_detectors"][channel]["Name"]
            viewer.add_image(
                values,
                name=f"{label} | " + ("raw counts" if movie else "time mean, noisy"),
                scale=((1.0, *spacing) if movie else tuple(spacing)),
                contrast_limits=(0, 2 if movie else 0.2),
                colormap=detector_color(label),
                blending="additive",
                rendering="mip",
                interpolation3d="linear",
            )
        overlay = self.layout is None or self.layout.currentIndex() == 0
        configure_detector_layout(viewer, record["channels"], overlay=overlay)
        viewer.dims.axis_labels = (
            ("time point", "Z (um)", "Y (um)", "X (um)")
            if movie
            else ("Z (um)", "Y (um)", "X (um)")
        )
        start = self.metadata["phase_time_starts"][phase]
        end = start + self.counts.shape[1] - 1
        name = Path(record["file"]).name
        viewer.title = f"4D survey | {name} | {record['channels']} channels | T {start}:{end}"
        viewer.text_overlay.text = (
            f"{phase}: source T={start}..{end}; {self.counts.shape[2]} Z planes; noisy observations"
        )
        viewer.reset_view()
        viewer.camera.angles = (25, 30, 45)
        if movie and self.counts.shape[1] > 1:
            QTimer.singleShot(500, self.start_playback)
        state = {
            "pid": os.getpid(),
            "dataset_id": record["id"],
            "phase": phase,
            "channels": record["channels"],
            "channel_names": [item["Name"] for item in record["active_detectors"]],
            "colormaps": [layer.colormap.name for layer in viewer.layers],
            "blending": [layer.blending for layer in viewer.layers],
            "visible": [layer.visible for layer in viewer.layers],
            "grid_enabled": viewer.grid.enabled,
            "channel_layout": (
                self.layout.currentText() if self.layout is not None else "Overlay channels"
            ),
            "shape_ctzyx": list(self.counts.shape),
            "source_time_start": start,
            "mode": self.display.currentText(),
        }
        (self.root / "napari_browser_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    def start_playback(self):
        """Start the current movie only if the user still selected movie display."""
        if self.display.currentIndex() == 0:
            self.viewer.window._qt_viewer.dims.play(
                axis=0, fps=3, loop_mode="loop", frame_range=(0, self.counts.shape[1] - 1)
            )


def main():
    """Open a dataset/phase picker with every recorded detector shown by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, default=Path("data/confocal_all_channels_survey_v2"))
    parser.add_argument("--dataset", help="Survey dataset ID to display initially")
    parser.add_argument("--phase", choices=("early", "middle", "late"), default="early")
    parser.add_argument("--mean", action="store_true", help="Start with noisy temporal means")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import napari
    from napari.utils import resize_dask_cache
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QComboBox, QFormLayout, QWidget

    records = json.loads((args.survey / "survey.json").read_text())["records"]
    viewer = napari.Viewer(title="4D dataset survey | all detector channels", ndisplay=3)
    resize_dask_cache(nbytes=2**30)
    panel = QWidget()
    layout = QFormLayout(panel)
    dataset, phase, display, channel_layout = QComboBox(), QComboBox(), QComboBox(), QComboBox()
    for record in records:
        dataset.addItem(f"{Path(record['file']).name} | {record['channels']} channel(s)")
    if args.dataset is not None:
        matches = [i for i, record in enumerate(records) if record["id"] == args.dataset]
        if not matches:
            parser.error(f"Unknown survey dataset: {args.dataset}")
        dataset.setCurrentIndex(matches[0])
    phase.addItems(["early", "middle", "late"])
    phase.setCurrentText(args.phase)
    display.addItems(["Raw movie", "Temporal mean (noisy context)"])
    display.setCurrentIndex(int(args.mean))
    channel_layout.addItems(["Overlay channels", "Separate channels"])
    layout.addRow("Acquisition", dataset)
    layout.addRow("Time range", phase)
    layout.addRow("Display", display)
    layout.addRow("Channels", channel_layout)
    viewer.window.add_dock_widget(panel, name="4D acquisitions", area="right")
    viewer.text_overlay.visible = True
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    browser = SurveyBrowser(viewer, args.survey, records, dataset, phase, display, channel_layout)
    dataset.currentIndexChanged.connect(browser.refresh)
    phase.currentIndexChanged.connect(browser.refresh)
    display.currentIndexChanged.connect(browser.refresh)
    channel_layout.currentIndexChanged.connect(browser.refresh)
    browser.refresh()
    QTimer.singleShot(
        1500,
        lambda: viewer.screenshot(path=args.survey / "napari_survey_preview.png", canvas_only=True),
    )
    napari.run()


if __name__ == "__main__":
    main()
