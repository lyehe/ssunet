"""Play continuous two-channel motion clips from both recordings with acquisition timing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .confocal_joint_napari import PredictionBrowser


class MotionBrowser(PredictionBrowser):
    """Preserve each recording's physical spacing and timestamps when changing clips."""

    def __init__(self, *args, fps, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = fps

    def refresh(self, *_):
        """Select the correct recording geometry before loading a continuous clip."""
        folder = self.root / "examples" / self.phase.currentData()
        self.geometry = json.loads((folder / "metadata.json").read_text())
        super().refresh()
        self.update_time()

    def play(self, *_):
        """Play at the requested display speed, keeping true acquisition time visible."""
        if self.display.currentIndex() == 0 and self.frames > 1:
            self.viewer.window._qt_viewer.dims.play(
                axis=0, fps=self.fps.value(), loop_mode="loop", frame_range=(0, self.frames - 1)
            )

    def update_time(self, *_):
        """Label temporal means explicitly and distinguish playback from recording time."""
        if not self.viewer.layers or not self.metadata:
            return
        first = self.source_start()[0]
        interval = self.metadata["volume_interval_seconds"]
        if self.display.currentIndex() == 0:
            time = self.viewer.dims.current_step[0]
            timing = f"Source T={first + time} | +{time * interval:.1f} s"
        else:
            timing = f"Noisy temporal context: mean of T={first}..{first + self.frames - 1}"
        self.viewer.text_overlay.text = (
            f"{timing} | acquired {interval:.3f} s/volume\n"
            "Left: observations; right: two-channel GAP | HyD2 blue; HyD4 green\n"
            "Looped exploratory motion review; no validated individual filament tracks"
        )


def main():
    """Open all four continuous ROI clips with additive detector overlays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("data/confocal_twochannel_motion_v1"))
    parser.add_argument("--clip", type=int, default=0)
    args = parser.parse_args()
    index = json.loads((args.study / "index.json").read_text())
    if index["status"] != "complete" or not 0 <= args.clip < len(index["clips"]):
        parser.error("Select an available completed motion clip")
    result = json.loads((Path(index["model_study"]) / "training/result.json").read_text())
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import napari
    from napari.utils import resize_dask_cache
    from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QWidget

    viewer = napari.Viewer(ndisplay=3)
    resize_dask_cache(nbytes=2**30)
    clip, display, layout = QComboBox(), QComboBox(), QComboBox()
    maximum, fps = QDoubleSpinBox(), QDoubleSpinBox()
    for entry in index["clips"]:
        clip.addItem(f"{entry['recording_name']} | {entry['phase']}", entry["directory"])
    clip.setCurrentIndex(args.clip)
    display.addItems(["Raw movie", "Temporal mean (noisy context)"])
    layout.addItems(["Overlay channels", "Separate channels"])
    maximum.setDecimals(3)
    maximum.setRange(0.001, 255)
    maximum.setValue(0.3)
    maximum.setSingleStep(0.05)
    fps.setRange(0.1, 30)
    fps.setValue(5)
    panel = QWidget()
    form = QFormLayout(panel)
    for label, widget in (
        ("Recording / region", clip),
        ("Display", display),
        ("Channels", layout),
        ("Max counts/voxel", maximum),
        ("Playback fps", fps),
    ):
        form.addRow(label, widget)
    viewer.window.add_dock_widget(panel, name="Two-channel motion review", area="right")
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.text_overlay.visible = True
    browser = MotionBrowser(viewer, args.study, result, {}, clip, display, maximum, layout, fps=fps)
    for selector in (clip, display, layout):
        selector.currentIndexChanged.connect(browser.refresh)
    maximum.valueChanged.connect(browser.adjust_contrast)
    fps.valueChanged.connect(browser.play)
    viewer.dims.events.current_step.connect(browser.update_time)
    browser.refresh()
    napari.run()


if __name__ == "__main__":
    main()
