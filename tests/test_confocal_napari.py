"""Check lazy physical-axis ordering and playback growth without opening a GUI."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.tools import confocal_napari
from src.tools.confocal_napari import LiveSeries


@pytest.mark.parametrize(("selected", "updates"), [(None, True), ((0, 3), True), ((1, 2), False)])
def test_live_series_expands_default_animation_but_preserves_custom_range(
    tmp_path, selected, updates
):
    """New frames join default playback; explicitly selected subranges stay selected."""
    values = np.arange(8 * 3 * 4 * 5, dtype=np.uint8).reshape(8, 3, 4, 5)
    np.save(tmp_path / "counts.npy", values)
    np.save(tmp_path / "expected_counts.npy", values.astype(np.float32))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "shape_tzyx": list(values.shape),
                "voxel_spacing_zyx_um": [1, 1, 1],
                "reverse_z_for_display": True,
            }
        )
    )
    (tmp_path / "progress.json").write_text(
        json.dumps({"completed_frames": 8, "status": "running"})
    )
    calls = []
    qt = SimpleNamespace(
        is_playing=True,
        slider_widgets=[SimpleNamespace(frame_range=selected)],
        play=lambda **kwargs: calls.append(kwargs),
    )
    viewer = SimpleNamespace(
        dims=SimpleNamespace(current_step=(1, 1, 2, 2), set_current_step=lambda *args: None),
        window=SimpleNamespace(_qt_viewer=SimpleNamespace(dims=qt)),
        grid=SimpleNamespace(),
    )
    live = LiveSeries(viewer, tmp_path)
    live.completed = 4
    live.layers = [SimpleNamespace(data=None), SimpleNamespace(data=None)]
    live.refresh()
    assert calls == ([{"axis": 0, "frame_range": (0, 7)}] if updates else [])
    assert live.completed == 8
    assert live.layers[0].data.chunks[0] == (1,) * 8
    np.testing.assert_array_equal(live.layers[0].data[2].compute(), values[2, ::-1])


def test_live_series_waits_for_a_complete_output_chunk(tmp_path):
    """An empty live directory is safe to open while inference is starting."""
    live = LiveSeries(SimpleNamespace(), tmp_path)
    live.refresh()
    assert not live.layers


def test_lazy_viewer_reads_only_the_requested_time_point(tmp_path, monkeypatch):
    """Building the graph must not copy the acquisition or read unrequested frames."""
    values = np.arange(8 * 3 * 4 * 5, dtype=np.uint8).reshape(8, 3, 4, 5)
    np.save(tmp_path / "counts.npy", values)
    read = confocal_napari._read_npy_frame
    calls = []

    def record(path, index, reverse_z):
        calls.append(index)
        return read(path, index, reverse_z)

    monkeypatch.setattr(confocal_napari, "_read_npy_frame", record)
    lazy = LiveSeries(None, tmp_path)._data("counts.npy", 8, True)
    assert calls == []
    np.testing.assert_array_equal(lazy[5].compute(), values[5, ::-1])
    assert calls == [5]
