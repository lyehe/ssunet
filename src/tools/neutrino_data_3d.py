"""Export observed detector data in 3D without fitting or evaluating any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from plotly.offline import get_plotlyjs

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from ..ssunet.datasets.neutrino import shard_sha256
from .neutrino_pulse_prepare import read_geometry

ROOT = Path("data/neutrino")
CATEGORIES = (
    (0, 0, "Empty record"),
    (1, 5, "1-5 detections"),
    (6, 30, "6-30 detections"),
    (31, 150, "31-150 detections"),
    (151, 500, "151-500 detections"),
    (501, 10000000, "More than 500 detections"),
)


def _examples(counts):
    """Select median record indices in count categories, without model scores."""
    result = []
    for low, high, label in CATEGORIES:
        eligible = np.flatnonzero((counts >= low) & (counts <= high))
        if len(eligible):
            result.append((int(eligible[len(eligible) // 2]), label))
    return result


def _event(index, label, positions, sensor, times, charge=None, duration_us=1.0, event_id=None):
    """Preserve all saved detections, original positions, times and fractional charge."""
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        message = "The 3D viewer requires finite source coordinates/timestamps"
        raise ValueError(message)
    if len(sensor) != len(times) or (charge is not None and len(charge) != len(times)):
        message = "Coordinates, timestamps and charge must describe the same detections"
        raise ValueError(message)
    low = float(times.min() / 1000) if len(times) else 0.0
    high = float(times.max() / 1000) if len(times) else max(float(duration_us), 1.0)
    if high <= low:
        high = low + 0.001
    return {
        "record": index,
        "event_id": int(event_id) if event_id is not None else index,
        "category": label,
        "positions": positions.tolist(),
        "sensor": sensor.tolist(),
        "time_us": (times / 1000).tolist(),
        "charge_pe": None if charge is None else charge.tolist(),
        "time_range_us": [low, high],
        "detections": len(times),
    }


def real_icecube():
    """Read original real IC59 pulse coordinates, without spatial quantization."""
    source = ROOT / "icecube_odf/Level1_IC59_data_Run00115150_Part00000000.h5"
    geometry_path = ROOT / "icecube_odf/geometry.odf"
    _identities, geometry = read_geometry(geometry_path)
    events = []
    with h5py.File(source, "r") as handle:
        offsets = handle["event_offsets"][:]
        for index, label in _examples(np.diff(offsets)):
            rows = slice(offsets[index], offsets[index + 1])
            coordinates = np.column_stack([handle[name][rows] for name in ("x_m", "y_m", "z_m")])
            positions, sensor = np.unique(coordinates, axis=0, return_inverse=True)
            events.append(
                _event(
                    index,
                    label,
                    positions,
                    sensor,
                    handle["time_ns"][rows],
                    handle["charge_pe"][rows],
                    float(handle["event_duration_us"][index]),
                )
            )
    return {
        "id": "icecube",
        "name": "Real IceCube IC59",
        "kind": "REAL DETECTOR DATA",
        "unit": "pulses",
        "description": (
            "Run 115150 (2010), triggered detector records. These are calibrated pulses "
            "with fractional charge in photoelectrons (PE). Event type is not assigned."
        ),
        "geometry_note": (
            "Gray points are reference sensor positions. Full detector includes "
            "the surface array; the initial view focuses on recorded hits."
        ),
        "geometry": geometry.tolist(),
        "events": events,
        "default_category": "151-500 detections",
        "source": str(source),
        "source_sha256": shard_sha256(source),
        "geometry_sha256": shard_sha256(geometry_path),
    }


def simulated_pone():
    """Read complete saved P-ONE photon rows and their sensor geometry."""
    source = ROOT / "study/validation.h5"
    events = []
    with h5py.File(source, "r") as handle:
        geometry = handle["geometry/positions_m"][:]
        offsets = handle["photons/event_offsets"][:]
        for index, label in _examples(np.diff(offsets)):
            rows = slice(offsets[index], offsets[index + 1])
            events.append(
                _event(
                    index,
                    label,
                    geometry,
                    handle["photons/sensor_index"][rows],
                    handle["photons/time_ns"][rows],
                    event_id=handle["events/event_id"][index],
                )
            )
    return {
        "id": "pone",
        "name": "Simulated P-ONE photons",
        "kind": "SIMULATION",
        "unit": "photons",
        "description": (
            "P-ONE water-detector simulation: saved unit-photon detections at 60 sensors. "
            "No predictions or inferred particle tracks are displayed."
        ),
        "geometry_note": "Gray points are the simulated detector sensors.",
        "geometry": geometry.tolist(),
        "events": events,
        "default_category": "31-150 detections",
        "source": str(source),
        "source_sha256": shard_sha256(source),
    }


def _aggregate(event):
    sensor = np.asarray(event["sensor"], dtype=int)
    positions = np.asarray(event["positions"], dtype=float).reshape(-1, 3)
    count = np.bincount(sensor, minlength=len(positions))
    time = np.full(len(positions), np.inf)
    np.minimum.at(time, sensor, np.asarray(event["time_us"]))
    return positions[count > 0], count[count > 0], time[count > 0]


def preview(datasets, output):
    """Create a static 3D overview of real and simulated recorded detections."""
    figure = plt.figure(figsize=(13, 6.8))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.13, top=0.85, wspace=0.12)
    for index, data in enumerate(datasets):
        event = next(
            value for value in data["events"] if value["category"] == data["default_category"]
        )
        positions, counts, times = _aggregate(event)
        geometry = np.asarray(data["geometry"])
        axis = figure.add_subplot(1, 2, index + 1, projection="3d")
        frame = geometry if data["id"] == "pone" else positions
        low, high = frame.min(0), frame.max(0)
        pad = max(float((high - low).max()) * 0.12, 40)
        bounds = np.column_stack([low - pad, high + pad])
        visible = ((geometry >= bounds[:, 0]) & (geometry <= bounds[:, 1])).all(1)
        axis.scatter(*geometry[visible].T, s=3, color="#9ca8b8", alpha=0.25, depthshade=False)
        points = axis.scatter(
            *positions.T,
            s=24 + 18 * np.sqrt(counts),
            c=times,
            cmap="turbo",
            vmin=event["time_range_us"][0],
            vmax=event["time_range_us"][1],
            depthshade=False,
            edgecolors="#24354a",
            linewidths=0.25,
        )
        axis.set_xlim(*bounds[0])
        axis.set_ylim(*bounds[1])
        axis.set_zlim(*bounds[2])
        axis.set_box_aspect(bounds[:, 1] - bounds[:, 0])
        axis.view_init(elev=17, azim=135)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        identity = (
            f"record {event['record']}" if data["id"] == "icecube" else f"event {event['event_id']}"
        )
        axis.set_title(
            f"{data['name']} · {identity}\n"
            f"{event['detections']} {data['unit']} at {len(positions)} positions",
            fontsize=10,
        )
        figure.colorbar(
            points, ax=axis, shrink=0.6, pad=0.09, label="First arrival (µs, source time)"
        )
    figure.suptitle("Observed detector data in 3D")
    figure.supxlabel(
        "Gray: reference sensors. Color: first arrival. "
        "Marker size increases with recorded count. No reconstructed tracks.",
        fontsize=9,
    )
    figure.savefig(output / "preview.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_viewer(output, *, refresh=False):
    """Export a self-contained viewer; existing data and studies remain untouched."""
    output.mkdir(parents=True, exist_ok=refresh)
    datasets = [real_icecube(), simulated_pone()]
    payload = {
        "datasets": datasets,
        "colorscale": [
            [float(value), to_hex(plt.get_cmap("turbo")(value))] for value in np.linspace(0, 1, 32)
        ],
    }
    for data in datasets:
        for event in data["events"]:
            _positions, counts, _times = _aggregate(event)
            if int(counts.sum()) != event["detections"]:
                message = "3D aggregation changed the number of recorded detections"
                raise ValueError(message)
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    page = template.replace("__PLOTLY_JS__", get_plotlyjs()).replace(
        "__OBSERVATIONS_JSON__", encoded
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "observations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    preview(datasets, output)
    manifest = {
        "scope": "observation_only_3d_display_no_model_or_track_inference",
        "selection": "median source record index within each recorded-count category",
        "times": (
            "complete saved source events; original timestamps shown in microseconds; "
            "no analysis crop or recentering"
        ),
        "marker": (
            "sensor location aggregated over currently revealed detections; "
            "size grows with count, color is first arrival"
        ),
        "real_charge": "fractional PE preserved, never rounded into photon counts",
        "events": {
            data["id"]: [
                {
                    "record": e["record"],
                    "event_id": e["event_id"],
                    "category": e["category"],
                    "detections": e["detections"],
                }
                for e in data["events"]
            ]
            for data in datasets
        },
        "inputs": {data["source"]: data["source_sha256"] for data in datasets},
        "code_sha256": shard_sha256(Path(__file__)),
        "template_sha256": shard_sha256(Path(__file__).with_suffix(".html")),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output / "index.html"


def main():
    """Create the requested 3D observation viewer once in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data_3d_v1")
    parser.add_argument("--refresh", action="store_true", help="Refresh an existing viewer export")
    args = parser.parse_args()
    print(create_viewer(args.output, refresh=args.refresh))  # noqa: T201


if __name__ == "__main__":
    main()
