"""Survey every detector channel in local confocal and STED 4D acquisitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from defusedxml import ElementTree

LOGGER = logging.getLogger(__name__)


def _statistics(frame):
    values, frequencies = np.unique(frame, return_counts=True)
    histogram = {str(int(k)): int(v) for k, v in zip(values, frequencies, strict=True)}
    present = {int(value) for value in values}
    return {
        "mean": float(frame.mean()),
        "nonzero_fraction": float(np.count_nonzero(frame) / frame.size),
        "maximum": int(frame.max()),
        "histogram": histogram,
        "missing_small_values": [
            i for i in range(min(int(values[-1]), 32) + 1) if i not in present
        ],
        "saturated_pixels": int(np.count_nonzero(frame == np.iinfo(frame.dtype).max)),
    }


def _settings_record(settings, detectors, bit_depths):
    averages = {
        key: settings.get(key)
        for key in ("FrameAverage", "LineAverage", "FrameAccumulation", "Line_Accumulation")
    }
    eligible = (
        all(d.get("AcquisitionModeName") == "PhotonCounting" for d in detectors)
        and all(value == "1" for value in averages.values())
        and all(bit == 8 for bit in bit_depths)
    )
    return {
        "active_detectors": detectors,
        "averaging": averages,
        "stored_bits": list(bit_depths),
        "unaveraged_count_mode_verified": eligible,
    }


def _sample(reader, shape, channels, slug, output, physical):
    """Sample distributed real planes and export time averages plus an all-channel clip."""
    t_size, z_size, _y_size, _x_size = shape
    times = np.unique(np.linspace(0, t_size - 1, min(t_size, 11), dtype=int)).tolist()
    planes = np.unique(np.linspace(0, z_size - 1, min(z_size, 5), dtype=int)).tolist()
    statistics, summaries, previews = [], [], []
    phase_previews = {"early": [], "middle": [], "late": []}
    clip_length = min(t_size, 16)
    phase_starts = {
        "early": 0,
        "middle": max(0, t_size // 2 - clip_length // 2),
        "late": t_size - clip_length,
    }
    for channel in range(channels):
        rows, time_means = [], []
        for t in times:
            for z in planes:
                frame = reader(channel, t, z)
                rows.append({"time": t, "z": z, **_statistics(frame)})
        for t in (0, t_size // 2, t_size - 1):
            begin = min(max(t - 4, 0), max(0, t_size - 8))
            mean = np.mean(
                [
                    reader(channel, k, z_size // 2).astype(np.float32)
                    for k in range(begin, min(t_size, begin + 8))
                ],
                axis=0,
            )
            time_means.append(mean)
        center_z = max(0, z_size // 2 - 8)
        begin_y, begin_x = 0, 0
        cached_clips = {}
        for phase, start in phase_starts.items():
            if start not in cached_clips:
                cached_clips[start] = np.stack(
                    [
                        np.stack(
                            [
                                reader(channel, t, z)
                                for z in range(center_z, min(z_size, center_z + 16))
                            ]
                        )
                        for t in range(start, start + clip_length)
                    ]
                )
            phase_previews[phase].append(cached_clips[start])
        clip = cached_clips[0]
        previews.append(clip)
        sampled_means = [np.mean([r["mean"] for r in rows if r["time"] == t]) for t in times]
        mean = float(np.mean([r["mean"] for r in rows]))
        late_ratio = float(sampled_means[-1] / max(sampled_means[0], 1e-12))
        summaries.append(
            {
                "channel_index": channel,
                "sample_mean_counts": mean,
                "sample_nonzero_fraction": float(np.mean([r["nonzero_fraction"] for r in rows])),
                "sample_maximum": max(r["maximum"] for r in rows),
                "sample_saturated_pixels": sum(r["saturated_pixels"] for r in rows),
                "first_mean": float(sampled_means[0]),
                "last_mean": float(sampled_means[-1]),
                "last_to_first_ratio": late_ratio,
                "sampled_times": times,
                "sampled_mean_by_time": list(map(float, sampled_means)),
            }
        )
        statistics.append({"channel_index": channel, "planes": rows})
        np.savez_compressed(
            output / f"{slug}_ch{channel}_means.npz",
            early=time_means[0],
            middle=time_means[1],
            late=time_means[2],
        )
    counts = np.stack(previews)
    np.save(output / f"{slug}_all_channels.npy", counts, allow_pickle=False)
    metadata = {
        "shape_ctzyx": list(counts.shape),
        "axis_order": "C,T,Z,Y,X",
        "source_time_start": 0,
        "source_z_start": center_z,
        "source_y_start": begin_y,
        "source_x_start": begin_x,
        "phase_time_starts": phase_starts,
        **physical,
        "notes": "Raw detector counts; channels remain separate; not clean ground truth",
    }
    (output / f"{slug}_all_channels.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for phase in ("middle", "late"):
        np.save(output / f"{slug}_{phase}.npy", np.stack(phase_previews[phase]), allow_pickle=False)
    return summaries, statistics


def _lif_records(root, output):
    from readlif.reader import LifFile

    records = []
    for path in sorted(root.glob("*.lif")):
        lif = LifFile(path)
        for index, image in enumerate(lif.get_iter_image()):
            shape = (image.dims.t, image.dims.z, image.dims.y, image.dims.x)
            if min(shape[:2]) < 2:
                continue
            node = next(
                e.find("./Data/Image")
                for e in lif.xml_root.iter("Element")
                if e.get("Name") == image.name and e.find("./Data/Image") is not None
            )
            detectors = [d.attrib for d in node.findall(".//Detector") if d.get("IsActive") == "1"]
            slug = f"lif_{len(records):02d}_{index:02d}"
            physical = {
                "voxel_spacing_zyx_um": [abs(1 / image.scale[i]) for i in (2, 1, 0)],
                "reverse_z_for_display": image.scale[2] < 0,
                "volume_interval_seconds": 1 / image.scale[3],
            }
            LOGGER.info("Sampling all %d channels: %s / %s", image.channels, path.name, image.name)
            with path.open("rb") as handle:
                image.filename = handle  # readlif supports a shared, read-only file handle.
                summaries, stats = _sample(
                    lambda c, t, z, image=image: np.asarray(image.get_frame(c=c, t=t, z=z)),
                    shape,
                    image.channels,
                    slug,
                    output,
                    physical,
                )
            records.append(
                {
                    "id": slug,
                    "format": "lif",
                    "file": str(path),
                    "series_index": index,
                    "series_name": image.name,
                    "shape_tzyx": list(shape),
                    "channels": image.channels,
                    "file_size_bytes": path.stat().st_size,
                    "file_mtime_ns": path.stat().st_mtime_ns,
                    "xml_sha256": hashlib.sha256(lif.xml_header.encode()).hexdigest(),
                    **_settings_record(
                        image.settings,
                        detectors,
                        image.bit_depths if hasattr(image, "bit_depths") else image.bit_depth,
                    ),
                    **physical,
                    "channel_summaries": summaries,
                    "sampled_statistics": stats,
                }
            )
    return records


class _StedReader:
    """Map both XY-plane and XZ-plane TIFF exports to canonical ZYX volumes."""

    def __init__(self, pages, shape, slow_axis):
        self.pages, self.shape, self.slow_axis = pages, shape, slow_axis
        self.cached_time, self.cached_volume = None, None

    def frame(self, channel, time, z):
        """Read a canonical XY slice without reordering time and physical Y."""
        if channel != 0:
            raise IndexError(channel)
        if self.slow_axis == 3:
            return self.pages[time * self.shape[1] + z].asarray()
        if self.slow_axis != 2:
            raise NotImplementedError(self.slow_axis)
        if time != self.cached_time:
            y_size = self.shape[2]
            self.cached_volume = np.stack(
                [self.pages[time * y_size + y].asarray() for y in range(y_size)]
            ).transpose(1, 0, 2)
            self.cached_time = time
        return self.cached_volume[z]


def _sted_records(root, output):
    import tifffile

    records = []
    for path in sorted((root / "sted").glob("*.tif")):
        stem = path.stem.rsplit("_ch", 1)[0]
        if stem.endswith(("_z000", "_y000")):
            stem = stem[:-5]
        xml_path = root / "sted/MetaData" / (stem + ".xml")
        xml = ElementTree.parse(xml_path).getroot()
        dims = {int(d.get("DimID")): d.attrib for d in xml.iter("DimensionDescription")}
        sizes = {key: int(value["NumberOfElements"]) for key, value in dims.items()}
        if sizes.get(3, 1) < 2 or sizes.get(4, 1) < 2:
            continue
        shape = (sizes[4], sizes[3], sizes[2], sizes[1])
        detector_rows = [d.attrib for d in xml.iter("Detector") if d.get("IsActive") == "1"]
        settings = next(iter(xml.iter("ATLConfocalSettingDefinition"))).attrib
        slug = f"sted_{len(records):02d}"
        display_axes = sorted((1, 2, 3), key=lambda axis: int(dims[axis]["BytesInc"]))[:2]
        slow_axis = next(axis for axis in (1, 2, 3) if axis not in display_axes)
        physical = {
            "voxel_spacing_zyx_um": [
                abs(float(dims[i]["Length"]) * 1e6 / (sizes[i] - 1)) for i in (3, 2, 1)
            ],
            "reverse_z_for_display": float(dims[3]["Length"]) < 0,
            "volume_interval_seconds": float(dims[4]["Length"]) / (sizes[4] - 1),
        }
        LOGGER.info("Sampling STED export: %s", path.name)
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != shape[0] * sizes[slow_axis]:
                message = "TIFF page layout does not match XML dimensions"
                raise ValueError(message)
            reader = _StedReader(tif.pages, shape, slow_axis)
            summaries, stats = _sample(reader.frame, shape, 1, slug, output, physical)
            bits = tif.pages[0].bitspersample
        records.append(
            {
                "id": slug,
                "format": "tiff_xml",
                "file": str(path),
                "series_name": stem,
                "metadata_file": str(xml_path),
                "metadata_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
                "shape_tzyx": list(shape),
                "channels": 1,
                **_settings_record(settings, detector_rows, [bits]),
                **physical,
                "channel_summaries": summaries,
                "sampled_statistics": stats,
            }
        )
    return records


def _plot_record_details(output, record, plt):
    fig, axes = plt.subplots(
        record["channels"],
        3,
        figsize=(11, 3.6 * record["channels"]),
        layout="constrained",
        squeeze=False,
    )
    fields = []
    for channel in range(record["channels"]):
        with np.load(output / f"{record['id']}_ch{channel}_means.npz") as saved:
            fields.append([saved[key] for key in ("early", "middle", "late")])
    maximum = max(float(np.quantile(np.asarray(fields), 0.997)), 0.05)
    for channel in range(record["channels"]):
        for column, label in enumerate(("Early", "Middle", "Late")):
            axis = axes[channel, column]
            artist = axis.imshow(fields[channel][column], cmap="magma", vmin=0, vmax=maximum)
            axis.set_title(f"{record['active_detectors'][channel]['Name']} | {label}")
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.02, label="counts/voxel")
    fig.suptitle(
        Path(record["file"]).name + " | noisy 8-time averages, no ground truth", fontsize=11
    )
    fig.savefig(output / f"{record['id']}_detail.png", dpi=150)
    plt.close(fig)


def _plots(output, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(records), 2, figsize=(11, 3.8 * len(records)), layout="constrained", squeeze=False
    )
    for row, record in enumerate(records):
        fields = []
        for channel in range(record["channels"]):
            with np.load(output / f"{record['id']}_ch{channel}_means.npz") as data:
                fields.append(data["early"])
        maximum = max(float(np.quantile(np.stack(fields), 0.997)), 0.05)
        for channel in range(2):
            axis = axes[row, channel]
            if channel >= len(fields):
                axis.axis("off")
                continue
            artist = axis.imshow(fields[channel], cmap="magma", vmin=0, vmax=maximum)
            label = record["active_detectors"][channel].get("Name", f"channel {channel}")
            axis.set_title(
                f"{Path(record['file']).name}\n{record['series_name']} | {label} | 8-time mean",
                fontsize=9,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.02, label="counts/voxel")
    fig.suptitle("Every available detector channel | noisy observation averages, no ground truth")
    fig.savefig(output / "all_channels_overview.png", dpi=130)
    plt.close(fig)
    for record in records:
        _plot_record_details(output, record, plt)
    fig, axes = plt.subplots(
        len(records), 1, figsize=(10, 2.2 * len(records)), layout="constrained", squeeze=False
    )
    for axis, record in zip(axes[:, 0], records, strict=True):
        for channel in record["channel_summaries"]:
            axis.plot(
                channel["sampled_times"],
                channel["sampled_mean_by_time"],
                "o-",
                label=f"channel {channel['channel_index']}",
            )
        axis.set_title(Path(record["file"]).name, fontsize=9)
        axis.set_ylabel("Sample counts/voxel")
        axis.legend()
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Recorded time point")
    fig.savefig(output / "all_channels_time_profiles.png", dpi=140)
    plt.close(fig)


def main():
    """Read bounded samples from each real 4D acquisition and every detector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"\\jnas2\Yehe\Confocal data"))
    parser.add_argument("--output", type=Path, default=Path("data/confocal_all_channels_survey_v2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    records = _lif_records(args.root, args.output) + _sted_records(args.root, args.output)
    document = {
        "source_root": str(args.root),
        "access": "Read-only samples; source files unchanged",
        "scope": "Every channel in genuine XYZ-time acquisitions; bounded samples",
        "records": records,
    }
    (args.output / "survey.json").write_text(
        json.dumps(document, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (args.output / "channels.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "dataset",
            "series",
            "channel",
            "T",
            "Z",
            "mean_counts",
            "nonzero_fraction",
            "max_count",
            "late_early_ratio",
            "count_mode_verified",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            for channel in record["channel_summaries"]:
                writer.writerow(
                    dict(
                        zip(
                            columns,
                            [
                                Path(record["file"]).name,
                                record["series_name"],
                                channel["channel_index"],
                                *record["shape_tzyx"][:2],
                                channel["sample_mean_counts"],
                                channel["sample_nonzero_fraction"],
                                channel["sample_maximum"],
                                channel["last_to_first_ratio"],
                                record["unaveraged_count_mode_verified"],
                            ],
                            strict=True,
                        )
                    )
                )
    _plots(args.output, records)
    LOGGER.info(
        "Checked %d 4D acquisitions / %d detector channels",
        len(records),
        sum(r["channels"] for r in records),
    )


if __name__ == "__main__":
    main()
