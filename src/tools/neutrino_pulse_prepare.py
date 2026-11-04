"""Audit and materialize real marked pulses while preserving charge and source clocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from ..ssunet.datasets.neutrino import shard_sha256, write_photon_shard
from .neutrino_prepare import write_json


def decode_decimal_floats(raw):
    """Decode the documented ODF decimal mantissa rather than IEEE doubles."""
    words = np.frombuffer(raw, dtype=">u8")
    sign = 1.0 - 2.0 * (words >> 63).astype(float)
    mantissa = (words & ((1 << 52) - 1)).astype(float) / 1e15
    exponent = ((words >> 52) & 2047).astype(np.int32) - 1023
    return sign * np.ldexp(mantissa, exponent)


def read_geometry(path):
    """Read the native string/DOM identities and unmodified calibration positions."""
    raw = Path(path).read_bytes()
    if len(raw) % 32:
        message = "The ODF geometry ends in a partial record"
        raise ValueError(message)
    identities, positions = [], []
    for first in range(0, len(raw), 32):
        record = raw[first : first + 32]
        identities.append([int.from_bytes(record[:4], "big"), int.from_bytes(record[4:8], "big")])
        positions.append(decode_decimal_floats(record[8:]))
    return np.asarray(identities), np.asarray(positions)


def match_geometry(coordinates, reference, maximum_distance, *, spatial_cells=False):
    """Check DOM correspondence or explicitly quantize positions into fixed cells."""
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    matched, distances = [], []
    for first in range(0, len(unique), 64):
        squared = ((unique[first : first + 64, None] - reference[None]) ** 2).sum(2)
        nearest = np.argmin(squared, axis=1)
        matched.extend(nearest.tolist())
        distances.extend(np.sqrt(squared[np.arange(len(nearest)), nearest]).tolist())
    matched, distances = np.asarray(matched), np.asarray(distances)
    if (distances > maximum_distance).any() or (
        not spatial_cells and len(np.unique(matched)) != len(matched)
    ):
        message = "Source pulse positions do not have a unique bounded geometry correspondence"
        raise ValueError(message)
    run_geometry = reference.copy()
    if not spatial_cells:
        run_geometry[matched] = unique
    return matched[inverse], run_geometry, distances


def overlapping_event_groups(starts, durations_us, run_id):
    """Keep overlapping reported acquisition intervals in one descriptive block."""
    order = np.argsort(starts, kind="stable")
    groups = np.empty(len(order), dtype=np.int64)
    last_end, family = -1, -1
    for index in order:
        start = int(starts[index])
        if start > last_end:
            family = run_id * 1_000_000 + int(index)
        last_end = max(last_end, start + round(float(durations_us[index]) * 10000))
        groups[index] = family
    return groups


def _prepare_fold(config, name, label, geometry_ids, reference):
    """Preserve one complete real run with pulse marks and no count-based filtering."""
    source = Path(config["data"]["source"]) / config["data"]["folds"][name]
    output = Path(config["data"]["output"]) / f"{name}.h5"
    partial = output.with_suffix(".partial.h5")
    with h5py.File(source, "r") as handle:
        offsets = handle["event_offsets"][:]
        charge, times = handle["charge_pe"][:], handle["time_ns"][:]
        coordinates = np.column_stack([handle[key][:] for key in ("x_m", "y_m", "z_m")])
        runs, years = handle["run_id"][:], handle["year"][:]
        starts, duration = handle["event_start_time_tenths_ns"][:], handle["event_duration_us"][:]
    if not np.isfinite(charge).all() or np.any(charge < 0) or len(np.unique(runs)) != 1:
        message = "A real-data fold requires finite nonnegative charge and exactly one run"
        raise ValueError(message)
    run = int(runs[0])
    mapped, positions, distances = match_geometry(
        coordinates,
        reference,
        config["data"]["maximum_spatial_quantization_distance_m"],
        spatial_cells=True,
    )
    marks = np.searchsorted(config["data"]["charge_interior_edges_pe"], charge, side="right")
    number_of_marks = len(config["data"]["charge_interior_edges_pe"]) + 1
    virtual_sensors = mapped * number_of_marks + marks
    event_ids = run * 1_000_000 + np.arange(len(offsets) - 1, dtype=np.int64)
    groups = overlapping_event_groups(starts, duration, run)
    provenance = {
        "kind": "empirical_real_marked_pulses",
        "count_object": "one_reconstructed_pulse_with_its_unchanged_charge_mark",
        "unit_photon_detections": False,
        "charge_is_photon_count": False,
        "charge_interior_edges_pe": config["data"]["charge_interior_edges_pe"],
        "clock": "unmodified source time; fixed predeclared analysis crop; no A+B alignment",
        "live_mask_semantics": "candidate_physical_geometry_only_unknown_actual_live_status",
        "spatial_representation": "fixed_geometry_anchor_Voronoi_cells_not_exact_DOM_identity",
        "source_positions": "unchanged source coordinates retained alongside cell indices",
        "spatial_quantization_maximum_distance_m": config["data"][
            "maximum_spatial_quantization_distance_m"
        ],
        "source_hdf5": str(source),
        "source_hdf5_sha256": shard_sha256(source),
        "source_page": "https://icecube.umd.edu/PublicData/",
        "independent_poisson_claimed": False,
        "physical_no_interaction_labels": False,
        "contract": config["contract"],
    }
    write_photon_shard(
        partial,
        offsets=offsets.astype(np.int64),
        sensor_index=virtual_sensors.astype(np.int32),
        time_ns=times,
        positions_m=np.repeat(positions, number_of_marks, axis=0),
        live=np.ones(len(positions) * number_of_marks, dtype=bool),
        event_ids=event_ids,
        configuration_ids=groups,
        partition=np.full(len(event_ids), label, dtype=np.uint8),
        time_edges_ns=np.linspace(
            *config["data"]["time_window_ns"], config["data"]["time_bins"] + 1
        ),
        provenance=provenance,
    )
    with h5py.File(partial, "a") as handle:
        handle.create_dataset("pulses/charge_pe", data=charge, compression="gzip")
        handle.create_dataset("pulses/spatial_cell_index", data=mapped, compression="gzip")
        handle.create_dataset("pulses/positions_m", data=coordinates, compression="gzip")
        handle.create_dataset("geometry/native_string_dom", data=geometry_ids, compression="gzip")
        for key, values in {
            "run_id": runs,
            "year": years,
            "event_start_time_tenths_ns": starts,
            "reported_duration_us": duration,
        }.items():
            handle.create_dataset(f"events/{key}", data=values, compression="gzip")
        handle.attrs["construction_complete"] = True
        handle.attrs["legacy_ragged_group_photons_contains_pulses"] = True
    partial.replace(output)
    with h5py.File(output, "r") as saved:
        np.testing.assert_array_equal(saved["pulses/charge_pe"][:], charge)
        np.testing.assert_array_equal(saved["pulses/positions_m"][:], coordinates)
    low, high = config["data"]["time_window_ns"]
    window = (times >= low) & (times < high)
    in_window = np.array(
        [np.count_nonzero(window[offsets[i] : offsets[i + 1]]) for i in range(len(event_ids))]
    )
    outside_reported = 0
    for index in range(len(event_ids)):
        item = times[offsets[index] : offsets[index + 1]]
        outside_reported += bool(
            len(item) and (item.min() < 0 or item.max() >= duration[index] * 1000)
        )
    original = source.with_suffix(".odf.gz")
    return {
        "path": str(output),
        "sha256": shard_sha256(output),
        "source_hdf5_sha256": provenance["source_hdf5_sha256"],
        "source_odf_sha256": shard_sha256(original),
        "run_id": run,
        "year": int(years[0]),
        "events": len(event_ids),
        "reported_pulses": len(times),
        "pulses_in_analysis_window": int(window.sum()),
        "pulses_outside_analysis_window": int((~window).sum()),
        "empty_source_records": int((np.diff(offsets) == 0).sum()),
        "empty_analysis_windows": int((in_window == 0).sum()),
        "observations_with_1_to_30_pulses_in_window": int(
            ((in_window >= 1) & (in_window <= 30)).sum()
        ),
        "records_with_pulses_outside_reported_duration": outside_reported,
        "spatial_quantization_distance_m_quantiles": np.quantile(
            distances, [0, 0.5, 0.99, 1]
        ).tolist(),
        "noninteger_charge_fraction": float(np.mean(np.abs(charge - np.rint(charge)) > 1e-6)),
        "charge_values_preserved_exactly": True,
        "source_positions_preserved_exactly": True,
        "reported_positions_sharing_one_spatial_cell": int(
            len(np.unique(coordinates, axis=0)) - len(np.unique(mapped))
        ),
        "known_physical_no_event_labels": 0,
    }


def prepare_pulse_study(config):
    """Create an explicitly qualified run-separated empirical pulse experiment."""
    output = Path(config["data"]["output"])
    output.mkdir(exist_ok=True, parents=True)
    if any((output / f"{name}.h5").exists() for name in config["data"]["folds"]):
        message = "Real pulse observation artifacts already exist; preserve their source identities"
        raise FileExistsError(message)
    geometry_path = Path(config["data"]["source"]) / config["data"]["geometry"]
    identities, geometry = read_geometry(geometry_path)
    reports = {
        name: _prepare_fold(config, name, label, identities, geometry)
        for label, name in enumerate(("train", "validation", "test"))
    }
    if len({value["run_id"] for value in reports.values()}) != 3:
        message = "Real-data train/validation/test run identities overlap"
        raise ValueError(message)
    write_json(
        output / "observation_audit.json",
        {
            "configuration": config,
            "geometry_sha256": shard_sha256(geometry_path),
            "fixed_spatial_cells": len(geometry),
            "charge_categories": len(config["data"]["charge_interior_edges_pe"]) + 1,
            "folds": reports,
            "run_overlap": 0,
            "clock_qualification": (
                "reported pulse times may exceed reported event duration; "
                "analysis uses fixed 0-64 us source clock without recentering"
            ),
            "live_status_and_daq_exposure": "not_provided; geometry mask is candidate support only",
            "zero_bias_or_no_event_sample": False,
            "source_selection": "triggered_and_filtered; run-dependent selections differ",
            "exact_cross_run_DOM_correspondence_claimed": False,
            "earlier_import": (
                "pulse_study strict native-DOM mapping stopped; 2013 positions drift up "
                "to 23.4 m with ambiguous nearest identities; preserved without training"
            ),
            "interpretation": (
                "empirical_marked_pulse_prediction_only_no_physical_photon_GT_or_Poisson_claim"
            ),
        },
    )
    return reports


def main():
    """Audit public real data and preserve all pulse records in run-separated folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_pulses.yml"))
    args = parser.parse_args()
    prepare_pulse_study(yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    main()
