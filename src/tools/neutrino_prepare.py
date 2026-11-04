"""Audit the archived source and build persistent, independently partitioned cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

if __package__ == "src.tools":
    from ..ssunet.datasets.neutrino import shard_sha256, validate_photon_arrays
else:
    from ssunet.datasets.neutrino import shard_sha256, validate_photon_arrays


def write_json(path: Path, content) -> None:
    """Save finite structured evidence without changing an input artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_connection(path):
    """Open the source in read-only mode with a bounded SQLite cache."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _evidence_index(output):
    """Fingerprint the pre-existing development results before independent fitting."""
    names = [
        "plans/neutrino_phased_plan.txt",
        "plans/neutrino_longer_pilot.txt",
        "plans/neutrino_study_protocol.txt",
        "config/neutrino_study.yml",
        "data/neutrino/controlled_reference/poisson_audit.json",
        "data/neutrino/controlled_reference/pone_pilot_evaluation.json",
        "data/neutrino/controlled_reference/longer_pilot_v1/summary.json",
        "data/neutrino/controlled_reference/visualizations_v1/manifest.json",
    ]
    write_json(
        output / "evidence_index.json",
        {
            "scope": "exploratory_pilot_and_prospective_protocol",
            "files": [
                {"path": name, "sha256": shard_sha256(name)}
                for name in names
                if Path(name).exists()
            ],
        },
    )


def audit_source(config: dict) -> dict:
    """Group exact stored initial states and inventory source assumptions honestly."""
    begin = time.perf_counter()
    source, output = config["source"], Path(config["data"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    for name, hash_key in (("archive", "archive_sha256"), ("geometry", "geometry_sha256")):
        if shard_sha256(source[name]) != source[hash_key]:
            message = f"Source hash mismatch for {name}"
            raise ValueError(message)
    database_hash = shard_sha256(source["database"])
    with source_connection(source["database"]) as connection:
        schema = connection.execute(
            "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        truth = pd.read_sql_query("SELECT * FROM mc_truth ORDER BY event_no", connection)
        photon_columns = [row[1] for row in connection.execute("PRAGMA table_info(photons)")]
        expected = {
            "sensor_pos_x",
            "sensor_pos_y",
            "sensor_pos_z",
            "string_id",
            "sensor_id",
            "t",
            "event_no",
        }
        if set(photon_columns) != expected:
            message = "Source photon schema contains unverified extra/weight/response fields"
            raise ValueError(message)
        if not np.isfinite(truth.to_numpy()).all() or truth.event_no.duplicated().any():
            message = "Source truth catalogue has nonfinite values or duplicate event IDs"
            raise ValueError(message)
        photon_stats = pd.read_sql_query(
            "SELECT event_no,COUNT(*) AS photon_rows,MIN(t) AS time_min_ns,MAX(t) AS time_max_ns "
            "FROM photons GROUP BY event_no",
            connection,
        )
        source_sensors = pd.read_sql_query(
            "SELECT DISTINCT string_id,sensor_id,sensor_pos_x,sensor_pos_y,sensor_pos_z "
            "FROM photons",
            connection,
        )
    configuration_columns = [column for column in truth if column != "event_no"]
    group_ids = truth.groupby(configuration_columns, sort=False, dropna=False).event_no.transform(
        "min"
    )
    index = pd.DataFrame({"event_no": truth.event_no, "group_id": group_ids.astype(np.int64)})
    index = index.merge(photon_stats, on="event_no", how="left", validate="one_to_one")
    index["photon_rows"] = index.photon_rows.fillna(0).astype(np.int64)
    index["grouping_basis"] = "exact_stored_initial_state;minimum_source_event_id"
    index.to_parquet(output / "source_group_index.parquet", index=False)
    truth.to_parquet(output / "source_truth.evaluation_only.parquet", index=False)
    source_sensors.to_parquet(output / "source_sensor_mapping.parquet", index=False)
    orphan = np.setdiff1d(photon_stats.event_no, truth.event_no)
    if len(orphan):
        message = "Photon rows refer to events absent from the source catalogue"
        raise ValueError(message)
    groups, sizes = np.unique(index.group_id, return_counts=True)
    result = {
        "source_database": source["database"],
        "source_database_sha256": database_hash,
        "source_archive_sha256": source["archive_sha256"],
        "schema": schema,
        "source_events": len(truth),
        "stored_configuration_groups": len(groups),
        "groups_with_repeated_initial_state": int(np.count_nonzero(sizes > 1)),
        "events_in_repeated_initial_states": int(sizes[sizes > 1].sum()),
        "photons": int(index.photon_rows.sum()),
        "empty_catalogue_events": int((index.photon_rows == 0).sum()),
        "photon_rows_per_event_quantiles": np.quantile(
            index.photon_rows, [0, 0.25, 0.5, 0.75, 0.99, 1]
        ).tolist(),
        "global_source_time_range_reported_ns": [
            float(photon_stats.time_min_ns.min()),
            float(photon_stats.time_max_ns.max()),
        ],
        "source_sensor_rows": len(source_sensors),
        "energy_range_source_units": [
            float(truth.initial_state_energy.min()),
            float(truth.initial_state_energy.max()),
        ],
        "particle_type_counts": {
            str(key): int(value) for key, value in truth.initial_state_type.value_counts().items()
        },
        "grouping_basis": configuration_columns,
        "all_latent_relatives_certified": False,
        "interpretation": "qualified_curated_predictive_photon_thinning",
        "unit_photon_rows": (
            "supported_by_upstream_photon_schema; archived_generator_configuration_not_supplied"
        ),
        "time_unit": "ns_in_upstream_Hit_documentation; archive_generation_version_not_supplied",
        "clock": "original_t_preserved; upstream_alignment_not_certified",
        "upstream_selection": "merged_cut_archive; full_selection_not_documented_in_SQLite",
        "conditional_poisson_certified": False,
        "detector_response": "no_charge_waveform_deadtime_or_weight_fields_in_photon_table",
        "data_license": "archive_license_not_embedded; software_license_is_not_archive_license",
        "unresolved": [
            "archived_generator_version/configuration",
            "complete_upstream_selection",
            "upstream_clock_alignment",
            "unit_weight/efficiency_configuration",
            "latent_relations_not_represented_by_initial_state",
            "explicit_archive_data_license",
        ],
        "primary_sources": [
            "https://graphnet-team.github.io/graphnet/api/graphnet.datasets.prometheus_datasets.html",
            "https://harvard-neutrino.github.io/prometheus/api-reference/prometheus/photon_propagation/hit/",
            "https://github.com/Harvard-Neutrino/prometheus",
        ],
        "elapsed_seconds": time.perf_counter() - begin,
    }
    write_json(output / "source_audit.json", result)
    (output / "source_audit.txt").write_text(
        "Archived P-ONE source audit\n\n"
        f"Events: {result['source_events']}; exact stored groups: {len(groups)}.\n"
        f"Photon rows: {result['photons']}; catalogue events without detections: "
        f"{result['empty_catalogue_events']}.\n"
        "Exact initial-state repeats are grouped before splitting. "
        "No stronger latent-family guarantee is inferred.\n"
        "Rows have photon identity/position/time and no fractional charge/ADC/weight fields.\n"
        "Current upstream Hit documentation uses ns; "
        "original t is preserved without event alignment.\n"
        "Archived generation version, selection, clock alignment "
        "and efficiency configuration remain qualified.\n"
        "This supports a curated predictive-thinning study, "
        "not certified physical Poisson independence.\n"
        "Exact Poisson theory remains tested by the separate analytic generator.\n"
        "Unresolved: " + ", ".join(result["unresolved"]) + ".\n",
        encoding="utf-8",
    )
    _evidence_index(output)
    return result


def stable_group_rank(group, seed, namespace):
    """Return a deterministic source-group rank independent of cohort membership."""
    payload = f"{namespace}:{seed}:{int(group)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def make_cohort(config: dict) -> dict:
    """Persist one source-level assignment and nested, pilot-disjoint cohort views."""
    settings, source = config["data"], config["source"]
    output = Path(settings["output"])
    destination = output / "cohort_manifest.parquet"
    if destination.exists():
        message = f"Refusing to replace an existing cohort: {destination}"
        raise FileExistsError(message)
    index = pd.read_parquet(output / "source_group_index.parquet", columns=["event_no", "group_id"])
    with h5py.File(source["pilot"], "r") as handle:
        pilot = handle["events/event_id"][:]
    pilot_groups = index.loc[index.event_no.isin(pilot), "group_id"].unique()
    eligible = index.loc[~index.group_id.isin(pilot_groups)].copy()
    group_sizes = eligible.groupby("group_id", sort=True).size()
    namespace = source["archive_sha256"]
    ranks = np.asarray(
        [
            stable_group_rank(group, settings["cohort_seed"], namespace)
            for group in group_sizes.index
        ],
        dtype=np.uint64,
    )
    order = np.argsort(ranks, kind="stable")
    cumulative = np.cumsum(group_sizes.to_numpy()[order])
    if cumulative[-1] < settings["cohort_events"]:
        message = "Source has too few eligible events after family-level pilot exclusion"
        raise ValueError(message)
    count = int(np.searchsorted(cumulative, settings["cohort_events"])) + 1
    engineering_count = int(np.searchsorted(cumulative, settings["engineering_events"])) + 1
    chosen_groups = group_sizes.index.to_numpy()[order[:count]]
    engineering_groups = chosen_groups[:engineering_count]
    selected = eligible.loc[eligible.group_id.isin(chosen_groups)].copy()
    partition_by_group = {
        int(group): (
            0
            if (rank := stable_group_rank(group, settings["partition_seed"], namespace) % 10000)
            < 8000
            else 1
            if rank < 9000
            else 2
        )
        for group in chosen_groups
    }
    selected["partition"] = selected.group_id.map(partition_by_group).astype(np.uint8)
    selected["engineering"] = selected.group_id.isin(engineering_groups)
    selected.sort_values("event_no", inplace=True)
    family_overlap = int((selected.groupby("group_id").partition.nunique() > 1).sum())
    if family_overlap or selected.event_no.isin(pilot).any():
        message = "Cohort has pilot or known-family partition leakage"
        raise ValueError(message)
    selected.to_parquet(destination, index=False)
    result = {
        "events": len(selected),
        "groups": len(chosen_groups),
        "engineering_events": int(selected.engineering.sum()),
        "partitions": {
            name: int((selected.partition == label).sum())
            for label, name in enumerate(("train", "validation", "test"))
        },
        "engineering_partitions": {
            name: int(((selected.partition == label) & selected.engineering).sum())
            for label, name in enumerate(("train", "validation", "test"))
        },
        "pilot_events_excluded": len(pilot),
        "pilot_related_groups_excluded": len(pilot_groups),
        "pilot_event_overlap": int(selected.event_no.isin(pilot).sum()),
        "pilot_group_overlap": int(selected.group_id.isin(pilot_groups).sum()),
        "known_family_partition_overlap": family_overlap,
        "cohort_manifest_sha256": shard_sha256(destination),
        "selection_uses_photon_counts": False,
        "group_ranking": "blake2b_64bit_source_namespace_and_cohort_seed",
        "partition_assignment": "blake2b_64bit_source_namespace_partition_seed_mod10000_80_10_10",
        "source_archive_sha256": namespace,
        "configuration": config,
    }
    write_json(output / "cohort_manifest.json", result)
    return result


def _geometry_mapping(config, output):
    """Resolve source local sensor IDs using the audited fixed geometry."""
    geometry = pd.read_parquet(config["source"]["geometry"])
    positions = geometry[["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]].to_numpy()
    string_key = "sensor_string_id" if "sensor_string_id" in geometry else "string_id"
    raw = pd.read_parquet(output / "source_sensor_mapping.parquet")
    mapping = {}
    for row in raw.itertuples(index=False):
        point = np.asarray([row.sensor_pos_x, row.sensor_pos_y, row.sensor_pos_z])
        matches = np.flatnonzero(
            np.isclose(positions, point, rtol=0, atol=1e-5).all(1)
            & (geometry[string_key].to_numpy() == row.string_id)
        )
        if (
            len(matches) != 1
            or not float(row.string_id).is_integer()
            or not float(row.sensor_id).is_integer()
        ):
            message = "Source sensor does not map uniquely to the fixed geometry"
            raise ValueError(message)
        key = (int(row.string_id), int(row.sensor_id))
        if key in mapping and mapping[key] != int(matches[0]):
            message = "One source sensor identity maps to multiple fixed positions"
            raise ValueError(message)
        mapping[key] = int(matches[0])
    return positions, mapping


def _write_fold(connection, selected, destination, positions, mapping, edges, config, provenance):
    """Append bounded photon blocks and atomically publish a completed fold."""
    if destination.exists():
        message = f"Refusing to overwrite observation shard: {destination}"
        raise FileExistsError(message)
    partial = destination.with_suffix(".h5.partial")
    if partial.exists():
        message = f"An unfinished import needs inspection before restarting: {partial}"
        raise FileExistsError(message)
    begin, cursor = time.perf_counter(), 0
    offsets = np.zeros(len(selected) + 1, dtype=np.int64)
    live = np.ones(len(positions), dtype=bool)
    with h5py.File(partial, "x") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        handle.attrs["construction_complete"] = False
        sensors = handle.create_dataset(
            "photons/sensor_index",
            shape=(0,),
            maxshape=(None,),
            dtype="i4",
            chunks=(65536,),
            compression="gzip",
            compression_opts=1,
        )
        times = handle.create_dataset(
            "photons/time_ns",
            shape=(0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(65536,),
            compression="gzip",
            compression_opts=1,
        )
        for first in range(0, len(selected), config["data"]["import_chunk_events"]):
            block = selected.iloc[first : first + config["data"]["import_chunk_events"]]
            identifiers = block.event_no.to_numpy()
            placeholders = ",".join("?" for _ in identifiers)
            # Only placeholder punctuation is constructed; event values are bound below.
            query = (
                "SELECT event_no,string_id,sensor_id,t,sensor_pos_x,sensor_pos_y,sensor_pos_z "  # noqa: S608
                f"FROM photons WHERE event_no IN ({placeholders}) "
                "ORDER BY event_no,t,string_id,sensor_id"
            )
            rows = np.asarray(
                connection.execute(query, identifiers.tolist()).fetchall(), dtype=float
            ).reshape(-1, 7)
            if len(rows):
                pairs, inverse = np.unique(rows[:, 1:3], axis=0, return_inverse=True)
                if np.any(pairs != np.floor(pairs)):
                    message = "Source sensor identifiers are not integer valued"
                    raise ValueError(message)
                indices = np.asarray([mapping[(int(a), int(b))] for a, b in pairs], dtype=np.int32)[
                    inverse
                ]
                if not np.allclose(positions[indices], rows[:, 4:7], rtol=0, atol=1e-5):
                    message = "Source photon positions disagree with fixed geometry"
                    raise ValueError(message)
                local = np.searchsorted(identifiers, rows[:, 0].astype(np.int64))
                number = np.bincount(local, minlength=len(block))
            else:
                indices, number = np.empty(0, np.int32), np.zeros(len(block), dtype=np.int64)
            block_offsets = np.concatenate(([0], np.cumsum(number))).astype(np.int64)
            validate_photon_arrays(block_offsets, indices, rows[:, 3], positions, live, edges)
            sensors.resize((cursor + len(rows),))
            times.resize((cursor + len(rows),))
            sensors[cursor:] = indices
            times[cursor:] = rows[:, 3]
            offsets[first + 1 : first + len(block) + 1] = cursor + block_offsets[1:]
            cursor += len(rows)
        for name, values in {
            "photons/event_offsets": offsets,
            "geometry/positions_m": positions,
            "geometry/live": live,
            "time_edges_ns": edges,
            "events/event_id": selected.event_no.to_numpy(np.int64),
            "events/configuration_id": selected.group_id.to_numpy(np.int64),
            "events/partition": selected.partition.to_numpy(np.uint8),
        }.items():
            handle.create_dataset(name, data=values, compression="gzip")
        handle.attrs["construction_complete"] = True
    partial.replace(destination)
    return {
        "path": str(destination),
        "sha256": shard_sha256(destination),
        "events": len(selected),
        "photons": cursor,
        "elapsed_seconds": time.perf_counter() - begin,
    }


def import_cohort(config: dict) -> dict:
    """Materialize separate development/test folds and nested engineering shards."""
    output = Path(config["data"]["output"])
    cohort = pd.read_parquet(output / "cohort_manifest.parquet")
    positions, mapping = _geometry_mapping(config, output)
    edges = np.linspace(*config["data"]["time_window_ns"], config["data"]["time_bins"] + 1)
    audit = json.loads((output / "source_audit.json").read_text())
    provenance = {
        "kind": "public_prometheus_pone_water_photon_simulation",
        "statistical_contract": audit["interpretation"],
        "conditional_poisson_certified": False,
        "source_database_sha256": audit["source_database_sha256"],
        "source_archive_sha256": config["source"]["archive_sha256"],
        "cohort_manifest_sha256": shard_sha256(output / "cohort_manifest.parquet"),
        "clock": "source_t_retained_no_per_event_alignment",
        "observed_source_sensor_mapping": [
            [a, b, value] for (a, b), value in sorted(mapping.items())
        ],
        "grouping": "exact_stored_initial_state; unknown_additional_latent_relatives_qualified",
    }
    results = {}
    with source_connection(config["source"]["database"]) as connection:
        for engineering in (True, False):
            prefix = "engineering_" if engineering else ""
            for label, name in enumerate(("train", "validation", "test")):
                selected = cohort.loc[
                    (cohort.partition == label) & (cohort.engineering if engineering else True)
                ].sort_values("event_no")
                results[prefix + name] = _write_fold(
                    connection,
                    selected,
                    output / f"{prefix}{name}.h5",
                    positions,
                    mapping,
                    edges,
                    config,
                    provenance,
                )
    write_json(output / "shards.json", results)
    return results


def main() -> None:
    """Execute one explicitly selected source/cohort preparation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "cohort", "import"))
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_study.yml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    action = {"audit": audit_source, "cohort": make_cohort, "import": import_cohort}[args.action]
    result = action(config)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
