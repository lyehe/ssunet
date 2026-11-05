"""Prepare fresh grouped observations for mechanism and labelled utility studies."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from ..ssunet.datasets.neutrino import shard_sha256
from .neutrino_prepare import import_cohort, stable_group_rank, write_json


def prepare_followup(config):
    """Exclude every known relative of main/pilot events before count-blind ranking."""
    main = yaml.safe_load(Path(config["main_study_config"]).read_text())
    parent, output = Path(main["data"]["output"]), Path(config["data"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "cohort_manifest.parquet"
    if destination.exists():
        message = "Follow-up cohort already exists; preserve its original membership"
        raise FileExistsError(message)
    index = pd.read_parquet(parent / "source_group_index.parquet")
    original = pd.read_parquet(parent / "cohort_manifest.parquet")
    with h5py.File(main["source"]["pilot"], "r") as handle:
        pilot_ids = handle["events/event_id"][:]
    pilot_groups = index.loc[index.event_no.isin(pilot_ids), "group_id"]
    excluded = np.union1d(original.group_id.unique(), pilot_groups.unique())
    eligible = index.loc[~index.group_id.isin(excluded), ["event_no", "group_id"]]
    sizes = eligible.groupby("group_id", sort=True).size()
    namespace = main["source"]["archive_sha256"]
    ranks = np.asarray(
        [
            stable_group_rank(group, config["data"]["cohort_seed"], namespace)
            for group in sizes.index
        ],
        dtype=np.uint64,
    )
    order = np.argsort(ranks, kind="stable")
    number = (
        int(np.searchsorted(np.cumsum(sizes.to_numpy()[order]), config["data"]["cohort_events"]))
        + 1
    )
    groups = sizes.index.to_numpy()[order[:number]]
    selected = eligible.loc[eligible.group_id.isin(groups)].copy()
    partition = {
        int(group): stable_group_rank(group, config["data"]["partition_seed"], namespace) % 10000
        for group in groups
    }
    hashed = selected.group_id.map(partition).to_numpy()
    selected["partition"] = np.where(hashed < 8000, 0, np.where(hashed < 9000, 1, 2)).astype(
        np.uint8
    )
    selected["engineering"] = False
    selected.sort_values("event_no", inplace=True)
    selected.to_parquet(destination, index=False)
    report = {
        "events": len(selected),
        "groups": len(groups),
        "excluded_main_and_pilot_groups": len(excluded),
        "known_main_group_overlap": int(selected.group_id.isin(original.group_id).sum()),
        "known_pilot_group_overlap": int(selected.group_id.isin(pilot_groups).sum()),
        "selection_uses_photon_counts": False,
        "group_partitions_disjoint": bool(
            (selected.groupby("group_id").partition.nunique() == 1).all()
        ),
        "partitions": {
            name: int((selected.partition == label).sum())
            for label, name in enumerate(("train", "validation", "test"))
        },
        "manifest_sha256": shard_sha256(destination),
        "configuration": config,
    }
    write_json(output / "cohort_manifest.json", report)
    for name in ("source_audit.json", "source_sensor_mapping.parquet"):
        shutil.copyfile(parent / name, output / name)
    imported = {**main, "data": {**main["data"], **config["data"]}}
    # The importer also writes nested engineering folds. Do not create empty
    # nested files: use a complete nested copy of this small follow-up cohort.
    selected["engineering"] = True
    selected.to_parquet(destination, index=False)
    report["manifest_sha256"] = shard_sha256(destination)
    write_json(output / "cohort_manifest.json", report)
    shards = import_cohort(imported)
    write_json(output / "shards.json", shards)
    return report


def main():
    """Materialize a new count-blind cohort and independently held-out probe fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_ablations.yml"))
    args = parser.parse_args()
    prepare_followup(yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    main()
