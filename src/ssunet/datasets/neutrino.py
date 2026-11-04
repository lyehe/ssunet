"""Event-based photon observations, preserving integer counts and empty events."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def partition_configurations(configuration_ids: np.ndarray, seed: int = 42) -> np.ndarray:
    """Assign whole latent configurations to an 80/10/10 partition."""
    groups = np.unique(configuration_ids)
    if len(groups) < 10:
        message = "At least ten configurations are required for an 80/10/10 split"
        raise ValueError(message)
    shuffled = np.random.default_rng(seed).permutation(groups)
    labels = np.zeros(len(groups), dtype=np.uint8)
    labels[int(0.8 * len(groups)) : int(0.9 * len(groups))] = 1
    labels[int(0.9 * len(groups)) :] = 2
    lookup = dict(zip(shuffled.tolist(), labels.tolist(), strict=True))
    return np.asarray([lookup[value] for value in configuration_ids], dtype=np.uint8)


def _validate_geometry(positions_m: np.ndarray, live: np.ndarray) -> None:
    """Validate the fixed geometry and physical live-sensor support."""
    if positions_m.ndim != 2 or positions_m.shape[1] != 3 or not np.isfinite(positions_m).all():
        message = "Sensor positions must be finite [sensor, 3] coordinates"
        raise ValueError(message)
    if live.shape != (len(positions_m),) or live.dtype != np.bool_ or not live.any():
        message = "Live-sensor mask must be boolean and contain live sensors"
        raise ValueError(message)


def validate_photon_arrays(
    offsets: np.ndarray,
    sensor_index: np.ndarray,
    time_ns: np.ndarray,
    positions_m: np.ndarray,
    live: np.ndarray,
    time_edges_ns: np.ndarray,
) -> None:
    """Reject malformed ragged data rather than rounding or clipping observations."""
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) < 2:
        message = "Event offsets must be a one-dimensional integer array"
        raise ValueError(message)
    if offsets[0] != 0 or np.any(np.diff(offsets.astype(np.int64)) < 0):
        message = "Event offsets must start at zero and be nondecreasing"
        raise ValueError(message)
    if sensor_index.ndim != 1 or sensor_index.dtype.kind not in "iu":
        message = "Photon sensor indices must be integers"
        raise ValueError(message)
    if time_ns.ndim != 1 or offsets[-1] != len(time_ns) or len(sensor_index) != len(time_ns):
        message = "Photon arrays and event offsets disagree"
        raise ValueError(message)
    _validate_geometry(positions_m, live)
    if len(sensor_index) and (sensor_index.min() < 0 or sensor_index.max() >= len(live)):
        message = "Photon sensor index is outside the geometry"
        raise ValueError(message)
    if len(sensor_index) and not live[sensor_index].all():
        message = "A photon was assigned to an inactive sensor"
        raise ValueError(message)
    if not np.isfinite(time_ns).all():
        message = "Photon timestamps must be finite"
        raise ValueError(message)
    if (
        time_edges_ns.ndim != 1
        or len(time_edges_ns) < 2
        or not np.isfinite(time_edges_ns).all()
        or np.any(np.diff(time_edges_ns) <= 0)
    ):
        message = "Time edges must be finite and strictly increasing"
        raise ValueError(message)


def write_photon_shard(
    path: str | Path,
    *,
    offsets: np.ndarray,
    sensor_index: np.ndarray,
    time_ns: np.ndarray,
    positions_m: np.ndarray,
    live: np.ndarray,
    event_ids: np.ndarray,
    configuration_ids: np.ndarray,
    partition: np.ndarray,
    time_edges_ns: np.ndarray,
    provenance: dict,
) -> None:
    """Save observations separately from any simulation truth or diagnostic rate."""
    validate_photon_arrays(offsets, sensor_index, time_ns, positions_m, live, time_edges_ns)
    n_events = len(offsets) - 1
    if any(array.shape != (n_events,) for array in (event_ids, configuration_ids, partition)):
        message = "Event metadata lengths disagree"
        raise ValueError(message)
    if len(np.unique(event_ids)) != n_events or not np.isin(partition, [0, 1, 2]).all():
        message = "Event IDs must be unique and partition labels must be 0/1/2"
        raise ValueError(message)
    order = np.argsort(configuration_ids)
    same_group = configuration_ids[order][1:] == configuration_ids[order][:-1]
    if np.any(same_group & (partition[order][1:] != partition[order][:-1])):
        message = "One latent configuration crosses evaluation partitions"
        raise ValueError(message)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        message = f"Refusing to overwrite an existing photon shard: {destination}"
        raise FileExistsError(message)
    with h5py.File(destination, "x") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        for name, data in {
            "photons/event_offsets": offsets,
            "photons/sensor_index": sensor_index,
            "photons/time_ns": time_ns,
            "geometry/positions_m": positions_m,
            "geometry/live": live,
            "time_edges_ns": time_edges_ns,
            "events/event_id": event_ids,
            "events/configuration_id": configuration_ids,
            "events/partition": partition,
        }.items():
            handle.create_dataset(name, data=data, compression="gzip")


def bin_photons(
    sensor_index: np.ndarray, time_ns: np.ndarray, n_sensors: int, edges: np.ndarray
) -> tuple[np.ndarray, int]:
    """Bin one event using a fixed clock and report photons outside the window."""
    n_bins = len(edges) - 1
    keep = (time_ns >= edges[0]) & (time_ns < edges[-1])
    bins = np.searchsorted(edges, time_ns[keep], side="right") - 1
    flat = sensor_index[keep].astype(np.int64) * n_bins + bins
    counts = np.bincount(flat, minlength=n_sensors * n_bins).reshape(n_sensors, n_bins)
    return counts.astype(np.int64), int(np.count_nonzero(~keep))


def split_photon_counts(
    counts: np.ndarray, p: float, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Thin true integer counts without clipping, redrawing, or target normalization."""
    if counts.dtype.kind not in "iu" or np.any(counts < 0):
        message = "Photon splitting requires nonnegative integer counts"
        raise ValueError(message)
    if not 0 < p < 1:
        message = "Splitting probability must lie strictly between zero and one"
        raise ValueError(message)
    retained = generator.binomial(counts, p).astype(np.int64)
    return retained, counts - retained


class NeutrinoPhotonDataset(Dataset):
    """Read one complete photon event per item, with reproducible epoch-level splits."""

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        *,
        p: float = 0.5,
        seed: int = 42,
        training: bool | None = None,
        lazy: bool = False,
        selected_event_ids: np.ndarray | None = None,
    ):
        """Load ragged observations and geometry, excluding simulation truth."""
        if split not in {"train", "validation", "test", "all"} or not 0 < p < 1:
            message = "Invalid dataset partition or splitting probability"
            raise ValueError(message)
        self.path = str(Path(path).resolve())
        self.lazy = lazy
        self.split = split
        self._handle = None
        self._handle_pid = None
        self._photon_datasets = None
        with h5py.File(self.path, "r") as handle:
            if handle.attrs.get("schema_version") != 1:
                message = "Unsupported photon shard schema"
                raise ValueError(message)
            if not handle.attrs.get("construction_complete", True):
                message = "Photon shard construction is incomplete"
                raise ValueError(message)
            self.offsets = handle["photons/event_offsets"][:]
            self.sensor_index = None if lazy else handle["photons/sensor_index"][:]
            self.time_ns = None if lazy else handle["photons/time_ns"][:]
            self.positions_m = handle["geometry/positions_m"][:]
            self.live = handle["geometry/live"][:]
            self.edges_ns = handle["time_edges_ns"][:]
            event_ids = handle["events/event_id"][:]
            configurations = handle["events/configuration_id"][:]
            partition = handle["events/partition"][:]
            self.provenance = json.loads(handle.attrs["provenance_json"])
            if lazy:
                self._validate_lazy_layout(handle)
        if not lazy:
            validate_photon_arrays(
                self.offsets,
                self.sensor_index,
                self.time_ns,
                self.positions_m,
                self.live,
                self.edges_ns,
            )
        self.indices = (
            np.arange(len(event_ids))
            if split == "all"
            else np.flatnonzero(partition == {"train": 0, "validation": 1, "test": 2}[split])
        )
        self.event_ids = event_ids
        self.configuration_ids = configurations
        if selected_event_ids is not None:
            self._select_events(np.asarray(selected_event_ids))
        self.p = p
        self.seed = seed
        self.training = split == "train" if training is None else training
        self.epoch = 0

    def _validate_lazy_layout(self, handle) -> None:
        """Check metadata without materializing cohort-sized photon arrays."""
        sensors = handle["photons/sensor_index"]
        times = handle["photons/time_ns"]
        if (
            self.offsets.ndim != 1
            or self.offsets.dtype.kind not in "iu"
            or len(self.offsets) < 2
            or self.offsets[0] != 0
            or np.any(np.diff(self.offsets.astype(np.int64)) < 0)
            or sensors.ndim != 1
            or times.ndim != 1
            or sensors.dtype.kind not in "iu"
            or self.offsets[-1] != len(sensors)
            or len(sensors) != len(times)
        ):
            message = "Malformed lazy photon arrays or event boundaries"
            raise ValueError(message)
        validate_photon_arrays(
            np.asarray([0, 0], dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0),
            self.positions_m,
            self.live,
            self.edges_ns,
        )

    def _select_events(self, identifiers) -> None:
        """Select persisted event IDs without silently including another partition."""
        if identifiers.ndim != 1 or len(np.unique(identifiers)) != len(identifiers):
            message = "Selected event IDs must be a unique one-dimensional array"
            raise ValueError(message)
        available = self.event_ids[self.indices]
        if not np.isin(identifiers, available).all():
            message = "Selected event IDs are missing or outside the requested partition"
            raise ValueError(message)
        self.indices = self.indices[np.isin(available, identifiers)]

    def __getstate__(self):
        """Give spawned workers paths/metadata, never a parent HDF5 handle."""
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        state["_photon_datasets"] = None
        return state

    def close(self) -> None:
        """Close this process's lazy file handle explicitly."""
        if self._handle is not None:
            self._photon_datasets = None
            self._handle.close()
            self._handle = None
            self._handle_pid = None

    def _lazy_handle(self):
        """Open one read-only handle per process, independently of worker order."""
        if self._handle_pid != os.getpid() or self._handle is None or not self._handle.id.valid:
            self.close()
            self._handle = h5py.File(self.path, "r", rdcc_nbytes=8 * 1024**2, rdcc_nslots=10007)
            self._handle_pid = os.getpid()
            self._photon_datasets = (
                self._handle["photons/sensor_index"],
                self._handle["photons/time_ns"],
            )
        return self._handle

    def __len__(self) -> int:
        """Return the number of original events, including empty observations."""
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        """Change training draws; evaluation draws remain fixed across epochs."""
        if epoch < 0:
            message = "Epoch must be nonnegative"
            raise ValueError(message)
        self.epoch = epoch

    def counts(self, index: int) -> tuple[np.ndarray, int]:
        """Return an event histogram before splitting and its out-of-window count."""
        source = self.indices[index]
        begin, end = self.offsets[source : source + 2]
        if self.lazy:
            self._lazy_handle()
            sensors = self._photon_datasets[0][begin:end]
            times = self._photon_datasets[1][begin:end]
            validate_photon_arrays(
                np.asarray([0, len(times)], dtype=np.int64),
                sensors,
                times,
                self.positions_m,
                self.live,
                self.edges_ns,
            )
        else:
            sensors, times = self.sensor_index[begin:end], self.time_ns[begin:end]
        return bin_photons(sensors, times, len(self.live), self.edges_ns)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, torch.Tensor]:
        """Return counts for supervision and inputs, plus observation metadata."""
        index, requested_epoch = index if isinstance(index, tuple) else (index, self.epoch)
        if requested_epoch < 0:
            message = "Epoch must be nonnegative"
            raise ValueError(message)
        source = self.indices[index]
        counts, dropped = self.counts(index)
        epoch = requested_epoch if self.training else 0
        sequence = np.random.SeedSequence([self.seed, int(self.event_ids[source]), epoch])
        retained, target = split_photon_counts(counts, self.p, np.random.default_rng(sequence))
        return {
            "input": torch.from_numpy(retained),
            "target": torch.from_numpy(target),
            "positions_m": torch.from_numpy(self.positions_m.astype(np.float32)),
            "live": torch.from_numpy(self.live),
            "p": torch.tensor(self.p, dtype=torch.float32),
            "event_id": torch.tensor(int(self.event_ids[source]), dtype=torch.int64),
            "outside_window": torch.tensor(dropped, dtype=torch.int64),
        }


class EpochPhotonBatchSampler(Sampler):
    """Carry epoch seeds in indices so persistent workers receive fresh splits."""

    def __init__(self, events: int, batch_size: int, *, seed: int, shuffle: bool = True):
        """Record a deterministic epoch order without storing dense observations."""
        if events < 1 or batch_size < 1:
            message = "Epoch batching needs positive event and batch counts"
            raise ValueError(message)
        self.events, self.batch_size, self.seed, self.shuffle = events, batch_size, seed, shuffle
        self.epoch = 0
        self.start_batch = 0

    def __len__(self):
        """Return the number of batches, preserving a final partial batch."""
        return max(0, (self.events + self.batch_size - 1) // self.batch_size - self.start_batch)

    def __iter__(self):
        """Yield explicit (event index, epoch) pairs for each bounded batch."""
        order = np.arange(self.events)
        if self.shuffle:
            order = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch])
            ).permutation(order)
        for begin in range(self.start_batch * self.batch_size, len(order), self.batch_size):
            yield [(int(index), self.epoch) for index in order[begin : begin + self.batch_size]]


def shard_sha256(path: str | Path) -> str:
    """Hash an artifact without loading the entire file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
