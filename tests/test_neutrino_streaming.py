"""Checks for bounded source loading, stable cohorts and grouped uncertainty."""

import pickle

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.ssunet.datasets.neutrino import (
    EpochPhotonBatchSampler,
    NeutrinoPhotonDataset,
    partition_configurations,
    write_photon_shard,
)
from src.ssunet.models.neutrino_dense import DensePhotonReference
from src.ssunet.neutrino_evaluation import (
    PhotonScoreRecords,
    freeze_photon_view,
    model_event_nll,
    paired_photon_interval,
)
from src.ssunet.neutrino_streaming import (
    BoundedPhotonDonors,
    PhotonBaselineBank,
    collect_training_reference,
    select_baselines,
    stream_donor_scores,
    stream_scores,
)
from src.ssunet.neutrino_training import train_photon_seed
from src.tools.neutrino_prepare import make_cohort


@pytest.fixture
def streaming_shard(tmp_path):
    """Use fixed events with zeros, multi-count bins and repeated latent groups."""
    identifiers = np.arange(60, dtype=np.int64)
    number = np.where(identifiers % 4 == 0, 0, 80)
    groups = identifiers // 2
    path = tmp_path / "stream.h5"
    write_photon_shard(
        path,
        offsets=np.concatenate(([0], np.cumsum(number))),
        sensor_index=np.repeat(identifiers % 2, number).astype(np.int32),
        time_ns=np.full(int(number.sum()), 10.0),
        positions_m=np.asarray([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float),
        live=np.asarray([True, True, False]),
        event_ids=identifiers,
        configuration_ids=groups,
        partition=partition_configurations(groups, 7),
        time_edges_ns=np.linspace(0, 200, 9),
        provenance={"kind": "test_fixture"},
    )
    return path


def test_lazy_and_eager_views_agree_without_resident_photon_arrays(streaming_shard):
    """Lazy mode reads the same event counts/draws and drops HDF5 handles on spawn."""
    eager = NeutrinoPhotonDataset(streaming_shard, "all")
    lazy = NeutrinoPhotonDataset(streaming_shard, "all", lazy=True)
    assert lazy.sensor_index is None
    assert lazy.time_ns is None
    for index in range(len(eager)):
        torch.testing.assert_close(eager[index]["input"], lazy[index]["input"])
        torch.testing.assert_close(eager[index]["target"], lazy[index]["target"])
    clone = pickle.loads(pickle.dumps(lazy))  # noqa: S301 -- local fixture roundtrip only.
    assert clone._handle is None
    torch.testing.assert_close(lazy[1]["input"], clone[1]["input"])
    lazy.close()
    clone.close()


def test_explicit_ids_reject_missing_or_wrong_partition_events(streaming_shard):
    """An engineering subset cannot accidentally select reserved test IDs."""
    train = NeutrinoPhotonDataset(streaming_shard, "train")
    ids = train.event_ids[train.indices][:3]
    selected = NeutrinoPhotonDataset(streaming_shard, "train", selected_event_ids=ids, lazy=True)
    np.testing.assert_array_equal(selected.event_ids[selected.indices], ids)
    with pytest.raises(ValueError, match="partition"):
        NeutrinoPhotonDataset(streaming_shard, "validation", selected_event_ids=ids, lazy=True)
    with pytest.raises(ValueError, match="unique"):
        NeutrinoPhotonDataset(streaming_shard, "train", selected_event_ids=np.repeat(ids[0], 2))


def test_lazy_read_rejects_corrupted_times_when_the_event_is_loaded(streaming_shard):
    """Deferred validation still rejects malformed observations rather than dropping them."""
    with h5py.File(streaming_shard, "r+") as handle:
        handle["photons/time_ns"][0] = np.nan
    dataset = NeutrinoPhotonDataset(streaming_shard, "all", lazy=True)
    with pytest.raises(ValueError, match="finite"):
        dataset.counts(1)
    dataset.close()


def test_persistent_spawn_workers_receive_epoch_in_indices(streaming_shard):
    """Training resplits change even when persistent worker dataset copies stay alive."""
    dataset = NeutrinoPhotonDataset(streaming_shard, "train", lazy=True)
    dataset[0]
    sampler = EpochPhotonBatchSampler(len(dataset), 8, seed=9, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    try:
        before = {
            int(event): counts.clone()
            for batch in loader
            for event, counts in zip(batch["event_id"], batch["input"], strict=True)
        }
        sampler.epoch = 1
        after = {
            int(event): counts.clone()
            for batch in loader
            for event, counts in zip(batch["event_id"], batch["input"], strict=True)
        }
        assert before.keys() == after.keys()
        assert any(not torch.equal(before[event], after[event]) for event in before)
        expected = {
            int(dataset[(i, 1)]["event_id"]): dataset[(i, 1)]["input"] for i in range(len(dataset))
        }
        assert all(torch.equal(after[event], expected[event]) for event in after)
    finally:
        loader._iterator._shutdown_workers()
        dataset.close()


def test_chunked_bootstrap_matches_full_resampling_and_chunk_order():
    """Whole-family resampling and ratio weighting survive bounded chunking."""
    groups = np.repeat(np.arange(17), 3)
    generator = np.random.default_rng(5)
    targets = generator.poisson(3, size=len(groups))
    delta = generator.normal(size=len(groups)) * targets
    records = PhotonScoreRecords(np.arange(len(groups)), groups, targets + 1, targets)
    mask = records.strata["all"]
    small = paired_photon_interval(delta, records, mask, seed=7, draws=1001, chunk_size=7)
    full = paired_photon_interval(delta, records, mask, seed=7, draws=1001, chunk_size=1001)
    assert small == full
    group_delta = np.asarray([delta[groups == group].sum() for group in np.unique(groups)])
    group_count = np.asarray([targets[groups == group].sum() for group in np.unique(groups)])
    samples = np.random.default_rng(7).integers(17, size=(1001, 17))
    values = group_delta[samples].sum(1) / group_count[samples].sum(1)
    np.testing.assert_allclose(
        small["descriptive_95_percent_interval"], np.quantile(values, [0.025, 0.975])
    )


def test_cohort_expansion_preserves_assignments_and_excludes_pilot_families(tmp_path):
    """Growing the cohort changes membership, never an included family's fold."""
    source_index = pd.DataFrame({"event_no": np.arange(200), "group_id": np.arange(200) // 2})
    pilot = tmp_path / "pilot.h5"
    with h5py.File(pilot, "w") as handle:
        handle.create_dataset("events/event_id", data=[0, 10])
    outputs = []
    for name, size in (("small", 40), ("large", 100)):
        output = tmp_path / name
        output.mkdir()
        source_index.to_parquet(output / "source_group_index.parquet", index=False)
        config = {
            "source": {"pilot": str(pilot), "archive_sha256": "source"},
            "data": {
                "output": str(output),
                "cohort_seed": 17,
                "partition_seed": 18,
                "cohort_events": size,
                "engineering_events": 20,
            },
        }
        result = make_cohort(config)
        assert result["pilot_event_overlap"] == 0
        assert result["pilot_group_overlap"] == 0
        outputs.append(pd.read_parquet(output / "cohort_manifest.parquet"))
    overlap = outputs[0].merge(outputs[1], on="event_no", suffixes=("_small", "_large"))
    assert len(overlap) == len(outputs[0])
    np.testing.assert_array_equal(overlap.partition_small, overlap.partition_large)
    assert all(frame.groupby("group_id").partition.nunique().max() == 1 for frame in outputs)


@pytest.fixture
def streaming_reference(streaming_shard):
    """Use a sparse retained view and a small prespecified control grid."""
    training = NeutrinoPhotonDataset(streaming_shard, "train", p=0.05, training=False, lazy=True)
    validation = NeutrinoPhotonDataset(streaming_shard, "validation", p=0.05, lazy=True)
    settings = {
        "original_prior_pseudocount": 0.5,
        "frozen_prior_time_ns": 25,
        "frozen_prior_pseudocount": 0.001,
        "frozen_mixture_prior_fraction": 0.25,
        "prior_time_ns": [0, 25],
        "prior_pseudocounts": [0.001, 0.5],
        "input_time_ns": [0, 25],
        "input_space_m": [0, None],
        "mixture_prior_fractions": [0.25, 0.5],
    }
    reference = collect_training_reference(training)
    bank = PhotonBaselineBank(
        reference["observed_counts"],
        training.positions_m,
        training.live,
        training.edges_ns,
        settings,
    )
    yield training, validation, reference, bank
    training.close()
    validation.close()


def test_streaming_likelihood_matches_frozen_pilot_and_batch_sizes(streaming_reference):
    """Streaming reduction preserves per-event IDs, photon weights and model scores."""
    torch.set_num_threads(1)
    training, validation, reference, bank = streaming_reference
    model = DensePhotonReference(4)
    view = freeze_photon_view(validation)
    expected = model_event_nll(model, view, "cpu", 3)
    records, losses = stream_scores(
        validation, batch_size=2, device="cpu", model=model, baselines=bank
    )
    records2, losses2 = stream_scores(
        validation, batch_size=7, device="cpu", model=model, baselines=bank
    )
    np.testing.assert_array_equal(records.event_ids, view.event_ids)
    np.testing.assert_array_equal(records.target_totals, view.target_totals)
    np.testing.assert_array_equal(records.retained_totals, records2.retained_totals)
    np.testing.assert_allclose(losses["model"], expected, rtol=2e-6, atol=2e-5)
    for name in losses:
        np.testing.assert_allclose(losses[name], losses2[name], rtol=2e-6, atol=2e-5)
    assert reference["photons_in_window"] == sum(
        int(training.counts(i)[0].sum()) for i in range(len(training))
    )
    for probability in bank.probabilities(view.inputs).values():
        torch.testing.assert_close(probability.sum((1, 2)), torch.ones(len(view.inputs)))
        assert torch.count_nonzero(probability[:, 2]) == 0


def test_finite_control_grid_uses_validation_and_fits_only_training(streaming_reference):
    """The recorded baseline winner is the lowest primary validation score."""
    training, validation, reference, bank = streaming_reference
    selected = select_baselines(reference, validation, bank, 3)
    assert len(selected["candidates"]) == 12
    values = selected["method_primary_nll"]
    assert selected["selected_method"] == min(values, key=values.get)
    with pytest.raises(ValueError, match="partition"):
        collect_training_reference(validation)
    assert reference["events"] == len(training)


def test_streaming_donors_are_training_only_and_cache_is_bounded(streaming_reference):
    """Control inputs preserve source strata without caching the entire cohort."""
    training, validation, reference, _bank = streaming_reference
    donors = BoundedPhotonDonors(training, reference["retained_totals"], seed=7, cache_events=2)
    model = DensePhotonReference(4)
    values, exact, identifiers = stream_donor_scores(
        model, validation, donors, batch_size=3, device="cpu", draws=2
    )
    assert np.isfinite(values).all()
    assert len(donors.cache) <= 2
    assert np.isin(identifiers, training.event_ids[training.indices]).all()
    assert not np.isin(identifiers, validation.event_ids[validation.indices]).any()
    totals = np.asarray([int(validation[i]["input"].sum()) for i in range(len(validation))])
    np.testing.assert_array_equal(exact, np.isin(totals, reference["retained_totals"]))
    before = donors.choose(123, 3, 0)
    assert donors.choose(123, 3, 0) == before


def test_checkpoint_resume_preserves_model_and_optimizer_updates(
    streaming_shard, tmp_path, monkeypatch
):
    """A mid-epoch interruption resumes the same batches and yields identical weights."""
    monkeypatch.setattr("src.ssunet.neutrino_training._peak_rss", lambda: 1.0)
    torch.set_num_threads(1)
    training = NeutrinoPhotonDataset(streaming_shard, "train", p=0.05, lazy=True)
    validation = NeutrinoPhotonDataset(streaming_shard, "validation", p=0.05, lazy=True)
    diagnostic = NeutrinoPhotonDataset(streaming_shard, "train", p=0.05, training=False, lazy=True)
    config = {
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "num_workers": 0,
            "max_process_rss_gib": 8,
            "max_gpu_allocated_gib": 12,
            "max_training_seconds_per_seed": 60,
            "max_evaluation_seconds_per_seed": 60,
        },
        "evaluation": {"epochs": [0, 1, 2]},
    }
    views = (training, validation, diagnostic)
    factory = lambda: DensePhotonReference(4)  # noqa: E731
    identity = {"data": "fixed_fixture"}
    full_dir, resumed_dir = tmp_path / "full", tmp_path / "resumed"
    full = train_photon_seed(config, 42, views, factory, full_dir, "cpu", identity)
    assert full["completed_fixed_epochs"]
    partial = train_photon_seed(
        config, 42, views, factory, resumed_dir, "cpu", identity, stop_after_updates=3
    )
    assert not partial["completed_fixed_epochs"]
    assert partial["training_state"]["next_batch"] == 3
    resumed = train_photon_seed(config, 42, views, factory, resumed_dir, "cpu", identity)
    assert resumed["completed_fixed_epochs"]
    assert resumed["training_state"]["optimizer_steps"] == full["training_state"]["optimizer_steps"]
    expected = torch.load(full_dir / "last.pt", weights_only=True)
    got = torch.load(resumed_dir / "last.pt", weights_only=True)
    for name, value in expected["state_dict"].items():
        torch.testing.assert_close(value, got["state_dict"][name], rtol=0, atol=0)
    for dataset in views:
        dataset.close()
