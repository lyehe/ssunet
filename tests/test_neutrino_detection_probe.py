"""Check frozen feature fairness, train-only scaling, fitting and locked fresh folds."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.ssunet.datasets.neutrino import shard_sha256
from src.ssunet.models.neutrino_dense import DensePhotonReference
from src.ssunet.neutrino_detection import PhotonWindows, normal_kernel
from src.ssunet.neutrino_detection_probe import (
    BASE_SCALARS,
    balanced_log_loss,
    encoder_features,
    encoder_features_tensor,
    feature_design,
    fit_head_candidates,
    observation_features,
    pdf_summary,
    predict_head,
)
from src.tools.neutrino_detection_probe import (
    FrozenFeatureExtractor,
    generate_fold,
    run_probe,
    validate_probe_config,
    verify_probe_inputs,
)


def tiny_observations():
    """Include an empty observation and repeated photons in one observed cell."""
    return PhotonWindows(
        np.array([0, 0, 3]),
        np.array([0, 0, 5]),
        np.array([True, False, True]),
        np.array([0.2, 0.8]),
    )


def test_raw_features_conserve_counts_and_are_symmetric_in_recorded_halves():
    """Base features do not depend on assigning an arbitrary name to A versus B."""
    observed = tiny_observations()
    a, b = observed.dense_batch(0, 2, (2, 4))
    reference = {
        "prior": np.arange(1, 9, dtype=float).reshape(2, 4) / 36,
        "positions": np.array([[0, 0, 0], [100, 0, 0]]),
        "background": 1.0,
        "amplitudes": [0.5, 2.0],
        "sensor_kernel": normal_kernel([[0], [100]], 100),
        "time_kernel": normal_kernel(np.arange(4)[:, None], 1),
        "mixture_fraction": 0.5,
    }
    features, controls = observation_features(observed, 0, 2, a, b, reference)
    swapped = PhotonWindows(
        observed.offsets, observed.locations, ~observed.retained, observed.randomizers
    )
    other, other_controls = observation_features(swapped, 0, 2, b, a, reference)
    np.testing.assert_allclose(features, other)
    for name, values in controls.items():
        np.testing.assert_allclose(values, other_controls[name])
    scalar_start = 2 + 4 + 2
    assert features[1, scalar_start + BASE_SCALARS.index("count")] == 3
    assert features[1, scalar_start + BASE_SCALARS.index("same_sensor_pairs")] == 1
    assert features[0, scalar_start + BASE_SCALARS.index("empty")] == 1
    assert np.isfinite(features).all()


def test_pdf_features_use_fixed_normalized_groups():
    """Coarse probability masses sum to one and empty priors carry no fake counts."""
    uniform = np.full((2, 4, 8), 1 / 32)
    features = pdf_summary(uniform, 2, 4)
    np.testing.assert_allclose(features[:, :2], 0.5)
    np.testing.assert_allclose(features[:, 2:6], 0.25)
    np.testing.assert_allclose(features[:, 6:], 0, atol=1e-15)
    with pytest.raises(ValueError, match="divide"):
        pdf_summary(uniform, 3, 4)


def test_encoder_features_are_symmetric_and_empty_cross_scores_are_zero():
    """Score actual photons once; swapping A/B leaves event-level features unchanged."""
    observed = tiny_observations()
    qa = np.full((2, 2, 4), 1 / 8)
    qb = np.broadcast_to(np.arange(1, 9).reshape(2, 4) / 36, qa.shape).copy()
    ha, hb = np.ones((2, 3)), np.arange(6).reshape(2, 3)
    prior = np.ones((2, 4)) / 8
    settings = {"sensor_groups": 1, "time_groups": 2}
    features, cross = encoder_features(observed, 0, 2, qa, qb, ha, hb, prior, settings)
    swapped = PhotonWindows(
        observed.offsets, observed.locations, ~observed.retained, observed.randomizers
    )
    other, other_cross = encoder_features(swapped, 0, 2, qb, qa, hb, ha, prior, settings)
    np.testing.assert_allclose(features, other)
    np.testing.assert_allclose(cross, other_cross)
    assert cross[0] == 0
    assert np.isfinite(features).all()
    a, b = observed.dense_batch(0, 2, (2, 4))
    device_features, device_cross = encoder_features_tensor(
        a,
        b,
        torch.tensor(qa),
        torch.tensor(qb),
        torch.tensor(ha),
        torch.tensor(hb),
        prior,
        settings,
    )
    np.testing.assert_allclose(device_features, features, atol=1e-6)
    np.testing.assert_allclose(device_cross, cross, atol=1e-12)


def test_random_encoder_construction_restores_cpu_rng_and_freezes_every_weight():
    """Random features use the same architecture and do not train an encoder."""
    torch.set_num_threads(1)
    model = DensePhotonReference(2)
    scorer = SimpleNamespace(
        models=[model],
        shape=(3, 8),
        device=torch.device("cpu"),
        coordinates=torch.zeros((3, 3)),
        live=torch.ones(3, dtype=torch.bool),
    )
    rng_before = torch.random.get_rng_state().clone()
    extractor = FrozenFeatureExtractor(scorer, {"random_encoder_seeds": [42]})
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for models in extractor.models.values():
        for network in models:
            assert all(not weight.requires_grad for weight in network.parameters())
            assert len(network.sensor_context._forward_hooks) == 0
    observed = np.zeros((3, 3, 8), dtype=np.int64)
    observed[1, 0, 2] = 3
    probability, hidden = extractor.predict(observed, "self_split")
    np.testing.assert_allclose(probability.sum((1, 2)), 1, atol=1e-12)
    np.testing.assert_array_equal(probability[0], probability[2])
    np.testing.assert_array_equal(hidden[0], hidden[2])
    assert hidden.shape == (3, 8)
    assert len(model.sensor_context._forward_hooks) == 0
    parameter_counts = [
        sum(p.numel() for p in models[0].parameters()) for models in extractor.models.values()
    ]
    assert len(set(parameter_counts)) == 1


def test_zero_padding_preserves_head_layout_without_inventing_varying_features():
    """The random control, not zero padding, controls added effective capacity."""
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    padded = feature_design(base, None, 3)
    assert padded.shape == (2, 5)
    np.testing.assert_array_equal(padded[:, :2], base)
    np.testing.assert_array_equal(padded[:, 2:], 0)
    with pytest.raises(ValueError, match="row identity"):
        feature_design(base, np.zeros((3, 3)), 3)


def test_balanced_validation_loss_is_independent_of_artificial_class_mix():
    """Validation class weighting does not pretend the injection prior is physical."""
    scores, labels = np.array([-2.0, -0.5, 0.5, 2.0]), np.array([0, 0, 1, 1])
    repeated = np.array([0, 0, 0, 1, 1, 1, 2, 3])
    assert balanced_log_loss(scores, labels) == pytest.approx(
        balanced_log_loss(scores[repeated], labels[repeated])
    )
    with pytest.raises(ValueError, match="both event classes"):
        balanced_log_loss(scores, np.ones(4))


def test_logistic_head_fits_training_scaler_only_and_predictions_do_not_mutate_it():
    """Fit a meaningful signal feature, retain constants and freeze preprocessing."""
    torch.set_num_threads(1)
    coordinate = np.linspace(-2, 2, 80)
    base = np.column_stack([coordinate, np.full(len(coordinate), 3)]).astype(np.float32)
    extra = np.zeros((len(coordinate), 2), dtype=np.float32)
    training = {"base": base.copy(), "self_split": extra, "untrained": extra}
    validation = {"base": base.copy(), "self_split": extra, "untrained": extra}
    labels = (coordinate > 0).astype(float)
    config = yaml.safe_load(Path("config/neutrino_detection_probe.yml").read_text())
    config["head"]["penalties"] = [0.01, 0.1]
    calls = []
    head, selection = fit_head_candidates(
        training, labels, validation, labels, "raw_prior", config, lambda: calls.append(1)
    )
    np.testing.assert_array_equal(training["base"], base)
    np.testing.assert_allclose(head["mean"][:2], [0, 3], atol=1e-8)
    assert head["weight"][0] > 0
    assert head["scale"][1] == pytest.approx(config["head"]["feature_scale_floor"])
    assert calls
    values = predict_head(head, base, None, batch_size=7)
    np.testing.assert_allclose(values, predict_head(head, base, None, batch_size=80), atol=1e-12)
    assert balanced_log_loss(values, labels) < 0.2
    assert selection["selected_candidate"] == 0
    before = {
        key: value.copy() if isinstance(value, np.ndarray) else value for key, value in head.items()
    }
    outlier = np.array([[1000, -1000]], dtype=np.float32)
    assert np.isfinite(predict_head(head, outlier, None)).all()
    for key in ("mean", "scale", "weight"):
        np.testing.assert_array_equal(head[key], before[key])


def test_source_generation_retains_every_window_and_rejects_exact_fold_overlap():
    """Event labels stay separate and no count-based source selection occurs."""
    config = yaml.safe_load(Path("config/neutrino_detection_probe.yml").read_text())
    config["generator"]["signal_expected_photons"] = [0.5, 2]
    config["generator"]["train"] = {
        "source_groups": 2,
        "repetitions_per_source": 2,
        "background_windows": 3,
    }
    scorer = SimpleNamespace(
        prior=np.ones((2, 2)) / 4, signal_pdf=lambda source: np.ones((2, 2)) / 4
    )
    streams = np.random.SeedSequence(123).spawn(2)
    observed, truth = generate_fold(config, "train", scorer, streams, np.empty((0, 3)))
    assert len(observed) == 11
    assert (truth["expected_signal"] == 0).sum() == 3
    np.testing.assert_array_equal(np.bincount(truth["source_group"][3:]), [4, 4])
    with pytest.raises(ValueError, match="overlap"):
        generate_fold(config, "train", scorer, streams, truth["sources_m"])


def test_locked_head_change_is_rejected(tmp_path):
    """Changing head coefficients cannot silently reuse a calibration/test lock."""
    artifact = tmp_path / "head.npz"
    artifact.write_bytes(b"selected coefficients")
    identity = {"fixed": True}
    (tmp_path / "head_lock.json").write_text(
        json.dumps({"identity": identity, "artifact_sha256": {"head.npz": shard_sha256(artifact)}})
    )
    verify_probe_inputs(tmp_path, identity)
    artifact.write_bytes(b"changed coefficients")
    with pytest.raises(ValueError, match="locked head"):
        verify_probe_inputs(tmp_path, identity)


def test_probe_rejects_existing_outputs_and_mismatched_encoder_budget(tmp_path):
    """A new probe cannot overwrite an opened study or expand one encoder arm."""
    config = yaml.safe_load(Path("config/neutrino_detection_probe.yml").read_text())
    frozen = yaml.safe_load(Path(config["frozen_config"]).read_text())
    config["output"] = str(tmp_path)
    (tmp_path / "test_results.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        run_probe(config, "cpu")
    assert (tmp_path / "test_results.json").read_text() == "preserve"
    config["features"]["random_encoder_seeds"] = [42]
    with pytest.raises(ValueError, match="parameter budget"):
        validate_probe_config(config, frozen)
