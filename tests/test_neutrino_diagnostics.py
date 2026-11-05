"""Verify diagnostic GT identities and loss-independent figure selection."""

import numpy as np
import pytest
import torch

from src.ssunet.models.neutrino_dense import DensePhotonReference
from src.ssunet.models.neutrino_precision import FullPrecisionPhotonReference
from src.ssunet.models.neutrino_variants import FinePhotonReference, PriorResidualPhotonReference
from src.ssunet.neutrino_evaluation import PhotonScoreRecords
from src.ssunet.photon_reference import photon_count_log_probabilities
from src.tools.neutrino_analytic_study import analytic_location_pdf, evaluate_oracle_model
from src.tools.neutrino_followup import ResponsePhotonDataset
from src.tools.neutrino_study_visualize import create_study_visualizations, select_cases


def test_zero_intensity_has_no_location_pdf():
    """Do not label a conditional prior as GT when no photons can exist."""
    with pytest.raises(ValueError, match="zero-intensity"):
        analytic_location_pdf(np.zeros((2, 3)), np.arange(5), np.zeros(3), 0, 0)
    pdf = analytic_location_pdf(np.zeros((2, 3)), np.arange(5), np.zeros(3), 0, 1)
    np.testing.assert_allclose(pdf, np.full((2, 4), 1 / 8))


def test_visualizations_cannot_open_an_unscored_reserved_test(tmp_path):
    """A request for GT figures must not bypass the separate locked test stage."""
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="completed locked test"):
        create_study_visualizations({"data": {"output": str(tmp_path)}}, output, "test")
    assert not output.exists()


def test_oracle_evaluation_has_zero_kl_for_exact_known_background():
    """Verify both expectation KL and realized held-out NLL against a known oracle."""

    class TinyDataset:
        positions_m = np.zeros((2, 3))
        edges_ns = np.linspace(0, 6400, 5)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            return {
                "input": torch.zeros((2, 4), dtype=torch.int64),
                "target": torch.tensor([[0, 1, 0, 0], [0, 0, 0, 2]]),
                "event_id": torch.tensor(0),
                "positions_m": torch.zeros((2, 3)),
                "live": torch.ones(2, dtype=torch.bool),
                "p": torch.tensor(0.5),
            }

    class ExactBackground(torch.nn.Module):
        def forward(self, counts, positions, live, p):
            del positions, live, p
            return torch.zeros_like(counts, dtype=torch.float32)

    class BackgroundBaseline:
        winner = "uniform"

        def probabilities(self, counts):
            return {"uniform": torch.ones_like(counts, dtype=torch.float32) / 8}

    config = {
        "generator": {
            "analytic": {"expected_signal_photons": [0], "expected_background_photons": 1}
        },
        "training": {"batch_size": 1},
    }
    values = evaluate_oracle_model(
        ExactBackground(), TinyDataset(), np.zeros((1, 3)), BackgroundBaseline(), config, "cpu"
    )
    np.testing.assert_allclose(values["model_kl"], 0, atol=1e-7)
    np.testing.assert_allclose(values["baseline_kl"], 0, atol=1e-12)
    np.testing.assert_allclose(values["model"], 3 * np.log(8), atol=1e-6)
    np.testing.assert_allclose(values["latent_oracle"], 3 * np.log(8), atol=1e-12)


def test_figure_examples_are_id_deterministic_and_include_empty_observations():
    """Case membership is independent of model quality and input row order."""
    records = PhotonScoreRecords(
        np.array([9, 2, 8, 3, 7, 4, 6, 5]),
        np.arange(8),
        np.array([0, 0, 1, 5, 8, 20, 50, 3]),
        np.array([0, 2, 1, 0, 3, 7, 4, 2]),
    )
    chosen = [(label, int(records.event_ids[index])) for label, index in select_cases(records)]
    permutation = np.arange(7, -1, -1)
    reordered = PhotonScoreRecords(
        *(
            getattr(records, key)[permutation]
            for key in ("event_ids", "configuration_ids", "retained_totals", "target_totals")
        )
    )
    assert chosen == [
        (label, int(reordered.event_ids[index])) for label, index in select_cases(reordered)
    ]
    assert chosen[:2] == [("Empty observation", 9), ("Empty retained input", 2)]


def test_prior_residual_contains_the_simple_mixture_at_initialization():
    """Verify the proposed learned correction starts exactly at its strong control."""
    prior = torch.tensor([[1, 2, 1, 2], [2, 1, 2, 1]], dtype=torch.float32) / 12
    model = PriorResidualPhotonReference(prior, torch.eye(4), fraction=0.25, channels=2)
    a = torch.tensor([[[0, 1, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]])
    live = torch.ones(2, dtype=torch.bool)
    logits = model(a, torch.zeros((2, 3)), live, torch.tensor([0.5, 0.5]))
    probability = photon_count_log_probabilities(logits, live).exp()
    expected = torch.stack([0.75 * a[0] + 0.25 * prior, prior])
    torch.testing.assert_close(probability, expected)
    loss = -(probability[0, 0, 1].log())
    loss.backward()
    assert model.correction_scale.grad is not None


def test_fine_temporal_variant_preserves_parameter_budget_and_sensor_equivariance():
    """The ablation changes temporal pooling, without adding a larger network."""
    torch.set_num_threads(1)
    fine, original = FinePhotonReference(2), DensePhotonReference(2)
    assert {k: v.shape for k, v in fine.state_dict().items()} == {
        k: v.shape for k, v in original.state_dict().items()
    }
    counts = torch.arange(32).reshape(1, 2, 16)
    positions = torch.tensor([[0, 0, 0], [0, 0, 20]], dtype=torch.float32)
    live, p = torch.ones(2, dtype=torch.bool), torch.tensor([0.5])
    actual = fine(counts, positions, live, p)
    reordered = fine(counts.flip(1), positions.flip(0), live.flip(0), p)
    torch.testing.assert_close(actual.flip(1), reordered)


def test_response_happens_before_thinning_and_uses_no_target_information():
    """Confirm conservation, deterministic response and actual dead-time effects."""

    class TinyBase:
        path, split, seed, p = "unused", "test", 42, 0.5
        indices, event_ids, configuration_ids = np.array([0]), np.array([9]), np.array([3])
        positions_m, edges_ns = np.zeros((2, 3)), np.arange(9) * 25.0
        live = np.ones(2, dtype=bool)

        def __len__(self):
            return 1

        def counts(self, index):
            del index
            return np.array([[2, 1, 1, 0, 2, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]]), 0

    settings = {
        "seed": 12,
        "efficiency_range": [0.5, 1.0],
        "timing_offset_ns": 50,
        "dead_time_ns": 100,
    }
    response = ResponsePhotonDataset(TinyBase(), "dead_time", settings)
    expected = np.array([[1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(response.counts(0)[0], expected)
    sample = response[0]
    np.testing.assert_array_equal((sample["input"] + sample["target"]).numpy(), expected)
    torch.testing.assert_close(sample["input"], response[0]["input"])


def test_precision_variant_keeps_fp32_projection_under_outer_autocast():
    """The numerical fix preserves the architecture and full-precision reference output."""
    model = FullPrecisionPhotonReference(2)
    reference = DensePhotonReference(2)
    reference.load_state_dict(model.state_dict())
    a = torch.arange(32).reshape(1, 2, 16)
    positions, live, p = torch.zeros((2, 3)), torch.ones(2, dtype=torch.bool), torch.tensor([0.5])
    expected = reference(a, positions, live, p)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(a, positions, live, p)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
