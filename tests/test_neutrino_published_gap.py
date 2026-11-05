"""Check the released GAP network, detector mapping and scheduler-safe resumption."""

import copy
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ssunet.models.neutrino_published_gap import PublishedGAPReference
from src.ssunet.photon_methods import paper_photon_loss
from src.ssunet.photon_reference import photon_count_log_probabilities


def positions():
    """Return a non-involutive physical-to-array sensor ordering."""
    return np.array([[20, 0, 0], [0, 0, 0], [30, 0, 0], [10, 0, 0]], dtype=np.float32)


def test_vendor_is_the_unmodified_pinned_release():
    """The comparison uses the released network bytes, not an approximate small U-Net."""
    module = importlib.import_module("src.ssunet.vendor.gap.GAP_UNET_ResBlock")
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert actual == "3568742971dec5e83e5ebabfe30de1fb331c75bad47264fa439e58d8c6fca9ed"


def test_paper_model_has_the_full_published_capacity():
    """The six-level, 28-initial-channel backbone retains every original parameter."""
    torch.set_num_threads(1)
    model = PublishedGAPReference(positions())
    assert model.network.depth == 6
    assert model.network.start_filts == 28
    assert model.network.levels == 10
    assert sum(parameter.numel() for parameter in model.parameters()) == 33443425


def test_adapter_forward_and_gradients_match_the_released_network():
    """Only physical row ordering is added around the unchanged network computation."""
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = PublishedGAPReference(positions(), depth=2, start_filts=2, levels=3)
    original = copy.deepcopy(model.network)
    counts = torch.arange(48).reshape(2, 4, 6) % 5
    live = torch.ones(4, dtype=torch.bool)
    actual = model(counts, torch.tensor(positions()), live, torch.tensor([0.5, 0.5]))
    expected = original(counts[:, [1, 3, 0, 2], None].transpose(1, 2).float())[:, 0]
    expected = expected[:, [2, 0, 3, 1]]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for a, b in zip(model.network.parameters(), original.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)


def test_padding_permutation_and_dead_sensors_preserve_physical_probabilities():
    """Padded rows/time never become probability bins; sensor permutation preserves meaning."""
    torch.set_num_threads(1)
    geometry = positions()[:3]
    model = PublishedGAPReference(geometry, depth=2, start_filts=2, levels=3)
    counts = torch.arange(15).reshape(1, 3, 5) % 4
    live = torch.tensor([True, False, True])
    arguments = torch.tensor(geometry), live, torch.tensor([0.5])
    logits = model(counts, *arguments)
    changed = counts.clone()
    changed[:, 1] = 100000
    torch.testing.assert_close(model(changed, *arguments), logits)
    order = torch.tensor([2, 0, 1])
    permuted = model(counts[:, order], arguments[0][order], live[order], arguments[2])
    torch.testing.assert_close(permuted, logits[:, order])
    assert logits.shape == counts.shape
    q = photon_count_log_probabilities(logits, live).exp()
    torch.testing.assert_close(q.sum((1, 2)), torch.ones(1))
    assert torch.count_nonzero(q[:, 1]) == 0
    with pytest.raises(ValueError, match="each configured detector"):
        model(counts, arguments[0][[0, 0, 2]], live, arguments[2])


def test_stable_loss_matches_the_released_gap_photon_loss_and_gradients():
    """Physical-support mean-normalized target loss matches the source expression exactly."""
    model = PublishedGAPReference(positions()[:3], depth=2, start_filts=2, levels=3)
    torch.manual_seed(317)
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, 5, logits.shape)
    normalized = target.double() / (target.double().mean((1, 2), keepdim=True) + 1e-8)
    expected = model.network.photonLoss(logits[:, None], normalized[:, None])
    actual = paper_photon_loss(
        logits, target, torch.ones(3, dtype=torch.bool), target, observation_kind="photon_counts"
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        rtol=1e-12,
        atol=1e-12,
    )


def test_paper_scheduler_and_optimizer_resume_the_same_training_trajectory(tmp_path, monkeypatch):
    """A mid-epoch restart preserves learning-rate reductions and final parameter values."""
    pytest.importorskip("torch_convnd")
    from src.tools import neutrino_published_gap as study

    torch.set_num_threads(1)

    class ToyData:
        positions_m = positions()

        def __len__(self):
            return 4

        def __getitem__(self, key):
            epoch, index = key
            inputs = torch.zeros(4, 8, dtype=torch.int64)
            inputs[0, (epoch + index) % 8] = 1
            target = torch.zeros_like(inputs)
            target[0, (epoch + index + 1) % 8] = 1
            return {
                "input": inputs,
                "target": target,
                "positions_m": torch.tensor(self.positions_m),
                "live": torch.ones(4, dtype=torch.bool),
                "p": torch.tensor(0.5),
            }

    def fake_curve(*args):
        step = args[-1]
        return {"step": step, "retained_1_to_10": 1.0, "event_mean_low": 1.0}, np.ones(2)

    monkeypatch.setattr(study.retry, "curve_point", fake_curve)
    monkeypatch.setattr(study, "normalized_validation_loss", lambda *args: 1.0)
    monkeypatch.setattr(study.retry.reference, "model_event_nll", lambda *args: np.ones(2))
    config = {
        "observation_kind": "photon_counts",
        "training": {
            "steps": 6,
            "batch_size": 2,
            "max_training_seconds_per_seed": 120,
            "max_gpu_allocated_gib": 12,
        },
        "evaluation": {"steps": list(range(7))},
    }
    settings = {
        "name": "published_test",
        "depth": 2,
        "start_filts": 2,
        "levels": 3,
        "gradient_clip": 0.5,
        "optimizer": {
            "kind": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 0,
            "scheduler_patience": 0,
        },
    }
    device = torch.device("cpu")
    training = ToyData()
    continuous = tmp_path / "continuous"
    restarted = tmp_path / "restarted"
    study.train_seed(config, settings, 18, training, None, None, continuous, device)
    original_step = study.retry.training_step
    calls = 0

    def interrupt_before_fourth_step(*args):
        nonlocal calls
        if calls == 3:
            message = "Test interruption"
            raise RuntimeError(message)
        calls += 1
        return original_step(*args)

    monkeypatch.setattr(study.retry, "training_step", interrupt_before_fourth_step)
    with pytest.raises(RuntimeError, match="Test interruption"):
        study.train_seed(config, settings, 18, training, None, None, restarted, device)
    monkeypatch.setattr(study.retry, "training_step", original_step)
    study.train_seed(config, settings, 18, training, None, None, restarted, device)
    checkpoints = [
        torch.load(root / "published_test/seed_18/last.pt", weights_only=True)
        for root in (continuous, restarted)
    ]
    for key, value in checkpoints[0]["state_dict"].items():
        torch.testing.assert_close(value, checkpoints[1]["state_dict"][key], rtol=0, atol=0)
    assert (
        checkpoints[0]["state"]["scheduler_state_dict"]
        == checkpoints[1]["state"]["scheduler_state_dict"]
    )
    assert checkpoints[1]["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(2.5e-5)


def test_published_event_intervals_do_not_weight_bright_targets_more():
    """A hundred-photon target still contributes one event in the equal-event contrast."""
    pytest.importorskip("torch_convnd")
    from src.ssunet.neutrino_evaluation import FrozenPhotonView
    from src.tools.neutrino_published_gap_report import event_intervals

    inputs = torch.tensor([[[1]], [[2]], [[3]]])
    target = torch.tensor([[[1]], [[100]], [[0]]])
    view = FrozenPhotonView(
        inputs=inputs,
        targets=target,
        positions_m=torch.zeros(1, 3),
        live=torch.ones(1, dtype=torch.bool),
        p=0.5,
        event_ids=np.arange(3),
        configuration_ids=np.arange(3),
    )
    saved = {
        "conv4d": np.array([[1.0, 200.0, 0.0]] * 3),
        "published_gap_paper": np.array([[2.0, 100.0, 0.0]] * 3),
        "published_gap_matched": np.array([[0.0, 100.0, 0.0]] * 3),
    }
    config = {
        "models": [{"name": "published_gap_paper"}, {"name": "published_gap_matched"}],
        "evaluation": {"bootstrap_seed": 17, "bootstrap_draws": 1000},
    }
    result = event_intervals(config, view, saved)
    assert result["published_gap_paper"]["all"][
        "model_minus_control_mean_event_cross_entropy"
    ] == pytest.approx(0.0)
    assert result["published_gap_matched"]["all"][
        "model_minus_control_mean_event_cross_entropy"
    ] == pytest.approx(1.0)
    assert result["published_gap_paper"]["retained_0"] is None
