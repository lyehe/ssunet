"""Verify equal effective batches, identical photon draws and deterministic resume."""

import copy
import json

import numpy as np
import pytest
import torch

from src.tools import confocal_train as training


def _config(tmp_path):
    config = training.load_config("config/confocal_gap4d_features32_batch4.yml")
    config["output"] = str(tmp_path / "training")
    config["training"].update(patch_tzyx=[2, 2, 4, 4], steps=4, checkpoint_every_steps=1)
    config["evaluation"].update(every_steps=2, patches=2)
    return config


def test_batch_regrouping_keeps_every_training_location_and_photon_draw_identical(tmp_path):
    """Batch 2 x accumulation 2 matches batch 1 x accumulation 4, example by example."""
    config = _config(tmp_path)
    counts = np.arange(4 * 4 * 8 * 8, dtype=np.uint8).reshape(4, 4, 8, 8) % 7
    four = training.resolve_model_config(config, "conv4d")
    three = training.resolve_model_config(config, "conv3d")
    for step in [1, 2, 7]:
        big = [training.training_batch(counts, four, step, micro) for micro in range(2)]
        small = [training.training_batch(counts, three, step, micro) for micro in range(4)]
        for side in (0, 1):
            torch.testing.assert_close(
                torch.cat([x[side] for x in big]),
                torch.cat([x[side] for x in small]),
                rtol=0,
                atol=0,
            )


def test_effective_batch_override_cannot_change_training_example_budget(tmp_path):
    """Reject a control with fewer photons/examples per optimizer update."""
    config = _config(tmp_path)
    config["training"]["model_batching"]["conv3d"]["accumulation_steps"] = 3
    with pytest.raises(ValueError, match="effective batch"):
        training.resolve_model_config(config, "conv3d")


def test_microbatch_regrouping_preserves_loss_and_parameter_gradients(tmp_path):
    """The shared per-patch GAP mean is averaged equally before one clipping/update."""

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
            self.bias = torch.nn.Parameter(
                torch.linspace(-0.2, 0.4, 64, dtype=torch.float64).reshape(1, 1, 2, 2, 4, 4)
            )

        def forward(self, inputs):
            return self.scale * inputs.double() + self.bias

    config = _config(tmp_path)
    config["training"]["gradient_clip"] = 1000
    counts = np.arange(4 * 4 * 8 * 8, dtype=np.uint16).reshape(4, 4, 8, 8) % 7
    models = [Tiny(), Tiny()]
    losses = []
    for model, name in zip(models, ["conv4d", "conv3d"], strict=True):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss, _ = training._optimizer_step(
            model,
            optimizer,
            counts,
            training.resolve_model_config(config, name),
            torch.device("cpu"),
            1,
        )
        losses.append(loss)
    assert losses[0] == pytest.approx(losses[1], rel=1e-12, abs=1e-12)
    for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_interrupted_training_restores_full_optimizer_trajectory(tmp_path, monkeypatch):
    """Resume a real small U-Net at an optimizer boundary without repeating patches."""
    torch.set_num_threads(1)
    config = _config(tmp_path)
    config["training"].update(batch_size=1, accumulation_steps=1, model_batching={})
    settings = {
        "name": "conv3d",
        "dimensions": 3,
        "base_channels": 2,
        "depth": 2,
        "encoding_levels": 3,
    }
    config["models"] = [settings]
    values = (np.arange(4 * 4 * 8 * 8) % 5).reshape(4, 4, 8, 8).astype(np.uint8)
    counts = {"train": values, "validation": values}
    manifests = {"test_fixture": True}
    complete = copy.deepcopy(config)
    complete["output"] = str(tmp_path / "complete")
    training.run_model(complete, settings, counts, manifests, torch.device("cpu"), smoke=False)
    original = training._optimizer_step

    def interrupted(*args):
        if args[-1] == 3:
            message = "simulated interruption"
            raise InterruptedError(message)
        return original(*args)

    monkeypatch.setattr(training, "_optimizer_step", interrupted)
    with pytest.raises(InterruptedError):
        training.run_model(config, settings, counts, manifests, torch.device("cpu"), smoke=False)
    monkeypatch.setattr(training, "_optimizer_step", original)
    training.run_model(
        config, settings, counts, manifests, torch.device("cpu"), smoke=False, resume=True
    )
    expected = torch.load(tmp_path / "complete/pilot/conv3d/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "training/pilot/conv3d/last.pt", weights_only=True)
    assert actual["step"] == 4
    for name, value in expected["model_state_dict"].items():
        torch.testing.assert_close(value, actual["model_state_dict"][name], rtol=0, atol=0)
    assert expected["state"]["losses"] == actual["state"]["losses"]
    assert expected["state"]["norms"] == actual["state"]["norms"]
    for key, state in expected["optimizer_state_dict"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, actual["optimizer_state_dict"]["state"][key][field], rtol=0, atol=0
            )
    result = json.loads((tmp_path / "training/pilot/conv3d/result.json").read_text())
    assert result["effective_batch_size"] == 1
    assert result["logit_normalization"] == "barron2026"
