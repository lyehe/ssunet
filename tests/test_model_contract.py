"""Distribution output, validation diagnostics and checkpoint integration."""

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ssunet.configs import ModelConfig, TrainConfig
from src.ssunet.losses import photon_loss, photon_nll
from src.ssunet.models import Bit2Bit


def small_model(**kwargs):
    """Build a small model for complete CPU training and validation steps."""
    options = {"depth": 1, "start_filts": 4, "group_norm": 0, "sin_encoding": False}
    return Bit2Bit(ModelConfig(**(options | kwargs)))


def test_predict_distribution_normalizes_each_sample():
    """The explicit prediction API returns distributions while forward retains logits."""
    model = small_model().eval()
    with torch.no_grad():
        probabilities = model.predict_distribution(torch.rand(2, 1, 2, 8, 8))
    torch.testing.assert_close(probabilities.sum(dim=(1, 2, 3, 4)), torch.ones(2))
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()


def test_framewise_distribution_ignores_frame_logit_offsets():
    """A 2D objective leaves an independent normalization constant in each frame."""
    model = small_model(loss_function="photon_2d")
    logits = torch.randn(2, 1, 3, 4, 4)
    model.forward = lambda _: logits
    expected = model.predict_distribution(logits)
    model.forward = lambda _: logits + torch.tensor([-100.0, 0.0, 100.0])[None, None, :, None, None]
    actual = model.predict_distribution(logits)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.sum(dim=(1, 3, 4)), torch.ones(2, 3))


@pytest.mark.parametrize("masked", [False, True])
def test_validation_uses_training_mask_and_photon_score(masked):
    """Validation scores observed held-out events with the configured mask."""
    model = small_model(masked=masked)
    logits = torch.linspace(-2, 2, 16).reshape(1, 1, 1, 4, 4)
    input_data = torch.zeros_like(logits)
    input_data[..., 0, 0] = 1
    target = torch.ones_like(logits)
    target[..., 0, 0] = 0
    recorded = {}
    model.forward = lambda _: logits
    model.log = lambda name, value, **kwargs: recorded.update({name: value})
    model.validation_step([target, input_data], 0)

    mask = (input_data < 1).float() if masked else None
    torch.testing.assert_close(recorded["val_loss"], photon_loss(logits, target, mask))
    name = "val_masked_photon_nll" if masked else "val_photon_nll"
    torch.testing.assert_close(recorded[name], photon_nll(logits, target, mask))


def test_reference_metrics_are_optional_and_distribution_based():
    """A reference diagnostic compares the center-frame probabilities and is shift invariant."""
    model = small_model()
    reference = torch.linspace(0.1, 0.9, 256).reshape(1, 1, 1, 16, 16)
    center = (reference * 0.7 + 0.05).log()
    output = torch.cat([torch.zeros_like(center), center, torch.ones_like(center)], dim=2)
    recorded = {}
    model.log = lambda name, value, **kwargs: recorded.update({name: value.detach().clone()})
    model._tb_val_log(torch.tensor(0.0), output, reference, reference, 0)
    assert "val_psnr" not in recorded

    model.config.log_reference_metrics = True
    model._tb_val_log(torch.tensor(0.0), output, reference, reference, 0)
    original_psnr = recorded["val_psnr"]
    original_ssim = recorded["val_ssim"]
    shifted = output + 4
    shifted[:, :, 0] = 1000
    model._log_metrics(shifted, reference, 0)
    torch.testing.assert_close(recorded["val_psnr"], original_psnr, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(recorded["val_ssim"], original_ssim, rtol=1e-5, atol=1e-5)
    assert original_psnr > 20


def test_framewise_test_score_matches_validation_reduction():
    """Multiple temporal frames use the same framewise photon score at test time."""
    model = small_model(masked=True, loss_function="photon_2d")
    logits = torch.randn(1, 1, 3, 4, 4)
    input_data = torch.zeros_like(logits)
    input_data[..., 0, 0] = 1
    target = 1 - input_data
    recorded = {}
    model.forward = lambda _: logits
    model.log = lambda name, value, **kwargs: recorded.update({name: value})
    model.test_step([target, input_data], 0)
    expected = photon_nll(logits, target, (input_data < 1).float(), dim=(-1, -2, -4))
    torch.testing.assert_close(recorded["test_masked_photon_nll"], expected)


@pytest.mark.parametrize("save_at_end", [False, True])
def test_cpu_fit_respects_checkpoint_configuration(tmp_path, save_at_end):
    """Training needs neither references nor image logging and writes only requested checkpoints."""
    config = TrainConfig(
        default_root_dir=tmp_path,
        callbacks_model_checkpoint=False,
        callbacks_learning_rate_monitor=False,
        callbacks_handle_interrupt=False,
        callbacks_save_on_train_end=save_at_end,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        default_root_dir=tmp_path,
        logger=False,
        callbacks=config.callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    model = small_model(masked=False)
    data = torch.ones(1, 1, 2, 8, 8)
    trainer.fit(model, train_dataloaders=DataLoader(TensorDataset(data, data), batch_size=1))
    names = sorted(path.name for path in tmp_path.iterdir() if path.is_file())
    assert names == (["final_model.ckpt"] if save_at_end else [])

    trainer.save_checkpoint(tmp_path / "explicit.ckpt")
    restored = Bit2Bit.load_from_checkpoint(tmp_path / "explicit.ckpt", weights_only=False).eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(data), model(data), rtol=0, atol=0)
