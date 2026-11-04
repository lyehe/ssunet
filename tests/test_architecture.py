"""Architecture options and legacy checkpoint compatibility."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch_convnd

from src.ssunet.checkpoints import checkpoint_config_imports
from src.ssunet.configs import ModelConfig
from src.ssunet.models import Bit2Bit
from src.ssunet.modules.partialconv import PartialConv2d, PartialConv3d


def small_config(**kwargs):
    """Create a small model with the same block construction as production."""
    options = {
        "depth": 2,
        "start_filts": 8,
        "group_norm": 2,
        "z_conv_stage": 1,
        "sin_encoding": False,
    }
    return ModelConfig(**(options | kwargs))


@pytest.mark.parametrize("block", ["tri", "dual", "LK"])
@pytest.mark.parametrize("merge", ["concat", "add"])
def test_configured_architecture_forward_backward(block, merge):
    """Configured blocks honor the options and propagate finite gradients."""
    config = small_config(
        architecture_version=2, partial_conv=True, merge_mode=merge, block_type=block
    )
    model = Bit2Bit(config)
    data = torch.rand(1, 1, 2, 8, 8)
    output = model(data)
    output.square().mean().backward()

    assert output.shape == data.shape
    assert any(isinstance(layer, PartialConv3d) for layer in model.modules())
    assert all(layer.merge_mode == merge for layer in model.up_convs)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_partial_layers_use_torch_convnd_and_the_model_initialization():
    """Partial layers come from torch-convnd and get the model's Xavier initialization."""
    model = Bit2Bit(small_config(architecture_version=2, partial_conv=True))
    layers = [layer for layer in model.modules() if isinstance(layer, PartialConv3d)]

    assert layers
    assert all(isinstance(layer, torch_convnd.PartialConv3d) for layer in layers)
    assert all(layer.multi_channel for layer in layers)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in layers)


@pytest.mark.parametrize("block", ["tri", "dual", "LK"])
def test_configured_architecture_without_skip(block):
    """Disabling an encoder skip keeps decoder channel counts consistent."""
    model = Bit2Bit(small_config(architecture_version=2, skip_depth=1, block_type=block))
    data = torch.rand(1, 1, 2, 8, 8)
    assert model(data).shape == data.shape


def test_versionless_checkpoint_retains_historical_predictions(tmp_path):
    """Old ignored flags cannot silently change predictions after a checkpoint load."""
    baseline = Bit2Bit(small_config(partial_conv=False, merge_mode="concat")).eval()
    data = torch.randn(1, 1, 2, 8, 8)
    with torch.no_grad():
        expected = baseline(data)
    old_config = small_config(partial_conv=True, merge_mode="add")
    vars(old_config).pop("architecture_version")
    path = tmp_path / "versionless.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "state_dict": baseline.state_dict(),
            "hyper_parameters": {"config": old_config},
        },
        path,
    )

    restored = Bit2Bit.load_from_checkpoint(path, map_location="cpu", weights_only=False).eval()
    with torch.no_grad():
        actual = restored(data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.architecture_version == 1
    assert not any(isinstance(layer, PartialConv3d) for layer in restored.modules())

    # An isolated process sees only the installed-style source root, not the
    # repository's src namespace that can otherwise hide old pickle paths.
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        import torch
        from ssunet.models import Bit2Bit
        assert 'src' not in sys.modules
        allowed_before = set(torch.serialization.get_safe_globals())
        for weights_only in (False, True):
            loaded = Bit2Bit.load_from_checkpoint(
                sys.argv[2], map_location='cpu', weights_only=weights_only
            )
            assert loaded.architecture_version == 1
            assert 'src' not in sys.modules
            assert set(torch.serialization.get_safe_globals()) == allowed_before
        try:
            Bit2Bit.load_from_checkpoint(sys.argv[2] + '.missing')
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing checkpoint unexpectedly loaded')
        assert 'src' not in sys.modules
        assert set(torch.serialization.get_safe_globals()) == allowed_before
        """
    )
    result = subprocess.run(  # noqa: S603 -- fixed Python harness, no shell
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(__file__).resolve().parents[1] / "src"),
            str(path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_checkpoint_allowlist_preserves_existing_and_nested_contexts():
    """A scoped compatibility entry must not remove an existing caller allowance."""
    before = set(torch.serialization.get_safe_globals())
    shared = (ModelConfig, "src.ssunet.configs.model_config.ModelConfig")
    with torch.serialization.safe_globals([shared]):
        caller = set(torch.serialization.get_safe_globals())
        with checkpoint_config_imports(ModelConfig):
            outer = set(torch.serialization.get_safe_globals())
            with checkpoint_config_imports(ModelConfig):
                assert shared in torch.serialization.get_safe_globals()
            assert set(torch.serialization.get_safe_globals()) == outer
        assert set(torch.serialization.get_safe_globals()) == caller
    assert set(torch.serialization.get_safe_globals()) == before


def test_configured_checkpoint_round_trip_and_mismatch(tmp_path):
    """Versioned checkpoints load in the recorded architecture and reject incompatible overrides."""
    config = small_config(architecture_version=2, partial_conv=True, merge_mode="add")
    model = Bit2Bit(config).eval()
    data = torch.rand(1, 1, 2, 8, 8)
    with torch.no_grad():
        expected = model(data)
    checkpoint = {
        "pytorch-lightning_version": pl.__version__,
        "state_dict": model.state_dict(),
        "hyper_parameters": {"config": config},
    }
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "configured.ckpt"
    torch.save(checkpoint, path)

    restored = Bit2Bit.load_from_checkpoint(path, weights_only=False).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(data), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="architecture version"):
        Bit2Bit.load_from_checkpoint(path, config=small_config(), weights_only=False)
    with pytest.raises(ValueError, match="architecture options"):
        Bit2Bit.load_from_checkpoint(
            path,
            config=small_config(architecture_version=2, partial_conv=False, merge_mode="add"),
            weights_only=False,
        )


@pytest.mark.parametrize(
    ("layer_class", "shape", "kernel_volume"),
    [(PartialConv2d, (1, 1, 4, 4), 9), (PartialConv3d, (1, 1, 2, 4, 4), 27)],
)
def test_partial_padding_mask_reset_and_checkpoint_buffers(layer_class, shape, kernel_volume):
    """Padding correction and explicit masks work across calls and precision changes."""
    layer = layer_class(1, 1, 3, padding=1)
    with torch.no_grad():
        layer.weight.fill_(1)
        layer.bias.fill_(2)
    data = torch.ones(shape)
    expected = torch.full(shape, kernel_volume + 2.0)

    output = layer(data)
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(layer(data, torch.zeros_like(data)), torch.zeros_like(data))
    torch.testing.assert_close(layer(data), expected)
    assert set(layer.state_dict()) == {"weight", "bias"}

    layer = layer.double()
    torch.testing.assert_close(layer(data.double()), expected.double())
    layer.return_mask = True
    output, mask = layer(data.double())
    torch.testing.assert_close(output, expected.double())
    torch.testing.assert_close(mask, torch.ones_like(data, dtype=torch.float64))
