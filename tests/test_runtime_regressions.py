"""Regression coverage for dataset ownership and inference handoff."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from run import inference_array
from src.ssunet.configs import DataConfig, SplitParams, SSUnetData
from src.ssunet.datasets import BinomDataset, PairedDataset


@pytest.mark.parametrize("virtual_size", [0, 4])
def test_loader_keeps_numpy_input_available_for_inference(virtual_size):
    """Reading batches must preserve the original arrays and photon counts."""
    primary = np.ones((5, 8, 8), dtype=np.float32)
    secondary = np.full_like(primary, 0.25)
    shared = SSUnetData(primary, secondary)
    dataset = BinomDataset(
        shared,
        DataConfig(xy_size=8, z_size=2, virtual_size=virtual_size),
        SplitParams(method="fixed", min_p=0.5, max_p=0.5, normalize_target=False),
    )

    target, noise, reference = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))

    assert shared.primary_data is primary
    assert shared.secondary_data is secondary
    np.testing.assert_array_equal(inference_array(shared.primary_data), primary)
    torch.testing.assert_close(target + noise, torch.ones_like(target))
    torch.testing.assert_close(reference, torch.full_like(reference, 0.25))


def test_paired_loader_keeps_reference_array_owned_by_caller():
    """Paired data must use the same non-mutating tensor conversion."""
    primary = np.ones((5, 8, 8), dtype=np.float32)
    secondary = np.full_like(primary, 0.25)
    shared = SSUnetData(primary, secondary)
    dataset = PairedDataset(shared, DataConfig(xy_size=8, z_size=2))

    observation, reference = dataset[0]

    assert shared.primary_data is primary
    assert shared.secondary_data is secondary
    torch.testing.assert_close(observation, torch.ones_like(observation))
    torch.testing.assert_close(reference, torch.full_like(reference, 0.25))


@pytest.mark.parametrize("random_crop", [True, False])
def test_crop_larger_than_image_has_clear_error(random_crop):
    """An oversized crop must fail before silently producing a smaller sample."""
    dataset = BinomDataset(
        SSUnetData(torch.ones(5, 8, 8)),
        DataConfig(xy_size=9, z_size=2, random_crop=random_crop),
        SplitParams(min_p=0.5, max_p=0.5),
    )

    with pytest.raises(ValueError, match="must fit image shape"):
        dataset[0]


def test_random_crop_can_include_last_image_row_and_column(monkeypatch):
    """The final valid crop must conserve the photons in the image corner."""
    monkeypatch.setattr("src.ssunet.datasets.singlevolume.randint", lambda high: high - 1)
    primary = torch.arange(5 * 8 * 8).reshape(5, 8, 8).float()
    dataset = BinomDataset(
        SSUnetData(primary),
        DataConfig(xy_size=4, z_size=2),
        SplitParams(min_p=0.5, max_p=0.5, normalize_target=False),
    )

    target, noise = dataset[0]

    torch.testing.assert_close(target + noise, primary[:2, -4:, -4:].unsqueeze(0))


def test_inference_handoff_accepts_tensor_requiring_gradients():
    """Inference accepts tensor inputs without NumPy dispatch or autograd errors."""
    tensor = torch.ones(2, 4, 4, dtype=torch.float64, requires_grad=True)

    result = inference_array(tensor)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.ones((2, 4, 4), dtype=np.float32))
