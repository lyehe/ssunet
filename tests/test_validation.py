"""Reproducible photon-only validation with optional external references."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.ssunet.configs import DataConfig, LoaderConfig, SplitParams, SSUnetData
from src.ssunet.datasets import ValidationDataset
from src.ssunet.datasets.spad import SPADDataModule, SPADValDataset


@pytest.mark.parametrize("count", [1, 4])
def test_validation_holds_out_photons_without_a_clean_reference(count):
    """Input and target conserve the observed events instead of copying one another."""
    observed = torch.full((4, 8, 8), float(count))
    config = DataConfig(xy_size=8, z_size=2, random_crop=False, normalize_target=False)
    target, input_photons = ValidationDataset(SSUnetData(observed), config)[0]

    torch.testing.assert_close(target + input_photons, observed[:2].unsqueeze(0))
    assert not torch.equal(target, input_photons)
    assert torch.count_nonzero(target) > 0
    assert torch.count_nonzero(input_photons) > 0
    if count == 1:
        assert torch.count_nonzero(target * input_photons) == 0


def test_validation_sampling_is_repeatable_without_changing_global_rngs():
    """Revisiting a sample does not change its split or the training RNG streams."""
    observed = torch.ones(8, 12, 12)
    config = DataConfig(xy_size=8, z_size=2, virtual_size=4, augments=True)
    dataset = ValidationDataset(SSUnetData(observed), config)
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    first = dataset[2]
    _ = dataset[1]
    repeated = dataset[2]

    for first_item, repeated_item in zip(first, repeated, strict=True):
        torch.testing.assert_close(first_item, repeated_item)
    assert torch.equal(torch.get_rng_state(), torch_state)
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]


def test_validation_reference_is_separate_from_the_photon_target():
    """Changing a supplied image reference leaves input and held-out events intact."""
    observed = torch.ones(4, 8, 8)
    config = DataConfig(xy_size=8, z_size=2, random_crop=False, normalize_target=False)
    first = ValidationDataset(SSUnetData(observed, torch.zeros_like(observed)), config)[0]
    second = ValidationDataset(SSUnetData(observed, torch.ones_like(observed)), config)[0]

    assert len(first) == 3
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert torch.count_nonzero(first[2]) == 0
    assert torch.all(second[2] == 1)


def test_validation_uses_supplied_split_policy_and_target_normalization():
    """Validation follows configured photon splitting while keeping each target normalized."""
    observed = torch.ones(4, 16, 16)
    config = DataConfig(xy_size=16, z_size=2, random_crop=False, normalize_target=False)
    params = SplitParams(method="fixed", min_p=0.8, max_p=0.8, normalize_target=True)
    dataset = ValidationDataset(SSUnetData(observed), config, split_params=params)
    target, input_photons = dataset[0]

    torch.testing.assert_close(target.mean(), torch.tensor(1.0))
    assert input_photons.mean() > 0.7
    assert config.validation_config.normalize_target == config.normalize_target


def test_validation_pairs_are_stable_across_loader_workers():
    """Worker scheduling cannot change validation samples used for model selection."""
    observed = torch.ones(4, 8, 8)
    config = DataConfig(xy_size=8, z_size=2, random_crop=False, normalize_target=False)
    dataset = ValidationDataset(SSUnetData(observed), config)
    expected = list(DataLoader(dataset, batch_size=1, num_workers=0))
    actual = list(DataLoader(dataset, batch_size=1, num_workers=2))

    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        for expected_item, actual_item in zip(expected_batch, actual_batch, strict=True):
            torch.testing.assert_close(expected_item, actual_item)


@pytest.mark.parametrize("with_reference", [False, True])
def test_spad_validation_is_deterministic_and_reference_optional(tmp_path, with_reference):
    """Packed validation data is sufficient to score photon predictions reproducibly."""
    binary = np.ones((8, 8, 8, 3), dtype=np.uint8)
    np.save(tmp_path / "000000.npy", np.packbits(binary, axis=2))
    if with_reference:
        Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8)).save(tmp_path / "000000.png")
    config = DataConfig(xy_size=4, z_size=4, normalize_target=False)
    params = SplitParams(method="fixed", min_p=0.5, max_p=0.5, normalize_target=False)
    loaders = LoaderConfig(batch_size=1, num_workers=0, persistent_workers=False)
    dm = SPADDataModule(tmp_path, config, params, loaders)
    dm.setup("fit")
    batch = next(iter(dm.val_dataloader()[0]))
    state = torch.get_rng_state().clone()
    first = dm.on_after_batch_transfer(batch, 0)
    second = dm.on_after_batch_transfer(batch, 0)

    assert len(SPADValDataset(tmp_path, z_size=4)) == 1
    assert len(first) == (3 if with_reference else 2)
    torch.testing.assert_close(first[0] + first[1], torch.ones(1, 3, 4, 4, 4))
    assert torch.count_nonzero(first[0] * first[1]) == 0
    for first_item, second_item in zip(first, second, strict=True):
        torch.testing.assert_close(first_item, second_item)
    assert torch.equal(torch.get_rng_state(), state)
    if with_reference:
        torch.testing.assert_close(first[2], torch.full((1, 3, 1, 4, 4), 64 / 255))
