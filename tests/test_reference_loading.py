"""Optional evaluation references retain explicit dtype and scale semantics."""

import numpy as np
import pytest
from tifffile import imwrite

from src.ssunet.configs import PathConfig


def _config(tmp_path, reference=None, **kwargs):
    """Create a data file and an optional separately configured reference."""
    data = np.ones((3, 4, 4), dtype=np.uint8)
    imwrite(tmp_path / "data.tif", data, photometric="minisblack")
    options = {"data_dir": tmp_path, "data_file": "data.tif", **kwargs}
    if reference is not None:
        imwrite(tmp_path / "reference.tif", reference, photometric="minisblack")
        options.update(ground_truth_dir=tmp_path, ground_truth_file="reference.tif")
    return PathConfig(**options)


@pytest.mark.parametrize(
    "method", ["load_data_and_ground_truth", "load_reference_and_ground_truth"]
)
def test_missing_reference_does_not_label_noisy_data_as_ground_truth(tmp_path, method):
    """Self-supervised loading must work without any evaluation reference."""
    config = _config(tmp_path)

    loaded = getattr(config, method)()

    assert loaded.primary_data.shape == (3, 4, 4)
    assert loaded.secondary_data is None


@pytest.mark.parametrize(
    ("dtype", "maximum", "expected"),
    [(np.float32, 1, 1.0), (np.uint8, 255, 1.0), (np.uint16, 255, 255 / 65535)],
)
def test_reference_normalization_uses_original_file_dtype(tmp_path, dtype, maximum, expected):
    """Dark uint16 images keep their sensor range after the float32 load conversion."""
    reference = np.full((3, 4, 4), maximum, dtype=dtype)
    config = _config(tmp_path, reference)

    loaded = config.load_reference_and_ground_truth()

    assert loaded.secondary_data is not None
    assert loaded.secondary_data.dtype == np.float32
    np.testing.assert_allclose(loaded.secondary_data, expected)


def test_float_reference_can_declare_its_intensity_scale(tmp_path):
    """An explicit intensity scale supports float images in physical units."""
    reference = np.full((3, 4, 4), 12.5, dtype=np.float32)
    config = _config(tmp_path, reference, ground_truth_scale=25)

    loaded = config.load_reference_and_ground_truth()

    np.testing.assert_allclose(loaded.secondary_data, 0.5)


def test_float_reference_with_unknown_range_requests_explicit_scale(tmp_path):
    """Float reference bit depth must not be inferred from its brightest pixel."""
    config = _config(tmp_path, np.full((3, 4, 4), 255, dtype=np.float32))

    with pytest.raises(ValueError, match="require ground_truth_scale"):
        config.load_reference_and_ground_truth()


def test_zero_file_index_resolves_optional_reference(tmp_path):
    """File index zero is a valid explicit reference selection."""
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    imwrite(
        reference_dir / "only.tif",
        np.full((3, 4, 4), 255, dtype=np.uint8),
        photometric="minisblack",
    )
    config = _config(tmp_path, ground_truth_dir=reference_dir, ground_truth_file=0)

    loaded = config.load_reference_and_ground_truth()

    np.testing.assert_array_equal(loaded.secondary_data, np.ones((3, 4, 4)))
