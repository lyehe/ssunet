"""Exercise the public training-to-inference command with synthetic photon data."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from tifffile import imread, imwrite


def test_cpu_pipeline_without_image_reference(tmp_path):
    """A CPU run trains, scores photons, saves a checkpoint and conserves output counts."""
    data = np.random.default_rng(17).binomial(1, 0.3, (4, 8, 8)).astype(np.uint8)
    imwrite(tmp_path / "photons.tif", data, photometric="minisblack")
    config = {
        "PATH": {"data_dir": str(tmp_path), "data_file": "photons.tif"},
        "DATA": {"xy_size": 8, "z_size": 2, "virtual_size": 2},
        "SPLIT": {"method": "fixed", "min_p": 0.5, "max_p": 0.5, "seed": 3},
        "MODEL": {
            "depth": 1,
            "start_filts": 4,
            "group_norm": 0,
            "sin_encoding": False,
            "partial_conv": False,
            "architecture_version": 2,
        },
        "LOADER": {
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "drop_last": False,
        },
        "TRAIN": {
            "default_root_dir": str(tmp_path / "runs"),
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "check_val_every_n_epoch": 1,
            "limit_val_batches": 1,
            "callbacks_handle_interrupt": False,
            "profiler": None,
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(  # noqa: S603 -- fixed local entry point, no shell
        [sys.executable, str(repository / "run.py"), "--config", str(path)],
        cwd=repository,
        env=os.environ | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoints = list((tmp_path / "runs").rglob("model.ckpt"))
    outputs = list((tmp_path / "runs").rglob("output.tif"))
    assert len(checkpoints) == len(outputs) == 1
    output = imread(outputs[0])
    assert output.shape == data.shape
    assert np.isfinite(output).all()
    assert (output >= 0).all()
    np.testing.assert_allclose(output.sum(), data.sum(), rtol=1e-5)
