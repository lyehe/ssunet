# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSUnet (Self-Supervised U-Net) is a parametric self-supervised denoising framework for binary/photon-counting images. The core model is called **Bit2Bit**, a 3D U-Net built on PyTorch Lightning that uses binomial splitting for self-supervised training (no clean ground truth needed).

## Commands

```bash
# Training & inference pipeline
python run.py --config config.yml

# Tests (runs in CPU-only mode via env vars)
pytest tests
tox                    # full test suite across py310, lint, format

# Linting & formatting
ruff check .           # or: tox -e lint
ruff format --check .  # or: tox -e format
ruff check --fix .     # auto-fix lint issues
ruff format .          # auto-format

# Run a single test
pytest tests/test_bit2bit.py -v
```

## Architecture

### Source layout (`src/`)
- **`ssunet/models/bit2bit.py`** — Main Lightning module. Encoder-decoder U-Net with skip connections, configurable depth, block types ("tri"/"dual"/"LK"), gradient checkpointing, and partial convolutions.
- **`ssunet/modules/`** — Reusable building blocks: conv helpers (`conv333`, `conv555`...), down/up blocks (`DownConvTri3D`, `UpConvTri3D`), custom layers (`PartialConv3d`, a thin wrapper over torch-convnd's partial convolution; `PixelShuffle3d`, `SeparableConv3d`).
- **`ssunet/datasets/`** — Dataset classes. `BinomDataset` is the primary one—splits input via binomial sampling into target/noise pairs for self-supervised learning. Also: `BernoulliDataset`, `N2NSkipFrameDataset`, `PairedDataset`, `ValidationDataset`.
- **`ssunet/losses.py`** — Loss functions: MSE, L1, Photon, Photon2D. All support optional masking.
- **`ssunet/configs/`** — Dataclass-based configuration system. `MasterConfig` aggregates `ModelConfig`, `DataConfig`, `PathConfig`, `SplitParams`, `LoaderConfig`, `TrainConfig`. Loaded from YAML via `MasterConfig.from_config()`.
- **`ssunet/exceptions.py`** — Exception hierarchy: `SSUnetError` → `ConfigError`, `DataError`, `ModelError`, `InferenceError`.
- **`ssunet/constants.py`** — Global constants: `EPSILON`, default optimizer config, default metrics list.

### Tools (`src/tools/`)
- **`gpuinference.py`** — Patch-based GPU inference with adaptive patch sizing, overlap handling, and CUDA streams.
- **`cpuinference.py`** — CPU-based alternative.
- **`metrics.py`** — Image quality metrics (PSNR, SSIM, etc.).

### Pipeline flow (`run.py`)
1. Load YAML config → `MasterConfig`
2. Load data (HDF5/TIFF) → `BinomDataset` + `ValidationDataset`
3. Create DataLoaders
4. Init `Bit2Bit` model → train with Lightning
5. Run `gpu_patch_inference` → save output as TIFF

## Configuration

All settings live in `config.yml` with sections: `PATH`, `DATA`, `SPLIT`, `MODEL`, `LOADER`, `TRAIN`. Key model options: block_type, loss_function, up_mode, down_mode, partial_conv, gradient checkpointing.

## Code Style

- Ruff with line-length 100, target py310, Google docstring convention
- Pre-commit hooks enforce formatting and linting
- `numpy < 2` constraint is intentional
- Tests use `assert` freely (S101 ignored in tests/)
- Imports from `src.ssunet.*` and `src.tools.*` (pythonpath includes `src`)
