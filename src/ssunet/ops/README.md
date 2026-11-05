# Native dense ConvNd

This optional addon implements **full, unrestricted N-dimensional convolution**
(cross-correlation, following PyTorch's naming convention). `Conv4d` is a wrapper
around the same `ConvNd` operator. The weights are not separable or factorized.
The implementation accepts 1-62 spatial dimensions, matching PyTorch's maximum
of 64 tensor axes after including batch and channels.

```python
import torch
from src.ssunet.ops import Conv4d, ConvNd, load_native_convnd

# Optional: compile explicitly before the first forward pass or torch.compile.
load_native_convnd(with_cuda=True)

layer = Conv4d(4, 8, kernel_size=(3, 3, 3, 5), padding=(1, 1, 1, 2)).cuda()
x = torch.randn(1, 4, 8, 8, 8, 16, device="cuda", requires_grad=True)
y = layer(x)  # [batch, channels, x, y, z, time]
y.square().mean().backward()

# The same backend, with five spatial axes:
layer5 = ConvNd(4, 8, kernel_size=3, dimensions=5, padding=1)
```

In an installed package, use `from ssunet.ops import ...`. Importing the package
does not compile anything. The first convolution builds into PyTorch's extension
cache. This does not replace any existing Bit2Bit model or resume experiments.

## Build

Install the small build dependency in the active environment:

```powershell
uv pip install --python .venv\Scripts\python.exe ninja
```

A C++17 compiler is required. A CUDA build additionally needs the CUDA toolkit
and `nvcc` compatible with the installed PyTorch build. On Windows, the loader
activates the installed Visual Studio compiler environment temporarily. On Linux,
use the normal compiler toolchain. Set `TORCH_CUDA_ARCH_LIST` to control GPU
targets, `MAX_JOBS` to control build parallelism, or `TORCH_EXTENSIONS_DIR` to
choose the cache location. Defaults are the visible GPU architectures and two
build jobs. Linux has not been verified locally.

`load_native_convnd(with_cuda=False)` explicitly builds a CPU-only addon; changing
from a loaded CPU-only addon to CUDA requires a new Python process. Compilation
errors propagate; there is no silent fallback to a different convolution.

## Implementation and scope

Native C++/CUDA kernels gather N-dimensional patches. Grouped matrix products
use PyTorch's GEMM backend. The backward pass computes weight derivatives with
GEMM and accumulates input derivatives with native N-dimensional kernels. It
recomputes patches instead of saving an expanded patch tensor for backward.
This is an explicit patch-extraction implementation, not a single fused direct
convolution kernel and not a cuDNN-quality optimized convolution implementation.

Supported features:

- Batched channels-first tensors, CPU/CUDA, FP32/FP64.
- Independent weights on every spatial axis, bias, groups and depthwise groups.
- Per-axis stride, dilation and symmetric zero padding.
- First-order gradients, noncontiguous inputs/weights, empty batches.
- Fake-tensor registration and AOTAutograd integration; preload before compiling.
- CUDA's current device/stream; deterministic input-gradient gathering when
  `torch.use_deterministic_algorithms(True)` is set. The default uses atomic
  input-gradient accumulation, so rounding can differ between runs.

Not implemented: FP16/BF16/autocast, higher-order gradients, transposed convolution,
string padding (`"same"`), asymmetric or nonzero padding modes, vmap rules, sparse
convolution, or a complete 4D U-Net. Those require separate work and validation.

`workspace_bytes` defaults to 64 MiB and budgets two patch buffers plus an output
tile. This is **not a cap on total process/GPU memory**: inputs, outputs,
contiguous copies, parameter gradients and GEMM/weight-gradient temporaries are
additional. Smaller tiles reduce patch storage but increase launch overhead;
deterministic gathering revisits the input for each tile. Channel count and
kernel volume must permit at least one output position within the budget.

For each axis, output size is
`floor((input + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1`.
Kernel volume grows as the product of kernel sizes, so adding dimensions remains
expensive even with native code. Benchmark the intended geometry before choosing
this over batched lower-dimensional convolution or a factorized architecture.

## Validation and benchmark

The optional tests invoke a compiler, so they require explicit opt-in:

```powershell
$env:SSUNET_TEST_NATIVE = "1"
.venv\Scripts\python.exe -m pytest tests/test_convnd.py -q
.venv\Scripts\python.exe -m src.tools.convnd_benchmark `
  --device cuda:0 --output data/native_convnd/benchmark.json
```

The tests compare values and gradients against PyTorch Conv1d/2d/3d and an
independent full N-dimensional reference, check finite differences, forced
tiling, noncontiguous storage, groups, rank 62, dispatcher contracts, AOTAutograd,
optimizer updates, empty batches, deterministic mode and stream handling.
The benchmark separately checks a full Conv4d assembled from batched Conv3d
taps; that reference has unrestricted 4D weights too. It records synchronized
forward/forward+backward times and extra allocated CUDA memory. Timings on one
shape are not evidence of end-to-end training efficiency.
The opt-in preserves `CUDA_VISIBLE_DEVICES` (set it to `0` to select GPU 0) and
sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` unless already specified, as required by
PyTorch for deterministic CUDA matrix multiplication.

### Local measurements

Windows, RTX 4090, PyTorch 2.10.0+cu130, FP32 with TF32 disabled for both
implementations; batch 1, kernel `3x3x3x3`, symmetric padding 1, 16 MiB patch
workspace, 3 warmups and 10 measured iterations. Values below are median
forward-plus-backward times and peak extra allocated CUDA tensor memory.

| Input `[B,C,*spatial]` / output channels | Native ConvNd | Conv3d-tap reference | Extra memory, native / reference |
| --- | --- | --- | --- |
| `[1,4,8,8,8,16]` / 8 | 1.29 ms | 2.18 ms | 8.5 / 1.9 MiB |
| `[1,8,16,16,16,32]` / 16 | 9.19 ms | 4.28 ms | 40.0 / 60.5 MiB |

The small case favors the addon; the larger case favors cuDNN-backed Conv3d
taps for speed. The native operator is not generally faster. These are bounded
operator measurements on a shared machine, not neutrino model results. Raw
records: [small](../../../data/native_convnd/benchmark_small_fp32.json) and
[medium](../../../data/native_convnd/benchmark_medium_fp32.json).

## Existing implementations and integration references

- [PyTorch custom C++ and CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
  documents dispatcher registration, fake kernels, autograd registration and
  `torch.library.opcheck` (which checks integration, not numerical gradients).
- [Conv4d-PyTorch](https://github.com/ZhengyuLiang24/Conv4d-PyTorch) and
  [pytorch_conv4D](https://github.com/dboyda/pytorch_conv4D) implement dense full
  Conv4d by composing lower-dimensional convolutions.
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) implements native
  high-dimensional sparse convolution. Its representation and sparse support
  semantics differ from this dense operator; current local compatibility has
  not been tested.
