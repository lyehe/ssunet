"""Full dense N-dimensional cross-correlation with optional native CPU/CUDA kernels.

Inputs use ``[batch, channels, *spatial]`` and weights use
``[out_channels, in_channels // groups, *kernel]``. All spatial axes participate
in one unrestricted kernel. This implementation does not factorize its weights
or assemble lower-dimensional convolutions.
"""

import math
import os
import shutil
import sys
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

DEFAULT_WORKSPACE_BYTES = 64 * 1024 * 1024
_BUILD_LOCK = threading.Lock()
_LOADED = False
_WITH_CUDA = False


def _tuple(value: int | Sequence[int], rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,) * rank
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        message = f"{name} must be an integer or a sequence of integers"
        raise TypeError(message)
    result = tuple(value)
    if len(result) != rank or any(not isinstance(v, int) or isinstance(v, bool) for v in result):
        message = f"{name} must contain exactly {rank} integers"
        raise ValueError(message)
    return result


@contextmanager
def _build_environment(with_cuda: bool):
    # Locate a venv-installed ninja even when Python was invoked without activation.
    changes = {}
    if sys.platform == "win32" and shutil.which("cl") is None:
        from torch.utils.cpp_extension import _get_vc_env

        changes.update({key.upper(): value for key, value in _get_vc_env("x86_amd64").items()})
        changes["DISTUTILS_USE_SDK"] = "1"
    changes["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + changes.get("PATH", os.environ["PATH"])
    )
    if "MAX_JOBS" not in os.environ:
        changes["MAX_JOBS"] = "2"
    if with_cuda and "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        capabilities = {
            torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
        }
        changes["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{a}.{b}" for a, b in sorted(capabilities))
    previous = {key: os.environ.get(key) for key in changes}
    os.environ.update(changes)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_native_convnd(*, with_cuda: bool | None = None, verbose: bool = False) -> None:
    """Build and cache the optional addon using PyTorch's C++ extension tooling.

    A C++ compiler and ninja are required. CUDA builds additionally require nvcc
    matching the installed PyTorch build. Set ``with_cuda=False`` for a CPU-only
    build. Build selection cannot be changed after loading in the same process.
    Importing :mod:`ssunet.ops` alone never invokes the compiler.
    """
    from torch.utils.cpp_extension import CUDA_HOME, load

    global _LOADED, _WITH_CUDA
    requested_cuda = (
        torch.version.cuda is not None and CUDA_HOME is not None if with_cuda is None else with_cuda
    )
    with _BUILD_LOCK:
        if _LOADED:
            if requested_cuda and not _WITH_CUDA:
                message = "CPU-only addon already loaded; use a fresh process for CUDA"
                raise RuntimeError(message)
            return
        source_dir = Path(__file__).parent / "csrc"
        sources = [str(source_dir / "convnd.cpp")]
        cflags = ["/O2"] if sys.platform == "win32" else ["-O3"]
        if requested_cuda:
            sources.append(str(source_dir / "convnd_cuda.cu"))
            cflags.append("-DWITH_CUDA")
        with _build_environment(requested_cuda):
            load(
                name="ssunet_convnd_cuda" if requested_cuda else "ssunet_convnd_cpu",
                sources=sources,
                extra_cflags=cflags,
                extra_cuda_cflags=["-O3", "-DWITH_CUDA"],
                with_cuda=requested_cuda,
                is_python_module=False,
                verbose=verbose,
            )
        _register_training()
        _WITH_CUDA = requested_cuda
        _LOADED = True


def _fake_forward(input, weight, bias, stride, padding, dilation, groups, workspace_bytes):
    shape = [input.shape[0], weight.shape[0]]
    shape.extend(
        (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
        for size, kernel, step, pad, dil in zip(
            input.shape[2:], weight.shape[2:], stride, padding, dilation, strict=True
        )
    )
    return input.new_empty(shape)


def _fake_backward(
    grad_output, input, weight, stride, padding, dilation, groups, workspace_bytes, output_mask
):
    return (
        torch.empty_like(input, memory_format=torch.contiguous_format) if output_mask[0] else None,
        torch.empty_like(weight, memory_format=torch.contiguous_format) if output_mask[1] else None,
        input.new_empty(weight.shape[0]) if output_mask[2] else None,
    )


def _setup_context(ctx, inputs, output):
    input, weight, bias, *parameters = inputs
    ctx.save_for_backward(input, weight)
    ctx.parameters = parameters
    ctx.has_bias = bias is not None


@once_differentiable
def _backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    mask = [
        ctx.needs_input_grad[0],
        ctx.needs_input_grad[1],
        ctx.has_bias and ctx.needs_input_grad[2],
    ]
    gradients = torch.ops.ssunet_native.conv_nd_backward.default(
        grad_output, input, weight, *ctx.parameters, mask
    )
    return (*gradients, None, None, None, None, None)


def _register_training() -> None:
    torch.library.register_fake("ssunet_native::conv_nd")(_fake_forward)
    torch.library.register_fake("ssunet_native::conv_nd_backward")(_fake_backward)
    torch.library.register_autograd(
        "ssunet_native::conv_nd", _backward, setup_context=_setup_context
    )


def conv_nd(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    *,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> Tensor:
    """Apply a full dense convolution over 1-62 spatial dimensions.

    Supports CPU/CUDA float32 and float64, groups, symmetric zero padding,
    stride, dilation, noncontiguous tensors and first-order gradients. An empty
    batch is supported; empty channels/spatial axes are rejected. Higher-order
    gradients, autocast, other padding modes and transposed convolution are not
    implemented. CUDA input gradients use atomic addition unless PyTorch's
    deterministic algorithms mode is enabled.

    ``workspace_bytes`` budgets patch buffers and an output tile. It excludes
    inputs, outputs, parameter gradients, contiguous copies and GEMM workspace.
    Backward recomputes patches rather than saving the unfolded input.
    """
    rank = input.ndim - 2
    if not 1 <= rank <= 62:
        message = "input must have batch, channel and 1-62 spatial dimensions"
        raise ValueError(message)
    parameters = (
        _tuple(stride, rank, "stride"),
        _tuple(padding, rank, "padding"),
        _tuple(dilation, rank, "dilation"),
    )
    if not _LOADED:
        load_native_convnd()
    return torch.ops.ssunet_native.conv_nd.default(
        input, weight, bias, *parameters, groups, workspace_bytes
    )


class ConvNd(nn.Module):
    """Trainable full N-dimensional convolution with PyTorch-style parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        dimensions: int,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or not 1 <= dimensions <= 62
        ):
            message = "dimensions must be an integer from 1 to 62"
            raise ValueError(message)
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuple(kernel_size, dimensions, "kernel_size")
        self.stride = _tuple(stride, dimensions, "stride")
        self.padding = _tuple(padding, dimensions, "padding")
        self.dilation = _tuple(dilation, dimensions, "dilation")
        self.groups = groups
        self.workspace_bytes = workspace_bytes
        if groups <= 0 or in_channels <= 0 or out_channels <= 0:
            message = "channels and groups must be positive"
            raise ValueError(message)
        if in_channels % groups or out_channels % groups:
            message = "channels must be divisible by groups"
            raise ValueError(message)
        if min(*self.kernel_size, *self.stride, *self.dilation) <= 0 or min(self.padding) < 0:
            message = "kernel, stride and dilation must be positive; padding nonnegative"
            raise ValueError(message)
        if workspace_bytes <= 0:
            message = "workspace_bytes must be positive"
            raise ValueError(message)
        options = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size, **options)
        )
        self.bias = nn.Parameter(torch.empty(out_channels, **options)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the same initialization rule as PyTorch convolution modules."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels // self.groups * math.prod(self.kernel_size)
            nn.init.uniform_(self.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, input: Tensor) -> Tensor:
        """Convolve all spatial axes of a batched channels-first tensor."""
        if input.ndim != self.dimensions + 2:
            message = f"expected {self.dimensions + 2} dimensions, got {input.ndim}"
            raise ValueError(message)
        return conv_nd(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            workspace_bytes=self.workspace_bytes,
        )

    def extra_repr(self) -> str:
        """Describe the kernel geometry in module summaries."""
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"dimensions={self.dimensions}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}"
        )


class Conv4d(ConvNd):
    """Convenience wrapper for inputs shaped ``[batch, channels, x, y, z, time]``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, dimensions=4, **kwargs)
