"""Numerical and dispatcher checks for the optional native convolution addon.

Run explicitly with SSUNET_TEST_NATIVE=1; normal project tests need no compiler.
"""

import itertools
import os

import pytest
import torch
from torch.nn import functional as functional
from torch.utils.cpp_extension import CUDA_HOME

from src.ssunet.ops import Conv4d, ConvNd, conv_nd, load_native_convnd

pytestmark = pytest.mark.skipif(
    os.environ.get("SSUNET_TEST_NATIVE") != "1",
    reason="set SSUNET_TEST_NATIVE=1 to build/test addon",
)
_CUDA_AVAILABLE = CUDA_HOME is not None and torch.cuda.is_available()
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA GPU/toolkit is unavailable"),
    ),
]


@pytest.fixture(scope="module", autouse=True)
def native_addon():
    """Build once, with CUDA when this test environment has a toolkit and GPU."""
    load_native_convnd(with_cuda=_CUDA_AVAILABLE)
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def reference_nd(input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1):
    """Independent tap-by-tap reference using channel contractions and spatial slices."""
    rank = input.ndim - 2
    stride = stride or (1,) * rank
    padding = padding or (0,) * rank
    dilation = dilation or (1,) * rank
    padded = functional.pad(input, tuple(v for pad in reversed(padding) for v in (pad, pad)))
    output_shape = tuple(
        (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
        for size, kernel, step, pad, dil in zip(
            input.shape[2:], weight.shape[2:], stride, padding, dilation, strict=True
        )
    )
    terms = []
    for tap in itertools.product(*(range(size) for size in weight.shape[2:])):
        slices = tuple(
            slice(k * dil, k * dil + size * step, step)
            for k, dil, size, step in zip(tap, dilation, output_shape, stride, strict=True)
        )
        patch = padded[(slice(None), slice(None), *slices)].reshape(
            input.shape[0], groups, input.shape[1] // groups, -1
        )
        coefficient = weight[(slice(None), slice(None), *tap)].reshape(
            groups, weight.shape[0] // groups, input.shape[1] // groups
        )
        terms.append(torch.einsum("goc,bgcp->bgop", coefficient, patch))
    result = torch.stack(terms).sum(0).reshape(input.shape[0], weight.shape[0], *output_shape)
    if bias is not None:
        result = result + bias.reshape(1, -1, *((1,) * rank))
    return result


def compare_values_and_gradients(actual, expected, inputs, *, atol=1e-9, rtol=1e-9):
    """Compare outputs and vector-Jacobian products using a shared random cotangent."""
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    cotangent = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, inputs, cotangent, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, inputs, cotangent)
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_pytorch_convolution(device, rank, dtype):
    """Native ranks 1-3 agree with PyTorch for groups, stride, padding and dilation."""
    torch.manual_seed(610)
    input = torch.randn(2, 4, *((6,) * rank), device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(6, 2, *((2,) * rank), device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(6, device=device, dtype=dtype, requires_grad=True)
    kwargs = {"stride": (2,) * rank, "padding": (1,) * rank, "dilation": (2,) * rank, "groups": 2}
    actual = conv_nd(input, weight, bias, **kwargs)
    expected = getattr(functional, f"conv{rank}d")(input, weight, bias, **kwargs)
    tolerance = 8e-5 if dtype == torch.float32 else 1e-9
    compare_values_and_gradients(
        actual, expected, (input, weight, bias), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [4, 5, 6, 8])
def test_full_nd_matches_independent_reference(device, rank):
    """Every spatial axis has independent taps, including ranks above four."""
    torch.manual_seed(44 + rank)
    input = torch.randn(
        1, 4, *((3,) * rank), device=device, dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn(
        4, 2, *((2,) * rank), device=device, dtype=torch.float64, requires_grad=True
    )
    bias = torch.randn(4, device=device, dtype=torch.float64, requires_grad=True)
    stride = (2,) + (1,) * (rank - 1)
    padding = (0, 1) + (0,) * (rank - 2)
    actual = conv_nd(input, weight, bias, stride=stride, padding=padding, groups=2)
    expected = reference_nd(input, weight, bias, stride=stride, padding=padding, groups=2)
    compare_values_and_gradients(actual, expected, (input, weight, bias), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_noncontiguous_dilated_4d(device):
    """Forced two-position tiles preserve values/gradients for noncontiguous tensors."""
    input = torch.randn(2, 4, 3, 4, 3, 4, device=device, dtype=torch.float64).transpose(-1, -2)
    weight = torch.randn(6, 2, 2, 2, 1, 2, device=device, dtype=torch.float64).transpose(-1, -2)
    input.requires_grad_()
    weight.requires_grad_()
    bias = torch.randn(12, device=device, dtype=torch.float64)[::2].requires_grad_()
    stride, padding, dilation = (1, 2, 1, 1), (1, 1, 1, 0), (2, 1, 1, 1)
    per_position = 8 * (2 * 4 * 8 + 6)
    actual = conv_nd(
        input, weight, bias, stride, padding, dilation, 2, workspace_bytes=2 * per_position
    )
    expected = reference_nd(input, weight, bias, stride, padding, dilation, groups=2)
    compare_values_and_gradients(actual, expected, (input, weight, bias), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_4d_gradcheck(device):
    """Finite differences independently check input, weight and bias derivatives."""
    input = torch.randn(1, 2, 2, 2, 2, 3, device=device, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(2, 2, 2, 1, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(2, device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, w, b: conv_nd(x, w, b, padding=(1, 0, 0, 1), stride=(1, 1, 1, 2)),
        (input, weight, bias),
        fast_mode=True,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_dispatcher_opcheck(device):
    """Check schema, fake tensors, autograd registration and AOT dispatch contracts."""
    input = torch.randn(1, 2, 3, 3, 3, 3, device=device, requires_grad=True)
    weight = torch.randn(4, 2, 2, 2, 2, 2, device=device, requires_grad=True)
    bias = torch.randn(4, device=device, requires_grad=True)
    results = torch.library.opcheck(
        torch.ops.ssunet_native.conv_nd.default,
        (input, weight, bias, [1] * 4, [0] * 4, [1] * 4, 1, 65536),
    )
    assert all(value == "SUCCESS" for value in results.values())


@pytest.mark.parametrize("device", DEVICES)
def test_module_compile_and_optimizer(device):
    """A full 4D layer trains through AOTAutograd and round-trips its parameters."""
    torch.manual_seed(901)
    layer = Conv4d(2, 3, 2, padding=1, device=device)
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    input = torch.randn(2, 2, 3, 3, 3, 3, device=device, requires_grad=True)
    expected = layer(input)
    torch.testing.assert_close(compiled(input), expected)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    before = layer.weight.detach().clone()
    compiled(input).square().mean().backward()
    optimizer.step()
    assert not torch.equal(layer.weight, before)
    restored = ConvNd(2, 3, 2, dimensions=4, padding=1, device=device)
    restored.load_state_dict(layer.state_dict())
    torch.testing.assert_close(restored(input), layer(input))


@pytest.mark.parametrize("device", DEVICES)
def test_empty_batch(device):
    """Empty batches return the correct shape and zero parameter gradients."""
    layer = Conv4d(2, 4, 2, device=device)
    input = torch.empty(0, 2, 3, 3, 3, 3, device=device, requires_grad=True)
    output = layer(input)
    assert output.shape == (0, 4, 2, 2, 2, 2)
    output.sum().backward()
    assert input.grad.shape == input.shape
    assert torch.count_nonzero(layer.weight.grad) == 0
    assert torch.count_nonzero(layer.bias.grad) == 0


@pytest.mark.parametrize("active", ["input", "weight", "bias"])
def test_partial_gradient_requirements(active):
    """Frozen inputs/parameters do not require unused derivative buffers."""
    tensors = {
        "input": torch.randn(1, 2, 3, 3, 3, 3, requires_grad=active == "input"),
        "weight": torch.randn(4, 2, 2, 2, 2, 2, requires_grad=active == "weight"),
        "bias": torch.randn(4, requires_grad=active == "bias"),
    }
    conv_nd(**tensors).sum().backward()
    assert tensors[active].grad is not None
    assert torch.isfinite(tensors[active].grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_rank_limit_and_depthwise(device):
    """The generic index mapping reaches PyTorch's 64-axis limit and handles depthwise groups."""
    layer = ConvNd(
        2, 4, (1,) * 61 + (2,), dimensions=62, groups=2, dtype=torch.float64, device=device
    )
    input = torch.randn(
        (1, 2) + (1,) * 61 + (3,), dtype=torch.float64, device=device, requires_grad=True
    )
    output = layer(input)
    expected = functional.conv1d(
        input.reshape(1, 2, 3), layer.weight.reshape(4, 1, 2), layer.bias, groups=2
    )
    compare_values_and_gradients(
        output.reshape(1, 4, 2), expected, (input, layer.weight, layer.bias)
    )


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA GPU/toolkit is unavailable")
def test_cuda_determinism_and_nondefault_stream():
    """Deterministic gather and atomic fold honor the caller's CUDA stream."""
    stream = torch.cuda.Stream()
    input = torch.randn(1, 2, 3, 3, 3, 3, device="cuda", requires_grad=True)
    weight = torch.randn(3, 2, 2, 2, 2, 2, device="cuda", requires_grad=True)
    stream.wait_stream(torch.cuda.current_stream())
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.cuda.stream(stream):
            output = conv_nd(input, weight, padding=1)
            grad1 = torch.autograd.grad(output.square().sum(), (input, weight))
            grad2 = torch.autograd.grad(
                conv_nd(input, weight, padding=1).square().sum(), (input, weight)
            )
            expected = reference_nd(input, weight, padding=(1,) * 4)
        stream.synchronize()
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
        for first, second in zip(grad1, grad2, strict=True):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"groups": 0}, "groups must be positive"),
        ({"stride": 0}, "stride and dilation"),
        ({"padding": -1}, "padding must be nonnegative"),
        ({"workspace_bytes": 1}, "workspace_bytes is too small"),
        ({"groups": 3}, "channels must match"),
    ],
)
def test_invalid_geometry(kwargs, message):
    """Invalid native parameters fail with actionable errors."""
    with pytest.raises(RuntimeError, match=message):
        conv_nd(torch.randn(1, 2, 3, 3, 3, 3), torch.randn(4, 2, 2, 2, 2, 2), **kwargs)


def test_unsupported_dtype_and_bad_bias():
    """Unsupported precision and inconsistent tensor metadata fail explicitly."""
    input, weight = torch.randn(1, 2, 3, 3, 3, 3), torch.randn(4, 2, 2, 2, 2, 2)
    with pytest.raises(RuntimeError, match="float32 and float64"):
        conv_nd(input.half(), weight.half())
    with pytest.raises(RuntimeError, match="same device and dtype"):
        conv_nd(input, weight.double())
    with pytest.raises(RuntimeError, match="bias must match"):
        conv_nd(input, weight, torch.zeros(3))
    with pytest.raises(RuntimeError, match="kernel exceeds"):
        conv_nd(input, torch.randn(4, 2, 4, 2, 2, 2))
    with pytest.raises(ValueError, match="exactly 4 integers"):
        conv_nd(input, weight, stride=(1, 1))


def test_module_validates_rank():
    """The four-dimensional wrapper cannot silently accept a different spatial rank."""
    with pytest.raises(ValueError, match="expected 6 dimensions"):
        Conv4d(2, 3, 2)(torch.randn(1, 2, 3, 3, 3))
    with pytest.raises(ValueError, match="dimensions must"):
        ConvNd(2, 3, 2, dimensions=0)
    with pytest.raises(ValueError, match="divisible"):
        ConvNd(3, 4, 2, dimensions=4, groups=2)


def test_higher_order_gradients_fail_explicitly():
    """Unsupported second derivatives fail instead of returning unverified values."""
    input = torch.randn(1, 1, 2, 2, 2, 2, requires_grad=True)
    weight = torch.randn(1, 1, 1, 1, 1, 1, requires_grad=True)
    gradient = torch.autograd.grad(conv_nd(input, weight).square().sum(), input, create_graph=True)[
        0
    ]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), weight)


def test_geometry_rejects_integer_overflow():
    """Large rank/padding cannot wrap native shape or workspace calculations."""
    with pytest.raises(RuntimeError, match="output spatial volume overflows"):
        conv_nd(torch.ones((1,) * 64), torch.ones((1,) * 64), padding=1)
    with pytest.raises(RuntimeError, match="padding overflows"):
        conv_nd(torch.ones(1, 1, 2), torch.ones(1, 1, 1), padding=2**63 - 1)
