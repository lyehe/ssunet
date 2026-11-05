"""Bounded correctness/timing comparison of native ConvNd and a full Conv3d-tap reference."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as functional

from src.ssunet.ops import conv_nd, load_native_convnd


def conv3d_tap_reference(input, weight, bias, padding):
    """Compute unrestricted full Conv4d by batching one spatial axis through Conv3d."""
    extent = input.shape[2]
    padded = functional.pad(input, (0, 0, 0, 0, 0, 0, padding, padding))
    terms = []
    for tap in range(weight.shape[2]):
        frames = padded[:, :, tap : tap + extent].permute(0, 2, 1, 3, 4, 5)
        frames = frames.reshape(input.shape[0] * extent, input.shape[1], *input.shape[3:])
        result = functional.conv3d(frames, weight[:, :, tap].contiguous(), padding=padding)
        terms.append(
            result.reshape(input.shape[0], extent, weight.shape[0], *result.shape[2:]).permute(
                0, 2, 1, 3, 4, 5
            )
        )
    return torch.stack(terms).sum(0) + bias.reshape(1, -1, 1, 1, 1, 1)


def measure(operation, tensors, *, backward, device, warmup, iterations):
    """Measure synchronized wall time and peak tensor memory without optimizer state."""

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run():
        with torch.set_grad_enabled(backward):
            output = operation(*tensors)
            if backward:
                torch.autograd.grad(output.square().mean(), tensors)

    for _ in range(warmup):
        run()
    synchronize()
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        run()
        synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "peak_extra_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) - baseline if device.type == "cuda" else None
        ),
    }


def main():
    """Build once, check values/gradients, then write a reproducible timing record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spatial", type=int, nargs=4, default=[8, 8, 8, 16])
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--workspace-mib", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kernel <= 0 or args.kernel % 2 != 1:
        parser.error("--kernel must be positive and odd for this same-shape comparison")
    if (
        min(
            *args.spatial,
            args.in_channels,
            args.out_channels,
            args.batch,
            args.workspace_mib,
            args.iterations,
        )
        <= 0
        or args.warmup < 0
    ):
        parser.error("shape, workspace and iterations must be positive; warmup nonnegative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    load_native_convnd(with_cuda=device.type == "cuda")
    # Compare the same FP32 arithmetic policy rather than IEEE GEMM versus TF32 cuDNN.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(638)
    padding = args.kernel // 2
    input = torch.randn(
        args.batch, args.in_channels, *args.spatial, device=device, requires_grad=True
    )
    weight = torch.randn(
        args.out_channels,
        args.in_channels,
        *((args.kernel,) * 4),
        device=device,
        requires_grad=True,
    )
    bias = torch.randn(args.out_channels, device=device, requires_grad=True)
    tensors = (input, weight, bias)

    def native(x, w, b):
        return conv_nd(x, w, b, padding=padding, workspace_bytes=args.workspace_mib * 1024 * 1024)

    def reference(x, w, b):
        return conv3d_tap_reference(x, w, b, padding)

    observed, expected = native(*tensors), reference(*tensors)
    torch.testing.assert_close(observed, expected, atol=2e-4, rtol=2e-4)
    cotangent = torch.randn_like(observed)
    actual_grad = torch.autograd.grad(observed, tensors, cotangent)
    reference_grad = torch.autograd.grad(expected, tensors, cotangent)
    for actual, target in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, target, atol=5e-3, rtol=3e-4)
    correctness = {
        "output_max_absolute_error": (observed - expected).abs().max().item(),
        "gradient_max_absolute_errors": [
            (actual - target).abs().max().item()
            for actual, target in zip(actual_grad, reference_grad, strict=True)
        ],
    }
    del observed, expected, actual_grad, reference_grad, cotangent
    results = {}
    for name, operation in [("native_nd", native), ("conv3d_tap_reference", reference)]:
        results[name] = {
            mode: measure(
                operation,
                tensors,
                backward=backward,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for mode, backward in [("forward", False), ("forward_backward", True)]
        }
    record = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dtype": str(input.dtype),
        "input_shape": list(input.shape),
        "weight_shape": list(weight.shape),
        "workspace_mib": args.workspace_mib,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "correctness": correctness,
        "results": results,
        "limitations": [
            "One shape, FP32, no optimizer state, shared machine; "
            "not an end-to-end training claim.",
            "Both operators implement full 4D weights; the reference uses batched Conv3d taps.",
            "Workspace excludes tensor storage, copies and weight-gradient/GEMM temporaries.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    parser.exit(message=json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
