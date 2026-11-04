#include "convnd.h"

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace ssunet {
namespace {

int64_t checked_product(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left >= 0 && right >= 0 &&
                  (right == 0 || left <= std::numeric_limits<int64_t>::max() / right),
              label, " overflows int64");
  return left * right;
}

int64_t checked_sum(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left >= 0 && right >= 0 &&
                  left <= std::numeric_limits<int64_t>::max() - right,
              label, " overflows int64");
  return left + right;
}

Geometry make_geometry(const at::Tensor& input, const at::Tensor& weight,
                       at::IntArrayRef stride, at::IntArrayRef padding,
                       at::IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(input.layout() == at::kStrided && weight.layout() == at::kStrided,
              "conv_nd requires strided tensors");
  TORCH_CHECK(input.device().is_cpu() || input.device().is_cuda(),
              "conv_nd supports CPU and CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble,
              "conv_nd currently supports float32 and float64");
  TORCH_CHECK(weight.device() == input.device() && weight.scalar_type() == input.scalar_type(),
              "input and weight must have the same device and dtype");
  TORCH_CHECK(input.dim() >= 3 && input.dim() <= kMaxSpatialDims + 2,
              "conv_nd requires 1 to 62 spatial dimensions");
  TORCH_CHECK(weight.dim() == input.dim(), "input and weight ranks must match");
  Geometry geometry;
  geometry.rank = input.dim() - 2;
  TORCH_CHECK(static_cast<int64_t>(stride.size()) == geometry.rank &&
                  static_cast<int64_t>(padding.size()) == geometry.rank &&
                  static_cast<int64_t>(dilation.size()) == geometry.rank,
              "stride, padding and dilation must match the spatial rank");
  TORCH_CHECK(groups > 0, "groups must be positive");
  geometry.channels = input.size(1);
  geometry.output_channels = weight.size(0);
  geometry.groups = groups;
  TORCH_CHECK(geometry.channels > 0 && geometry.output_channels > 0 &&
                  geometry.channels % groups == 0 && geometry.output_channels % groups == 0 &&
                  weight.size(1) == geometry.channels / groups,
              "input/weight channels must match and be divisible by groups");
  for (int64_t axis = 0; axis < geometry.rank; ++axis) {
    TORCH_CHECK(input.size(axis + 2) > 0 && weight.size(axis + 2) > 0,
                "spatial and kernel dimensions must be positive");
    TORCH_CHECK(stride[axis] > 0 && dilation[axis] > 0 && padding[axis] >= 0,
                "stride and dilation must be positive; padding must be nonnegative");
    geometry.input[axis] = input.size(axis + 2);
    geometry.kernel[axis] = weight.size(axis + 2);
    geometry.stride[axis] = stride[axis];
    geometry.padding[axis] = padding[axis];
    geometry.dilation[axis] = dilation[axis];
    const int64_t effective_kernel = checked_sum(
        checked_product(dilation[axis], geometry.kernel[axis] - 1, "dilated kernel"),
        1, "dilated kernel");
    const int64_t padded_input = checked_sum(
        geometry.input[axis], checked_product(2, padding[axis], "padding"), "padded input");
    TORCH_CHECK(padded_input >= effective_kernel, "kernel exceeds padded input");
    geometry.output[axis] = (padded_input - effective_kernel) / stride[axis] + 1;
    geometry.input_volume = checked_product(
        geometry.input_volume, geometry.input[axis], "input spatial volume");
    geometry.output_volume = checked_product(
        geometry.output_volume, geometry.output[axis], "output spatial volume");
    geometry.kernel_volume = checked_product(
        geometry.kernel_volume, geometry.kernel[axis], "kernel volume");
  }
  return geometry;
}

int64_t tile_size(const Geometry& geometry, int64_t element_size, int64_t workspace_bytes) {
  // Budget two column buffers and one output tile; tensor storage and GEMM workspace are separate.
  const int64_t column_rows = checked_product(
      geometry.channels, geometry.kernel_volume, "patch rows");
  const int64_t tile_elements = checked_sum(
      checked_product(2, column_rows, "patch tile"), geometry.output_channels, "workspace tile");
  const int64_t bytes_per_position = checked_product(
      element_size, tile_elements, "workspace bytes per position");
  TORCH_CHECK(workspace_bytes >= bytes_per_position,
              "workspace_bytes is too small for one output position; requires at least ",
              bytes_per_position);
  return std::min(geometry.output_volume, workspace_bytes / bytes_per_position);
}

void unfold(const at::Tensor& input, at::Tensor& columns, const Geometry& geometry,
            int64_t start, int64_t length) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    unfold_cuda(input, columns, geometry, start, length);
#else
    TORCH_CHECK(false, "conv_nd was built without CUDA; rebuild with with_cuda=True");
#endif
  } else {
    unfold_cpu(input, columns, geometry, start, length);
  }
}

void fold(const at::Tensor& columns, at::Tensor& output, const Geometry& geometry,
          int64_t start, int64_t length) {
  if (output.is_cuda()) {
#ifdef WITH_CUDA
    fold_cuda(columns, output, geometry, start, length);
#else
    TORCH_CHECK(false, "conv_nd was built without CUDA; rebuild with with_cuda=True");
#endif
  } else {
    fold_cpu(columns, output, geometry, start, length);
  }
}

std::vector<int64_t> output_shape(const at::Tensor& input, const Geometry& geometry) {
  std::vector<int64_t> result = {input.size(0), geometry.output_channels};
  result.insert(result.end(), geometry.output, geometry.output + geometry.rank);
  return result;
}

}  // namespace

void unfold_cpu(const at::Tensor& input, at::Tensor& columns, const Geometry& geometry,
                int64_t start, int64_t length) {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unfold_nd_cpu", [&] {
    const scalar_t* source = input.const_data_ptr<scalar_t>();
    scalar_t* target = columns.mutable_data_ptr<scalar_t>();
    at::parallel_for(0, columns.numel(), 4096, [&](int64_t begin, int64_t end) {
      for (int64_t index = begin; index < end; ++index) {
        const int64_t offset = column_input_offset(index / length, start + index % length, geometry);
        target[index] = offset < 0 ? scalar_t(0) : source[offset];
      }
    });
  });
}

void fold_cpu(const at::Tensor& columns, at::Tensor& output, const Geometry& geometry,
              int64_t start, int64_t length) {
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "fold_nd_cpu", [&] {
    const scalar_t* source = columns.const_data_ptr<scalar_t>();
    scalar_t* target = output.mutable_data_ptr<scalar_t>();
    at::parallel_for(0, output.numel(), 512, [&](int64_t begin, int64_t end) {
      for (int64_t index = begin; index < end; ++index) {
        scalar_t sum = 0;
        for (int64_t tap = 0; tap < geometry.kernel_volume; ++tap) {
          const int64_t offset = input_column_offset(index, tap, geometry, start, length);
          if (offset >= 0) {
            sum += source[offset];
          }
        }
        target[index] += sum;
      }
    });
  });
}

at::Tensor conv_nd(const at::Tensor& input, const at::Tensor& weight,
                   const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                   at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                   int64_t workspace_bytes) {
  const auto geometry = make_geometry(input, weight, stride, padding, dilation, groups);
  if (bias.has_value()) {
    TORCH_CHECK(bias->layout() == at::kStrided && bias->dim() == 1 &&
                    bias->size(0) == geometry.output_channels &&
                    bias->device() == input.device() && bias->scalar_type() == input.scalar_type(),
                "bias must match output channels, device and dtype");
  }
  const c10::OptionalDeviceGuard guard(input.device());
  const auto contiguous_input = input.contiguous();
  const auto flat_weight = weight.contiguous().view(
      {groups, geometry.output_channels / groups, -1});
  auto output = at::empty(output_shape(input, geometry), input.options());
  auto flat_output = output.view({input.size(0), groups, geometry.output_channels / groups,
                                 geometry.output_volume});
  const int64_t tile = tile_size(geometry, input.element_size(), workspace_bytes);
  const int64_t rows = geometry.channels / groups * geometry.kernel_volume;
  for (int64_t batch = 0; batch < input.size(0); ++batch) {
    const auto image = contiguous_input.select(0, batch);
    for (int64_t start = 0; start < geometry.output_volume; start += tile) {
      const int64_t length = std::min(tile, geometry.output_volume - start);
      auto columns = at::empty({groups, rows, length}, input.options());
      unfold(image, columns, geometry, start, length);
      flat_output.select(0, batch).slice(2, start, start + length).copy_(
          at::bmm(flat_weight, columns));
    }
  }
  if (bias.has_value()) {
    std::vector<int64_t> bias_shape(input.dim(), 1);
    bias_shape[1] = geometry.output_channels;
    output.add_(bias->reshape(bias_shape));
  }
  return output;
}

std::tuple<std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>>
conv_nd_backward(const at::Tensor& grad_output, const at::Tensor& input,
                 const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
                 at::IntArrayRef dilation, int64_t groups, int64_t workspace_bytes,
                 std::array<bool, 3> output_mask) {
  const auto geometry = make_geometry(input, weight, stride, padding, dilation, groups);
  TORCH_CHECK(grad_output.sizes().vec() == output_shape(input, geometry) &&
                  grad_output.device() == input.device() &&
                  grad_output.scalar_type() == input.scalar_type() &&
                  grad_output.layout() == at::kStrided,
              "grad_output must match the convolution output shape, device and dtype");
  const c10::OptionalDeviceGuard guard(input.device());
  const int64_t tile = tile_size(geometry, input.element_size(), workspace_bytes);
  const int64_t rows = geometry.channels / groups * geometry.kernel_volume;
  const auto contiguous_input = input.contiguous();
  const auto flat_weight = weight.contiguous().view(
      {groups, geometry.output_channels / groups, rows});
  const auto flat_grad = grad_output.contiguous().view(
      {input.size(0), groups, geometry.output_channels / groups, geometry.output_volume});
  std::optional<at::Tensor> grad_input;
  std::optional<at::Tensor> grad_weight;
  std::optional<at::Tensor> grad_bias;
  if (output_mask[0]) grad_input = at::zeros(input.sizes(), input.options());
  if (output_mask[1]) grad_weight = at::zeros(weight.sizes(), weight.options());
  if (output_mask[2]) {
    grad_bias = flat_grad.sum(at::IntArrayRef({0, 3})).reshape({geometry.output_channels});
  }
  if (!output_mask[0] && !output_mask[1]) return {grad_input, grad_weight, grad_bias};

  for (int64_t batch = 0; batch < input.size(0); ++batch) {
    const auto image = contiguous_input.select(0, batch);
    for (int64_t start = 0; start < geometry.output_volume; start += tile) {
      const int64_t length = std::min(tile, geometry.output_volume - start);
      const auto tile_grad = flat_grad.select(0, batch).slice(2, start, start + length);
      if (output_mask[1]) {
        auto columns = at::empty({groups, rows, length}, input.options());
        unfold(image, columns, geometry, start, length);
        grad_weight->view({groups, geometry.output_channels / groups, rows}).add_(
            at::bmm(tile_grad, columns.transpose(1, 2)));
      }
      if (output_mask[0]) {
        const auto grad_columns = at::bmm(flat_weight.transpose(1, 2), tile_grad);
        auto image_grad = grad_input->select(0, batch);
        fold(grad_columns, image_grad, geometry, start, length);
      }
    }
  }
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace ssunet

TORCH_LIBRARY(ssunet_native, library) {
  library.def("conv_nd(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, "
              "int[] dilation, int groups, int workspace_bytes) -> Tensor");
  library.def("conv_nd_backward(Tensor grad_output, Tensor input, Tensor weight, int[] stride, "
              "int[] padding, int[] dilation, int groups, int workspace_bytes, "
              "bool[3] output_mask) -> (Tensor?, Tensor?, Tensor?)");
}

TORCH_LIBRARY_IMPL(ssunet_native, CPU, library) {
  library.impl("conv_nd", &ssunet::conv_nd);
  library.impl("conv_nd_backward", &ssunet::conv_nd_backward);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(ssunet_native, CUDA, library) {
  library.impl("conv_nd", &ssunet::conv_nd);
  library.impl("conv_nd_backward", &ssunet::conv_nd_backward);
}
#endif
