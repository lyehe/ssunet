#pragma once

#include <ATen/ATen.h>

namespace ssunet {

// PyTorch supports at most 64 tensor dimensions, including batch and channels.
constexpr int64_t kMaxSpatialDims = 62;

struct Geometry {
  int64_t rank = 0;
  int64_t channels = 0;
  int64_t output_channels = 0;
  int64_t groups = 0;
  int64_t input_volume = 1;
  int64_t output_volume = 1;
  int64_t kernel_volume = 1;
  int64_t input[kMaxSpatialDims] = {};
  int64_t output[kMaxSpatialDims] = {};
  int64_t kernel[kMaxSpatialDims] = {};
  int64_t stride[kMaxSpatialDims] = {};
  int64_t padding[kMaxSpatialDims] = {};
  int64_t dilation[kMaxSpatialDims] = {};
};

#ifdef __CUDACC__
#define SSUNET_HD __host__ __device__
#else
#define SSUNET_HD
#endif

// Columns have layout [groups, channels_per_group * kernel_volume, tile_size].
SSUNET_HD inline int64_t column_input_offset(
    int64_t row, int64_t position, const Geometry& geometry) {
  int64_t channel = row / geometry.kernel_volume;
  int64_t tap = row % geometry.kernel_volume;
  int64_t offset = channel * geometry.input_volume;
  int64_t multiplier = 1;
  for (int64_t axis = geometry.rank - 1; axis >= 0; --axis) {
    const int64_t coordinate =
        (position % geometry.output[axis]) * geometry.stride[axis] -
        geometry.padding[axis] + (tap % geometry.kernel[axis]) * geometry.dilation[axis];
    if (coordinate < 0 || coordinate >= geometry.input[axis]) {
      return -1;
    }
    offset += coordinate * multiplier;
    multiplier *= geometry.input[axis];
    position /= geometry.output[axis];
    tap /= geometry.kernel[axis];
  }
  return offset;
}

// Inverse map for deterministic, race-free accumulation into an input element.
SSUNET_HD inline int64_t input_column_offset(
    int64_t index, int64_t tap, const Geometry& geometry, int64_t start, int64_t length) {
  const int64_t row = (index / geometry.input_volume) * geometry.kernel_volume + tap;
  int64_t spatial = index % geometry.input_volume;
  int64_t output_position = 0;
  int64_t multiplier = 1;
  for (int64_t axis = geometry.rank - 1; axis >= 0; --axis) {
    const int64_t numerator = spatial % geometry.input[axis] + geometry.padding[axis] -
        (tap % geometry.kernel[axis]) * geometry.dilation[axis];
    if (numerator < 0 || numerator % geometry.stride[axis] != 0) {
      return -1;
    }
    const int64_t coordinate = numerator / geometry.stride[axis];
    if (coordinate >= geometry.output[axis]) {
      return -1;
    }
    output_position += coordinate * multiplier;
    multiplier *= geometry.output[axis];
    spatial /= geometry.input[axis];
    tap /= geometry.kernel[axis];
  }
  if (output_position < start || output_position >= start + length) {
    return -1;
  }
  return row * length + output_position - start;
}

#undef SSUNET_HD

void unfold_cpu(const at::Tensor& input, at::Tensor& columns, const Geometry& geometry,
                int64_t start, int64_t length);
void fold_cpu(const at::Tensor& columns, at::Tensor& output, const Geometry& geometry,
              int64_t start, int64_t length);

#ifdef WITH_CUDA
void unfold_cuda(const at::Tensor& input, at::Tensor& columns, const Geometry& geometry,
                 int64_t start, int64_t length);
void fold_cuda(const at::Tensor& columns, at::Tensor& output, const Geometry& geometry,
               int64_t start, int64_t length);
#endif

}  // namespace ssunet
