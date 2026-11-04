#include "convnd.h"

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>

namespace ssunet {
namespace {

constexpr int kThreads = 256;

int blocks_for(int64_t elements) {
  return static_cast<int>(std::min<int64_t>((elements + kThreads - 1) / kThreads, 65535));
}

template <typename scalar_t>
__global__ void unfold_kernel(const scalar_t* input, scalar_t* columns, Geometry geometry,
                              int64_t start, int64_t length, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t offset = column_input_offset(index / length, start + index % length, geometry);
    columns[index] = offset < 0 ? scalar_t(0) : input[offset];
  }
}

template <typename scalar_t>
__global__ void fold_atomic_kernel(const scalar_t* columns, scalar_t* output, Geometry geometry,
                                   int64_t start, int64_t length, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t offset = column_input_offset(index / length, start + index % length, geometry);
    if (offset >= 0) gpuAtomicAdd(output + offset, columns[index]);
  }
}

template <typename scalar_t>
__global__ void fold_gather_kernel(const scalar_t* columns, scalar_t* output, Geometry geometry,
                                   int64_t start, int64_t length, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    scalar_t sum = 0;
    for (int64_t tap = 0; tap < geometry.kernel_volume; ++tap) {
      const int64_t offset = input_column_offset(index, tap, geometry, start, length);
      if (offset >= 0) sum += columns[offset];
    }
    output[index] += sum;
  }
}

}  // namespace

void unfold_cuda(const at::Tensor& input, at::Tensor& columns, const Geometry& geometry,
                 int64_t start, int64_t length) {
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unfold_nd_cuda", [&] {
    unfold_kernel<<<blocks_for(columns.numel()), kThreads, 0, stream>>>(
        input.const_data_ptr<scalar_t>(), columns.mutable_data_ptr<scalar_t>(), geometry,
        start, length, columns.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fold_cuda(const at::Tensor& columns, at::Tensor& output, const Geometry& geometry,
               int64_t start, int64_t length) {
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "fold_nd_cuda", [&] {
    if (at::globalContext().deterministicAlgorithms()) {
      fold_gather_kernel<<<blocks_for(output.numel()), kThreads, 0, stream>>>(
          columns.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(), geometry,
          start, length, output.numel());
    } else {
      fold_atomic_kernel<<<blocks_for(columns.numel()), kThreads, 0, stream>>>(
          columns.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(), geometry,
          start, length, columns.numel());
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ssunet
