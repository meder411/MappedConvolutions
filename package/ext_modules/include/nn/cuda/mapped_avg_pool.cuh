#ifndef MAPPED_AVG_POOL_CUH_
#define MAPPED_AVG_POOL_CUH_

#include <torch/extension.h>

#include "core/resample.h"
#include "cuda_helper.h"
#include "nn/common/mapped_avg_pool.h"

namespace mapped_conv {
namespace nn {
namespace cuda {

template <typename T>
__global__ void MappedAvgPool2DKernel(
    const int n, const T *__restrict__ in_data_ptr,
    const T *__restrict__ sample_map_ptr,  // OH x OW x K x 2
    const int channels, const int in_height, const int in_width,
    const int out_height, const int out_width, const int kernel_size,
    const int interpolation, T *__restrict__ out_data_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedAvgPool2D(index, in_data_ptr, sample_map_ptr, channels,
                          in_height, in_width, out_height, out_width,
                          kernel_size, interpolation, out_data_ptr);
}

void MappedAvgPool2DLauncher(torch::Tensor in_data, torch::Tensor sample_map,
                             const int channels, const int in_height,
                             const int in_width, const int out_height,
                             const int out_width, const int kernel_size,
                             const int interpolation, torch::Tensor out_data) {
  const int num_kernels = channels * out_height * out_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(
      in_data.scalar_type(), "MappedAvgPool2DKernel", ([&] {
        MappedAvgPool2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, in_data.data<scalar_t>(), sample_map.data<scalar_t>(),
            channels, in_height, in_width, out_height, out_width, kernel_size,
            interpolation, out_data.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

template <typename T>
__global__ void MappedAvgUnpool2DKernel(
    const int n, const T *__restrict__ grad_output_ptr,
    const T *__restrict__ sample_map_ptr, const int channels,
    const int orig_height, const int orig_width, const int pooled_height,
    const int pooled_width, const int kernel_size, const int interpolation,
    T *const grad_input_ptr) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedAvgUnpool2D(index, grad_output_ptr, sample_map_ptr, channels,
                            orig_height, orig_width, pooled_height,
                            pooled_width, kernel_size, interpolation,
                            grad_input_ptr);
}

void MappedAvgUnpool2DLauncher(torch::Tensor grad_output,
                               torch::Tensor sample_map, const int channels,
                               const int orig_height, const int orig_width,
                               const int pooled_height, const int pooled_width,
                               const int kernel_size, const int interpolation,
                               torch::Tensor grad_input) {
  const int num_kernels = channels * pooled_height * pooled_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "MappedAvgUnpool2DLauncher", ([&] {
        MappedAvgUnpool2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, grad_output.data<scalar_t>(),
            sample_map.data<scalar_t>(), channels, orig_height, orig_width,
            pooled_height, pooled_width, kernel_size, interpolation,
            grad_input.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

// -------------------------------------------------
// -------------------------------------------------

template <typename T>
__global__ void MappedAvgPool2DWeightedKernel(
    const int n, const T *__restrict__ in_data_ptr,
    const T *__restrict__ sample_map_ptr,      // OH x OW x K x P x 2
    const T *__restrict__ interp_weights_ptr,  // OH x OW x K x P
    const int channels, const int in_height, const int in_width,
    const int out_height, const int out_width, const int kernel_size,
    const int interpolation, const int num_interp_pts,
    T *__restrict__ out_data_ptr) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedAvgPool2DWeighted(index, in_data_ptr, sample_map_ptr,
                                  interp_weights_ptr, channels, in_height,
                                  in_width, out_height, out_width, kernel_size,
                                  interpolation, num_interp_pts, out_data_ptr);
}

void MappedAvgPool2DWeightedLauncher(
    torch::Tensor in_data, torch::Tensor sample_map,
    torch::Tensor interp_weights, const int channels, const int in_height,
    const int in_width, const int out_height, const int out_width,
    const int kernel_size, const int interpolation, const int num_interp_pts,
    torch::Tensor out_data) {
  const int num_kernels = channels * out_height * out_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(
      in_data.scalar_type(), "MappedAvgPool2DWeightedLauncher", ([&] {
        MappedAvgPool2DWeightedKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, in_data.data<scalar_t>(), sample_map.data<scalar_t>(),
            interp_weights.data<scalar_t>(), channels, in_height, in_width,
            out_height, out_width, kernel_size, interpolation, num_interp_pts,
            out_data.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

template <typename T>
__global__ void MappedAvgUnpool2DWeightedKernel(
    const int n, const T *__restrict__ grad_output_ptr,
    const T *__restrict__ sample_map_ptr,
    const T *__restrict__ interp_weights_ptr, const int channels,
    const int orig_height, const int orig_width, const int pooled_height,
    const int pooled_width, const int kernel_size, const int interpolation,
    const int num_interp_pts, T *const grad_input_ptr) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedAvgUnpool2DWeighted(
      index, grad_output_ptr, sample_map_ptr, interp_weights_ptr, channels,
      orig_height, orig_width, pooled_height, pooled_width, kernel_size,
      interpolation, num_interp_pts, grad_input_ptr);
}

void MappedAvgUnpool2DWeightedLauncher(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, const int channels, const int orig_height,
    const int orig_width, const int pooled_height, const int pooled_width,
    const int kernel_size, const int interpolation, const int num_interp_pts,
    torch::Tensor grad_input) {
  const int num_kernels = channels * pooled_height * pooled_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "MappedAvgUnpool2DWeightedLauncher", ([&] {
        MappedAvgUnpool2DWeightedKernel<scalar_t>
            <<<blocks, CUDA_NUM_THREADS>>>(
                num_kernels, grad_output.data<scalar_t>(),
                sample_map.data<scalar_t>(), interp_weights.data<scalar_t>(),
                channels, orig_height, orig_width, pooled_height, pooled_width,
                kernel_size, interpolation, num_interp_pts,
                grad_input.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv
#endif