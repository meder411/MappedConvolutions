#ifndef MAPPED_POOL_CUH_
#define MAPPED_POOL_CUH_

#include <ATen/ATen.h>

#include "core/resample.h"
#include "cuda_helper.h"
#include "nn/common/mapped_max_pool.h"

namespace mapped_conv {
namespace nn {
namespace cuda {

template <typename T>
__global__ void MappedMaxPool2DKernel(
    const int n, const T *__restrict__ in_data_ptr,
    const T *__restrict__ sample_map_ptr,  // OH x OW x K x 2
    const int channels, const int in_height, const int in_width,
    const int out_height, const int out_width, const int kernel_size,
    const int interpolation, T *__restrict__ out_data_ptr,
    int64_t
        *__restrict__ out_idx_ptr)  // Indices of kernel sample in sample_map
{
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedMaxPool2D(index, in_data_ptr, sample_map_ptr, channels,
                          in_height, in_width, out_height, out_width,
                          kernel_size, interpolation, out_data_ptr,
                          out_idx_ptr);
}

void MappedMaxPool2DLauncher(
    at::Tensor in_data, at::Tensor sample_map, const int channels,
    const int in_height, const int in_width, const int out_height,
    const int out_width, const int kernel_size, const int interpolation,
    at::Tensor out_data,
    at::Tensor out_idx)  // Indices of kernel samples in sample_map
{
  const int num_kernels = channels * out_height * out_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(
      in_data.type(), "MappedMaxPool2DLauncher", ([&] {
        MappedMaxPool2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, in_data.data<scalar_t>(), sample_map.data<scalar_t>(),
            channels, in_height, in_width, out_height, out_width, kernel_size,
            interpolation, out_data.data<scalar_t>(), out_idx.data<int64_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

template <typename T>
__global__ void MappedMaxUnpool2DKernel(
    const int n, const T *__restrict__ grad_output_ptr,
    const int64_t *__restrict__ idx_mask_ptr,
    const T *__restrict__ sample_map_ptr, const int channels,
    const int orig_height, const int orig_width, const int pooled_height,
    const int pooled_width, const int kernel_size, const int interpolation,
    T *const grad_input_ptr) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedMaxUnpool2D(index, grad_output_ptr, idx_mask_ptr,
                            sample_map_ptr, channels, orig_height, orig_width,
                            pooled_height, pooled_width, kernel_size,
                            interpolation, grad_input_ptr);
}

void MappedMaxUnpool2DLauncher(at::Tensor grad_output, at::Tensor idx_mask,
                               at::Tensor sample_map, const int channels,
                               const int orig_height, const int orig_width,
                               const int pooled_height, const int pooled_width,
                               const int kernel_size, const int interpolation,
                               at::Tensor grad_input) {
  const int num_kernels = channels * pooled_height * pooled_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "MappedMaxUnpool2DLauncher", ([&] {
        MappedMaxUnpool2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, grad_output.data<scalar_t>(),
            idx_mask.data<int64_t>(), sample_map.data<scalar_t>(), channels,
            orig_height, orig_width, pooled_height, pooled_width, kernel_size,
            interpolation, grad_input.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

// -------------------------------------------------
// -------------------------------------------------

template <typename T>
__global__ void MappedMaxPool2DWeightedKernel(
    const int n, const T *__restrict__ in_data_ptr,
    const T *__restrict__ sample_map_ptr,      // OH x OW x K x P x 2
    const T *__restrict__ interp_weights_ptr,  // OH x OW x K x P
    const int channels, const int in_height, const int in_width,
    const int out_height, const int out_width, const int kernel_size,
    const int interpolation, const int num_interp_pts,
    T *__restrict__ out_data_ptr,
    int64_t
        *__restrict__ out_idx_ptr)  // Indices of kernel sample in sample_map
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedMaxPool2DWeighted(
      index, in_data_ptr, sample_map_ptr, interp_weights_ptr, channels,
      in_height, in_width, out_height, out_width, kernel_size, interpolation,
      num_interp_pts, out_data_ptr, out_idx_ptr);
}

void MappedMaxPool2DWeightedLauncher(
    at::Tensor in_data, at::Tensor sample_map, at::Tensor interp_weights,
    const int channels, const int in_height, const int in_width,
    const int out_height, const int out_width, const int kernel_size,
    const int interpolation, const int num_interp_pts, at::Tensor out_data,
    at::Tensor out_idx)  // Indices of kernel samples in sample_map
{
  const int num_kernels = channels * out_height * out_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(
      in_data.type(), "MappedMaxPool2DWeightedLauncher", ([&] {
        MappedMaxPool2DWeightedKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, in_data.data<scalar_t>(), sample_map.data<scalar_t>(),
            interp_weights.data<scalar_t>(), channels, in_height, in_width,
            out_height, out_width, kernel_size, interpolation, num_interp_pts,
            out_data.data<scalar_t>(), out_idx.data<int64_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

template <typename T>
__global__ void MappedMaxUnpool2DWeightedKernel(
    const int n, const T *__restrict__ grad_output_ptr,
    const int64_t *__restrict__ idx_mask_ptr,
    const T *__restrict__ sample_map_ptr,
    const T *__restrict__ interp_weights_ptr, const int channels,
    const int orig_height, const int orig_width, const int pooled_height,
    const int pooled_width, const int kernel_size, const int interpolation,
    const int num_interp_pts, T *const grad_input_ptr) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedMaxUnpool2DWeighted(
      index, grad_output_ptr, idx_mask_ptr, sample_map_ptr, interp_weights_ptr,
      channels, orig_height, orig_width, pooled_height, pooled_width,
      kernel_size, interpolation, num_interp_pts, grad_input_ptr);
}

void MappedMaxUnpool2DWeightedLauncher(
    at::Tensor grad_output, at::Tensor idx_mask, at::Tensor sample_map,
    at::Tensor interp_weights, const int channels, const int orig_height,
    const int orig_width, const int pooled_height, const int pooled_width,
    const int kernel_size, const int interpolation, const int num_interp_pts,
    at::Tensor grad_input) {
  const int num_kernels = channels * pooled_height * pooled_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "MappedMaxUnpool2DWeightedLauncher", ([&] {
        MappedMaxUnpool2DWeightedKernel<scalar_t>
            <<<blocks, CUDA_NUM_THREADS>>>(
                num_kernels, grad_output.data<scalar_t>(),
                idx_mask.data<int64_t>(), sample_map.data<scalar_t>(),
                interp_weights.data<scalar_t>(), channels, orig_height,
                orig_width, pooled_height, pooled_width, kernel_size,
                interpolation, num_interp_pts, grad_input.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv
#endif