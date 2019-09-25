#ifndef RESAMPLE_CUH_
#define RESAMPLE_CUH_

#include <ATen/ATen.h>

#include "cuda_helper.h"
#include "nn/common/resample.h"

namespace mapped_conv {
namespace nn {
namespace cuda {

template <typename T>
__global__ void ResampleToMap2DKernel(
    const int64_t n, const T *__restrict__ data_in_ptr,
    const T *__restrict__ sample_map_ptr, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation,
    T *__restrict__ data_out_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleToMap2D(index, data_in_ptr, sample_map_ptr, channels,
                          in_height, in_width, out_height, out_width,
                          interpolation, data_out_ptr);
}

void ResampleToMap2DLauncher(at::Tensor data_in,
                             at::Tensor sample_map,  // IH, IW, 2
                             const int64_t channels, const int64_t in_height,
                             const int64_t in_width, const int64_t out_height,
                             const int64_t out_width,
                             const int64_t interpolation,
                             at::Tensor data_out) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.type(), "ResampleToMap2DKernel", ([&] {
        ResampleToMap2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_in.data<scalar_t>(), sample_map.data<scalar_t>(),
            channels, in_height, in_width, out_height, out_width,
            interpolation, data_out.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

template <typename T>
__global__ void ResampleFromMap2DKernel(
    const int64_t n, const T *__restrict__ data_out_ptr,
    const T *__restrict__ sample_map_ptr,  // IH, IW, 2
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, T *__restrict__ data_in_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleFromMap2D(index, data_out_ptr, sample_map_ptr, channels,
                            in_height, in_width, out_height, out_width,
                            interpolation, data_in_ptr);
}

void ResampleFromMap2DLauncher(
    at::Tensor data_out, at::Tensor sample_map, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation, at::Tensor data_in) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.type(), "ResampleFromMap2DKernel", ([&] {
        ResampleFromMap2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_out.data<scalar_t>(),
            sample_map.data<scalar_t>(), channels, in_height, in_width,
            out_height, out_width, interpolation, data_in.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

// ----------------------------------------------
// ----------------------------------------------

template <typename T>
__global__ void ResampleToMap2DWeightedKernel(
    const int64_t n, const T *__restrict__ data_in_ptr,
    const T *__restrict__ sample_map_ptr,
    const T *__restrict__ interp_weights_ptr, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation,
    const int64_t num_interp_pts, T *__restrict__ data_out_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleToMap2DWeighted(index, data_in_ptr, sample_map_ptr,
                                  interp_weights_ptr, channels, in_height,
                                  in_width, out_height, out_width,
                                  interpolation, num_interp_pts, data_out_ptr);
}

void ResampleToMap2DWeightedLauncher(
    at::Tensor data_in,
    at::Tensor sample_map,      // IH, IW, P, 2
    at::Tensor interp_weights,  // IH, IW, P
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, const int64_t num_interp_pts,
    at::Tensor data_out) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.type(), "ResampleToMap2DWeightedKernel", ([&] {
        ResampleToMap2DWeightedKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_in.data<scalar_t>(), sample_map.data<scalar_t>(),
            interp_weights.data<scalar_t>(), channels, in_height, in_width,
            out_height, out_width, interpolation, num_interp_pts,
            data_out.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

template <typename T>
__global__ void ResampleFromMap2DWeightedKernel(
    const int64_t n, const T *__restrict__ data_out_ptr,
    const T *__restrict__ sample_map_ptr,      // IH, IW, P, 2
    const T *__restrict__ interp_weights_ptr,  // IH, IW, P
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, const int64_t num_interp_pts,
    T *__restrict__ data_in_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleFromMap2DWeighted(
      index, data_out_ptr, sample_map_ptr, interp_weights_ptr, channels,
      in_height, in_width, out_height, out_width, interpolation,
      num_interp_pts, data_in_ptr);
}

void ResampleFromMap2DWeightedLauncher(
    at::Tensor data_out, at::Tensor sample_map, at::Tensor interp_weights,
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, const int64_t num_interp_pts,
    at::Tensor data_in) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.type(), "ResampleFromMap2DWeightedKernel", ([&] {
        ResampleFromMap2DWeightedKernel<scalar_t>
            <<<blocks, CUDA_NUM_THREADS>>>(
                num_kernels, data_out.data<scalar_t>(),
                sample_map.data<scalar_t>(), interp_weights.data<scalar_t>(),
                channels, in_height, in_width, out_height, out_width,
                interpolation, num_interp_pts, data_in.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

// ----------------------------------------------
// ----------------------------------------------

template <typename T>
__global__ void ResampleToMap2DVotingKernel(
    const int64_t n, const T *__restrict__ data_in_ptr,
    const T *__restrict__ sample_map_ptr, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t numCandidates,
    T *__restrict__ data_out_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleToMap2DVoting(index, data_in_ptr, sample_map_ptr, channels,
                                in_height, in_width, out_height, out_width,
                                numCandidates, data_out_ptr);
}

void ResampleToMap2DVotingLauncher(
    at::Tensor data_in,
    at::Tensor sample_map,  // IH, IW, P, 2
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t numCandidates, at::Tensor data_out) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  ResampleToMap2DVotingKernel<<<blocks, CUDA_NUM_THREADS>>>(
      num_kernels, data_in.data<int64_t>(), sample_map.data<int64_t>(),
      channels, in_height, in_width, out_height, out_width, numCandidates,
      data_out.data<int64_t>());
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv
#endif