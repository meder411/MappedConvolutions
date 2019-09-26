#ifndef MAPPED_IM2COL_CUH_
#define MAPPED_IM2COL_CUH_

#include <torch/extension.h>

#include "cuda_helper.h"
#include "nn/common/mapped_im2col.h"

namespace mapped_conv {
namespace nn {
namespace cuda {

template <typename T>
__global__ void MappedIm2Col2DKernel(
    const int64_t num_kernels, const T *__restrict__ data_im_ptr,
    const T *__restrict__ sample_map_ptr, const int64_t height_im,
    const int64_t width_im, const int64_t width_out, const int64_t width_col,
    const int64_t kernel_size, const int64_t interpolation,
    T *__restrict__ data_col_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_kernels) { return; }

  common::MappedIm2Col2D(index, data_im_ptr, sample_map_ptr, height_im,
                         width_im, width_out, width_col, kernel_size,
                         interpolation, data_col_ptr);
}

void MappedIm2Col2DLauncher(torch::Tensor data_im,
                            torch::Tensor sample_map,  // OH, OW, KH, KW, 2
                            const int64_t channels, const int64_t height_im,
                            const int64_t width_im, const int64_t width_out,
                            const int64_t width_col, const int64_t kernel_size,
                            const int64_t interpolation,
                            torch::Tensor data_col) {
  const int64_t num_kernels = channels * width_col;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // Launch channels * width_col kernels, with each kernel responsible for
  // copying a the convolutions over a single channel.
  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "MappedIm2Col2DLauncher", ([&] {
        MappedIm2Col2DKernel<<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data<scalar_t>(), sample_map.data<scalar_t>(),
            height_im, width_im, width_out, width_col, kernel_size,
            interpolation, data_col.data<scalar_t>());
        CUDA_CHECK(cudaGetLastError())
      }));
}

template <typename T>
__global__ void MappedCol2Im2DKernel(
    const int64_t n, const T *__restrict__ data_col_ptr,
    const T *__restrict__ sample_map_ptr,  // OH, OW, K, 2
    const int64_t height_im, const int64_t width_im, const int64_t width_out,
    const int64_t width_col, const int64_t kernel_size,
    const int64_t interpolation, T *__restrict__ data_im_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedCol2Im2D(index, data_col_ptr,
                         sample_map_ptr,  // OH, OW, K, 2
                         height_im, width_im, width_out, width_col,
                         kernel_size, interpolation, data_im_ptr);
}

void MappedCol2Im2DLauncher(torch::Tensor data_col,
                            torch::Tensor sample_map,  // OH, OW, KH, KW, 2
                            const int64_t channels, const int64_t height_im,
                            const int64_t width_im, const int64_t width_out,
                            const int64_t width_col, const int64_t kernel_size,
                            const int64_t interpolation,
                            torch::Tensor data_im) {
  const int64_t num_kernels = channels * width_col;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // Launch with the same number of kernels as with im2col. This differs
  // from the standard col2im which determines kernels to launch as a
  // function of input image dimensions. We use the output image dimensions
  // because we determine the input image locations from the mapping, rather
  // than a search over the kernel extent.
  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "MappedCol2Im2DLauncher", ([&] {
        MappedCol2Im2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_col.data<scalar_t>(),
            sample_map.data<scalar_t>(), height_im, width_im, width_out,
            width_col, kernel_size, interpolation, (data_im.data<scalar_t>()));
      }));
  CUDA_CHECK(cudaGetLastError())
}

// -----------------------------------------------------------
// -----------------------------------------------------------

template <typename T>
__global__ void MappedIm2Col2DWeightedKernel(
    const int64_t num_kernels, const T *__restrict__ data_im_ptr,
    const T *__restrict__ sample_map_ptr,
    const T *__restrict__ interp_weights_ptr, const int64_t height_im,
    const int64_t width_im, const int64_t width_out, const int64_t width_col,
    const int64_t kernel_size, const int64_t interpolation,
    const int64_t num_interp_pts, T *__restrict__ data_col_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_kernels) { return; }

  common::MappedIm2Col2DWeighted(index, data_im_ptr, sample_map_ptr,
                                 interp_weights_ptr, height_im, width_im,
                                 width_out, width_col, kernel_size,
                                 interpolation, num_interp_pts, data_col_ptr);
}

void MappedIm2Col2DWeightedLauncher(
    torch::Tensor data_im,
    torch::Tensor sample_map,      // OH, OW, KH, KW, P, 2
    torch::Tensor interp_weights,  // OH, OW, KH, KW, P
    const int64_t channels, const int64_t height_im, const int64_t width_im,
    const int64_t width_out, const int64_t width_col,
    const int64_t kernel_size, const int64_t interpolation,
    const int64_t num_interp_pts, torch::Tensor data_col) {
  const int64_t num_kernels = channels * width_col;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // Launch channels * width_col kernels, with each kernel responsible for
  // copying a the convolutions over a single channel.
  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "MappedIm2Col2DWeightedLauncher", ([&] {
        MappedIm2Col2DWeightedKernel<<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data<scalar_t>(), sample_map.data<scalar_t>(),
            interp_weights.data<scalar_t>(), height_im, width_im, width_out,
            width_col, kernel_size, interpolation, num_interp_pts,
            data_col.data<scalar_t>());
        CUDA_CHECK(cudaGetLastError())
      }));
}

template <typename T>
__global__ void MappedCol2Im2DWeightedKernel(
    const int64_t n, const T *__restrict__ data_col_ptr,
    const T *__restrict__ sample_map_ptr,      // OH, OW, K, P, 2
    const T *__restrict__ interp_weights_ptr,  // OH, OW, K, P
    const int64_t height_im, const int64_t width_im, const int64_t width_out,
    const int64_t width_col, const int64_t kernel_size,
    const int64_t interpolation, const int64_t num_interp_pts,
    T *__restrict__ data_im_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::MappedCol2Im2DWeighted(index, data_col_ptr,
                                 sample_map_ptr,      // OH, OW, K, P, 2
                                 interp_weights_ptr,  // OH, OW, K, P
                                 height_im, width_im, width_out, width_col,
                                 kernel_size, interpolation, num_interp_pts,
                                 data_im_ptr);
}

void MappedCol2Im2DWeightedLauncher(
    torch::Tensor data_col,
    torch::Tensor sample_map,      // OH, OW, KH, KW, P, 2
    torch::Tensor interp_weights,  // OH, OW, KH, KW, P
    const int64_t channels, const int64_t height_im, const int64_t width_im,
    const int64_t width_out, const int64_t width_col,
    const int64_t kernel_size, const int64_t interpolation,
    const int64_t num_interp_pts, torch::Tensor data_im) {
  const int64_t num_kernels = channels * width_col;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // Launch with the same number of kernels as with im2col. This differs
  // from the standard col2im which determines kernels to launch as a
  // function of input image dimensions. We use the output image dimensions
  // because we determine the input image locations from the mapping, rather
  // than a search over the kernel extent.
  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "MappedCol2Im2DWeightedLauncher", ([&] {
        MappedCol2Im2DWeightedKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_col.data<scalar_t>(),
            sample_map.data<scalar_t>(), interp_weights.data<scalar_t>(),
            height_im, width_im, width_out, width_col, kernel_size,
            interpolation, num_interp_pts, (data_im.data<scalar_t>()));
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv
#endif