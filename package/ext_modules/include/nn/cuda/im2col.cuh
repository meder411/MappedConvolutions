#ifndef MAPPED_IM2COL_CUH_
#define MAPPED_IM2COL_CUH_

#include <torch/extension.h>

#include "cuda_helper.h"
#include "nn/common/im2col.h"

namespace mapped_conv {
namespace nn {
namespace cuda {

template <typename T>
__global__ void Im2Col2DKernel(
    const int64_t num_kernels, const T *data_im_ptr, const int64_t height_im,
    const int64_t width_im, const int64_t width_out, const int64_t width_col,
    const int64_t kernel_h, const int64_t kernel_w, const int64_t pad_h,
    const int64_t pad_w, const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w, T *data_col_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_kernels) { return; }

  common::Im2Col2D(index, data_im_ptr, height_im, width_im, width_out,
                   width_col, kernel_h, kernel_w, pad_h, pad_w, stride_h,
                   stride_w, dilation_h, dilation_w, data_col_ptr);
}

void Im2Col2DLauncher(torch::Tensor data_im, const int64_t channels,
                      const int64_t height_im, const int64_t width_im,
                      const int64_t width_out, const int64_t width_col,
                      const int64_t kernel_h, const int64_t kernel_w,
                      const int64_t pad_h, const int64_t pad_w,
                      const int64_t stride_h, const int64_t stride_w,
                      const int64_t dilation_h, const int64_t dilation_w,
                      torch::Tensor data_col) {
  const int64_t num_kernels = channels * width_col;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // Launch channels * width_col kernels, with each kernel responsible for
  // copying a the convolutions over a single channel.
  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "Im2Col2DKernel", ([&] {
        Im2Col2DKernel<<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data<scalar_t>(), height_im, width_im,
            width_out, width_col, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, data_col.data<scalar_t>());
        CUDA_CHECK(cudaGetLastError())
      }));
}

template <typename T>
__global__ void Col2Im2DKernel(
    const int64_t num_kernels, const T *data_col_ptr, const int64_t height,
    const int64_t width, const int64_t output_height,
    const int64_t output_width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w, const int64_t stride_h,
    const int64_t stride_w, const int64_t dilation_h, const int64_t dilation_w,
    T *data_im_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_kernels) { return; }

  common::Col2Im2D(index, data_col_ptr, height, width, output_height,
                   output_width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
                   stride_w, dilation_h, dilation_w, data_im_ptr);
}

void Col2Im2DLauncher(torch::Tensor data_col, const int64_t channels,
                      const int64_t height, const int64_t width,
                      const int64_t output_height, const int64_t output_width,
                      const int64_t kernel_h, const int64_t kernel_w,
                      const int64_t pad_h, const int64_t pad_w,
                      const int64_t stride_h, const int64_t stride_w,
                      const int64_t dilation_h, const int64_t dilation_w,
                      torch::Tensor data_im) {
  // Launching num_kernels == output image dimension
  // Avoids need for atomic ops
  const int64_t num_kernels = channels * height * width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // Launch with the same number of kernels as with im2col. This differs

  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "Col2Im2DKernel", ([&] {
        Col2Im2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_col.data<scalar_t>(), height, width,
            output_height, output_width, kernel_h, kernel_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            (data_im.data<scalar_t>()));
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv
#endif