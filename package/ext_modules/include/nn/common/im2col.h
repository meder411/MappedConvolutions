#ifndef COMMON_IM2COL_H_
#define COMMON_IM2COL_H_

#include "core/util.h"

#include <algorithm>

namespace mapped_conv {
namespace nn {
namespace common {

template <typename T>
__host__ __device__ void Im2Col2D(
    const int64_t index, const T *data_im_ptr, const int64_t height_im,
    const int64_t width_im, const int64_t width_out, const int64_t width_col,
    const int64_t kernel_h, const int64_t kernel_w, const int64_t pad_h,
    const int64_t pad_w, const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w, T *data_col_ptr) {
  // Location in the flattened image
  const int64_t x_col = index % width_col;

  // Output image location depends on location in row in col matrix (x_col)
  const int64_t x_out = x_col % width_out;
  const int64_t y_out = x_col / width_out;

  // Input image location is a function of the stride and padding
  // Result is the location of the top-left of the convolutional kernel
  const int64_t x_im = stride_w * x_out - pad_w;
  const int64_t y_im = stride_h * y_out - pad_h;
  const int64_t c_im = index / width_col;

  // Location in column matrix
  int64_t index_col = c_im * (kernel_h * kernel_w) * width_col + x_col;

  // Determine sampling location in the image and fill in all kernels
  for (int64_t y_kern = 0; y_kern < kernel_h; y_kern++) {
    const int64_t y = y_im + y_kern * dilation_h;
    for (int64_t x_kern = 0; x_kern < kernel_w; x_kern++) {
      const int64_t x = x_im + x_kern * dilation_w;

      // Perform the copy
      data_col_ptr[index_col] =
          (x >= 0 && y >= 0 && x < width_im && y < height_im)
              ? data_im_ptr[c_im * height_im * width_im + y * width_im + x]
              : T(0);

      // Move the pointer to the next row
      index_col += width_col;
    }
  }
}

template <typename T>
__host__ __device__ void Col2Im2D(
    const int64_t index, const T *data_col_ptr, const int64_t height,
    const int64_t width, const int64_t output_height,
    const int64_t output_width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w, const int64_t stride_h,
    const int64_t stride_w, const int64_t dilation_h, const int64_t dilation_w,
    T *data_im_ptr) {
  // Accumulator
  T val = 0;

  // Location in input image
  const int64_t x_im = index % width + pad_w;
  const int64_t y_im = (index / width) % height + pad_h;
  const int64_t c_im = index / (width * height);

  // Get the effective size of the kernel including dilation
  // We assume that the image pixel will fall inside the kernel at some point
  int64_t kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
  int64_t kernel_extent_h = (kernel_h - 1) * dilation_h + 1;

  // Output image location
  // The region computed here is all of the output image features that use
  // a given input image pixel (x_im, y_im) in its computation. That is,
  // if the kernel covers (x_im, y_im) when computing some (x_out, y_out).
  // This is the coverage region. Start is the leftmost col or topmost row
  // covered when the current image pixel falls at the rightmost col or
  // bottommost row. End is the rightmost col or bottommost row covered
  // when the image pixel falls at the leftmost col or bottommost row.
  const int64_t x_out_start =
      (x_im < kernel_extent_w) ? 0 : (x_im - kernel_extent_w) / stride_w + 1;
  const int64_t y_out_start =
      (y_im < kernel_extent_h) ? 0 : (y_im - kernel_extent_h) / stride_h + 1;
#ifdef __CUDACC__  // CUDA compilation only
  const int64_t x_out_end = min(x_im / stride_w + 1, output_width);
  const int64_t y_out_end = min(y_im / stride_h + 1, output_height);
#else
  const int64_t x_out_end = std::min(x_im / stride_w + 1, output_width);
  const int64_t y_out_end = std::min(y_im / stride_h + 1, output_height);
#endif

  // Go over the output pixels that use our current input pixel information
  // (the current input pixel is given by the kernel)
  for (int64_t y_out = y_out_start; y_out < y_out_end; y_out += 1) {
    for (int64_t c_out = x_out_start; c_out < x_out_end; c_out += 1) {
      // Get the location in the kernel
      int64_t y_k = (y_im - y_out * stride_h);
      int64_t x_k = (x_im - c_out * stride_w);

      // Check if there is a sample here or if it's a hole (due to
      // dilation)
      if (y_k % dilation_h == 0 && x_k % dilation_w == 0) {
        y_k /= dilation_h;
        x_k /= dilation_w;

        // Sample from the column matrix
        int64_t data_col_index =
            (((c_im * kernel_h + y_k) * kernel_w + x_k) * output_height +
             y_out) *
                output_width +
            c_out;

        // Accumulate the data from the column matrix
        val += data_col_ptr[data_col_index];
      }
    }
  }

  // After accumulating over the entire region, assign that value to the
  // input gradient
  data_im_ptr[index] = val;
}

}  // namespace common
}  // namespace nn
}  // namespace mapped_conv
#endif