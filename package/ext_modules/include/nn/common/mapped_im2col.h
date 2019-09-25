#ifndef COMMON_MAPPED_IM2COL_H_
#define COMMON_MAPPED_IM2COL_H_

#include "core/resample.h"
#include "core/util.h"

namespace mapped_conv {
namespace nn {
namespace common {

template <typename T>
__host__ __device__ void MappedIm2Col2D(
    const int64_t index, const T *data_im_ptr, const T *sample_map_ptr,
    const int64_t height_im, const int64_t width_im, const int64_t width_out,
    const int64_t width_col, const int64_t kernel_size,
    const int64_t interpolation, T *data_col_ptr) {
  // Location in the flattened image
  const int64_t x_col = index % width_col;

  // Output image location depends on location in row in col matrix (x_col)
  const int64_t x_out = x_col % width_out;
  const int64_t y_out = x_col / width_out;

  // Current channel
  const int64_t c_im = index / width_col;

  // Compute the starting locations outside the loop
  const int64_t c_im_stride = c_im * height_im * width_im;
  const int64_t sample_map_idx =
      y_out * width_out * kernel_size * 2 + x_out * kernel_size * 2;
  int64_t index_col = c_im * kernel_size * width_col + x_col;
  for (int64_t kern_idx = 0; kern_idx < kernel_size; kern_idx++) {
    // Perform the interpolation of the input data
    data_col_ptr[index_col] =
        core::ResampleFromMap2D(data_im_ptr + c_im_stride,
                                sample_map_ptr + sample_map_idx + kern_idx * 2,
                                height_im, width_im, interpolation);

    // Move the pointer to the next row
    index_col += width_col;
  }
}

template <typename T>
__host__ __device__ void MappedCol2Im2D(
    const int64_t index, const T *data_col_ptr,
    const T *sample_map_ptr,  // OH, OW, K, 2
    const int64_t height_im, const int64_t width_im, const int64_t width_out,
    const int64_t width_col, const int64_t kernel_size,
    const int64_t interpolation, T *data_im_ptr) {
  // Location in the flattened image
  const int64_t x_col = index % width_col;

  // Output image location depends on location in row in col matrix (x_col)
  const int64_t x_out = x_col % width_out;
  const int64_t y_out = x_col / width_out;

  // Current channel
  const int64_t c_im = index / width_col;

  // Compute the starting locations outside the loop
  const int64_t c_im_stride = c_im * height_im * width_im;
  const int64_t sample_map_idx =
      y_out * width_out * kernel_size * 2 + x_out * kernel_size * 2;
  int64_t index_col = c_im * kernel_size * width_col + x_col;
  for (int64_t kern_idx = 0; kern_idx < kernel_size; kern_idx++) {
    // Grab the relevant element from the column matrix
    const T data = data_col_ptr[index_col];

    // Perform uninterpolation
    core::ResampleToMap2D(data, sample_map_ptr + sample_map_idx + kern_idx * 2,
                          height_im, width_im, interpolation,
                          data_im_ptr + c_im_stride);

    // Move the column index to the next row
    index_col += width_col;
  }
}

template <typename T>
__host__ __device__ void MappedIm2Col2DWeighted(
    const int64_t index, const T *data_im_ptr, const T *sample_map_ptr,
    const T *interp_weights_ptr, const int64_t height_im,
    const int64_t width_im, const int64_t width_out, const int64_t width_col,
    const int64_t kernel_size, const int64_t interpolation,
    const int64_t num_interp_pts, T *data_col_ptr) {
  // Location in the flattened image
  const int64_t x_col = index % width_col;

  // Output image location depends on location in row in col matrix (x_col)
  const int64_t x_out = x_col % width_out;
  const int64_t y_out = x_col / width_out;

  // Current channel
  const int64_t c_im = index / width_col;

  // Compute the starting points outside the loop
  const int64_t c_im_stride = c_im * height_im * width_im;
  const int64_t sample_map_idx =
      y_out * width_out * kernel_size * num_interp_pts * 2 +
      x_out * kernel_size * num_interp_pts * 2;
  const int64_t interp_weights_idx =
      y_out * width_out * kernel_size * num_interp_pts +
      x_out * kernel_size * num_interp_pts;
  int64_t index_col = c_im * kernel_size * width_col + x_col;
  for (int64_t kern_idx = 0; kern_idx < kernel_size; kern_idx++) {
    // Perform the interpolation of the input data
    data_col_ptr[index_col] = core::ResampleFromMap2DWeighted(
        data_im_ptr + c_im_stride,
        sample_map_ptr + sample_map_idx + kern_idx * num_interp_pts * 2,
        interp_weights_ptr + interp_weights_idx + kern_idx * num_interp_pts,
        num_interp_pts, interpolation, height_im, width_im);

    // Move the pointer to the next row
    index_col += width_col;
  }
}

template <typename T>
__host__ __device__ void MappedCol2Im2DWeighted(
    const int64_t index, const T *data_col_ptr, const T *sample_map_ptr,
    const T *interp_weights_ptr, const int64_t height_im,
    const int64_t width_im, const int64_t width_out, const int64_t width_col,
    const int64_t kernel_size, const int64_t interpolation,
    const int64_t num_interp_pts, T *data_im_ptr) {
  // Location in the flattened image
  const int64_t x_col = index % width_col;

  // Output image location depends on location in row in col matrix (x_col)
  const int64_t x_out = x_col % width_out;
  const int64_t y_out = x_col / width_out;

  // Current channel
  const int64_t c_im = index / width_col;

  // Compute the starting points outside the loop
  const int64_t c_im_stride = c_im * height_im * width_im;
  const int64_t sample_map_idx =
      y_out * width_out * kernel_size * num_interp_pts * 2 +
      x_out * kernel_size * num_interp_pts * 2;
  const int64_t interp_weights_idx =
      y_out * width_out * kernel_size * num_interp_pts +
      x_out * kernel_size * num_interp_pts;
  int64_t index_col = c_im * kernel_size * width_col + x_col;
  for (int64_t kern_idx = 0; kern_idx < kernel_size; kern_idx++) {
    // Grab the relevant element from the column matrix
    const T data = data_col_ptr[index_col];

    // Perform uninterpolation
    core::ResampleToMap2DWeighted(
        data, sample_map_ptr + sample_map_idx + kern_idx * num_interp_pts * 2,
        interp_weights_ptr + interp_weights_idx + kern_idx * num_interp_pts,
        num_interp_pts, interpolation, height_im, width_im,
        data_im_ptr + c_im_stride);

    // Move the column index to the next row
    index_col += width_col;
  }
}

}  // namespace common
}  // namespace nn
}  // namespace mapped_conv
#endif