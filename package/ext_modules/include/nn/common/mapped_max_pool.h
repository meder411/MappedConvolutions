#ifndef COMMON_MAPPED_MAX_POOL_H_
#define COMMON_MAPPED_MAX_POOL_H_

#include <float.h>

#include "core/resample.h"
#include "core/util.h"

namespace mapped_conv {
namespace nn {
namespace common {

template <typename T>
__host__ __device__ void MappedMaxPool2D(
    const int index, const T *in_data_ptr,
    const T *sample_map_ptr,  // OH x OW x K x 2
    const int channels, const int in_height, const int in_width,
    const int out_height, const int out_width, const int kernel_size,
    const int interpolation, T *out_data_ptr,
    int64_t *out_idx_ptr)  // Indices of kernel sample in map
{
  // Determine current index in output image
  const int x_out = index % out_width;
  const int y_out = (index / out_width) % out_height;
  const int c     = (index / out_width / out_height) % channels;

  // Initialize the storage variables
#ifdef __CUDACC__
  T max_val = -FLT_MAX;
#else
  T max_val = std::numeric_limits<T>::min();
#endif
  int64_t max_idx = -1;

  // Starting points
  const int c_stride = c * in_height * in_width;
  const int sample_map_idx =
      y_out * out_width * kernel_size * 2 + x_out * kernel_size * 2;

  // Go through the kernel
  for (int kern_idx = 0; kern_idx < kernel_size; kern_idx++) {
    // Perform the interpolation of the input data
    const T val = core::ResampleFromMap2D(
        in_data_ptr + c_stride, sample_map_ptr + sample_map_idx + kern_idx * 2,
        in_height, in_width, interpolation);

    // Update the max value
    if (val > max_val) {
      max_val = val;
      max_idx = sample_map_idx + kern_idx * 2;
    }
  }
  // Store the max value in the output data tensor
  out_data_ptr[index] = max_val;

  // Also store the index in corresponding tensor for use in backprop
  out_idx_ptr[index] = max_idx;
}

template <typename T>
__host__ __device__ void MappedMaxUnpool2D(
    const int index, const T *grad_output_ptr, const int64_t *idx_mask_ptr,
    const T *sample_map_ptr, const int channels, const int orig_height,
    const int orig_width, const int pooled_height, const int pooled_width,
    const int kernel_size, const int interpolation, T *const grad_input_ptr) {
  // Determine current channel
  const int c        = (index / pooled_width / pooled_height) % channels;
  const int c_stride = c * orig_height * orig_width;

  // Current data
  const T value = grad_output_ptr[index];  // Gradient at this location
  const int sample_map_idx =
      idx_mask_ptr[index];  // Index in map where max value was

  // Perform uninterpolation
  core::ResampleToMap2D(value, sample_map_ptr + sample_map_idx, orig_height,
                        orig_width, interpolation, grad_input_ptr + c_stride);
}

template <typename T>
__host__ __device__ void MappedMaxPool2DWeighted(
    const int index, const T *in_data_ptr,
    const T *sample_map_ptr,      // OH x OW x K x P x 2
    const T *interp_weights_ptr,  // OH x OW x K x P
    const int channels, const int in_height, const int in_width,
    const int out_height, const int out_width, const int kernel_size,
    const int interpolation, const int num_interp_pts, T *out_data_ptr,
    int64_t *out_idx_ptr)  // Indices of kernel sample in map
{
  // Determine current index in output image
  const int x_out = index % out_width;
  const int y_out = (index / out_width) % out_height;
  const int c     = (index / out_width / out_height) % channels;

  // Initialize the storage variables
#ifdef __CUDACC__
  T max_val = -FLT_MAX;
#else
  T max_val = std::numeric_limits<T>::min();
#endif
  int64_t max_idx = -1;

  // Starting points
  const int c_stride = c * in_height * in_width;
  const int sample_map_idx =
      y_out * out_width * kernel_size * num_interp_pts * 2 +
      x_out * kernel_size * num_interp_pts * 2;
  const int interp_weights_idx =
      y_out * out_width * kernel_size * num_interp_pts +
      x_out * kernel_size * num_interp_pts;

  // Go through the kernel
  for (int kern_idx = 0; kern_idx < kernel_size; kern_idx++) {
    // Perform the interpolation of the input data
    const T val = core::ResampleFromMap2DWeighted(
        in_data_ptr + c_stride,
        sample_map_ptr + sample_map_idx + kern_idx * num_interp_pts * 2,
        interp_weights_ptr + interp_weights_idx + kern_idx * num_interp_pts,
        num_interp_pts, interpolation, in_height, in_width);

    // Update the max value
    if (val > max_val) {
      max_val = val;
      max_idx = sample_map_idx + kern_idx * num_interp_pts * 2;
    }
  }
  // Store the max value in the output data tensor
  out_data_ptr[index] = max_val;

  // Also store the index in corresponding tensor for use in backprop
  out_idx_ptr[index] = max_idx;
}

template <typename T>
__host__ __device__ void MappedMaxUnpool2DWeighted(
    const int index, const T *grad_output_ptr, const int64_t *idx_mask_ptr,
    const T *sample_map_ptr, const T *interp_weights_ptr, const int channels,
    const int orig_height, const int orig_width, const int pooled_height,
    const int pooled_width, const int kernel_size, const int interpolation,
    const int num_interp_pts, T *const grad_input_ptr) {
  // Current location in pooled image
  const int x_pool = index % pooled_width;
  const int y_pool = (index / pooled_width) % pooled_height;

  // Determine current channel
  const int c        = (index / pooled_width / pooled_height) % channels;
  const int c_stride = c * orig_height * orig_width;

  // Current data
  const T value = grad_output_ptr[index];  // Gradient at this location
  const int sample_map_idx =
      idx_mask_ptr[index];  // Index in map where max value was

  // Compute the kernel index from the sample index
  const int kern_idx =
      (sample_map_idx -
       y_pool * pooled_width * kernel_size * num_interp_pts * 2 -
       x_pool * kernel_size * num_interp_pts * 2) /
      (num_interp_pts * 2);
  const int interp_weights_idx =
      y_pool * pooled_width * kernel_size * num_interp_pts +
      x_pool * kernel_size * num_interp_pts + kern_idx * num_interp_pts;

  // Perform uninterpolation
  core::ResampleToMap2DWeighted(value, sample_map_ptr + sample_map_idx,
                                interp_weights_ptr + interp_weights_idx,
                                num_interp_pts, interpolation, orig_height,
                                orig_width, grad_input_ptr + c_stride);
}

}  // namespace common
}  // namespace nn
}  // namespace mapped_conv
#endif