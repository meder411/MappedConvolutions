#ifndef MAPPED_MAX_POOL_H_
#define MAPPED_MAX_POOL_H_

#include <math.h>
#include <omp.h>
#include <torch/extension.h>
#include <limits>

#include "core/resample.h"
#include "nn/common/mapped_max_pool.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

template <typename T>
void MappedMaxPool2D(const int num_kernels, torch::Tensor in_data,
                     torch::Tensor sample_map,  // OH x OW x K x 2
                     const int channels, const int in_height,
                     const int in_width, const int out_height,
                     const int out_width, const int kernel_size,
                     const int interpolation, torch::Tensor out_data,
                     torch::Tensor out_idx)  // Indices of kernel sample in map
{
  const T *in_data_ptr    = in_data.data<T>();
  const T *sample_map_ptr = sample_map.data<T>();
  T *out_data_ptr         = out_data.data<T>();
  int64_t *out_idx_ptr    = out_idx.data<int64_t>();
  int index;
#pragma omp parallel for shared(in_data_ptr, sample_map_ptr, out_data_ptr, \
                                out_idx_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::MappedMaxPool2D(index, in_data_ptr, sample_map_ptr, channels,
                            in_height, in_width, out_height, out_width,
                            kernel_size, interpolation, out_data_ptr,
                            out_idx_ptr);
  }
}

template <typename T>
void MappedMaxUnpool2D(const int num_kernels, torch::Tensor grad_output,
                       torch::Tensor idx_mask, torch::Tensor sample_map,
                       const int channels, const int orig_height,
                       const int orig_width, const int pooled_height,
                       const int pooled_width, const int kernel_size,
                       const int interpolation, torch::Tensor grad_input) {
  const T *grad_output_ptr    = grad_output.data<T>();
  const int64_t *idx_mask_ptr = idx_mask.data<int64_t>();
  const T *sample_map_ptr     = sample_map.data<T>();
  T *grad_input_ptr           = grad_input.data<T>();
  int index;
#pragma omp parallel for shared(                   \
    grad_output_ptr, idx_mask_ptr, sample_map_ptr, \
    grad_input_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::MappedMaxUnpool2D(index, grad_output_ptr, idx_mask_ptr,
                              sample_map_ptr, channels, orig_height,
                              orig_width, pooled_height, pooled_width,
                              kernel_size, interpolation, grad_input_ptr);
  }
}

// -------------------------------------------------
// -------------------------------------------------

template <typename T>
void MappedMaxPool2DWeighted(
    const int num_kernels, torch::Tensor in_data,
    torch::Tensor sample_map,      // OH x OW x K x P x 2
    torch::Tensor interp_weights,  // OH x OW x K x P
    const int channels, const int in_height, const int in_width,
    const int out_height, const int out_width, const int kernel_size,
    const int interpolation, const int num_interp_pts, torch::Tensor out_data,
    torch::Tensor out_idx)  // Indices of kernel sample in map
{
  const T *in_data_ptr        = in_data.data<T>();
  const T *sample_map_ptr     = sample_map.data<T>();
  const T *interp_weights_ptr = interp_weights.data<T>();
  T *out_data_ptr             = out_data.data<T>();
  int64_t *out_idx_ptr        = out_idx.data<int64_t>();
  int index;
#pragma omp parallel for shared(in_data_ptr, sample_map_ptr,      \
                                interp_weights_ptr, out_data_ptr, \
                                out_idx_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::MappedMaxPool2DWeighted(
        index, in_data_ptr, sample_map_ptr, interp_weights_ptr, channels,
        in_height, in_width, out_height, out_width, kernel_size, interpolation,
        num_interp_pts, out_data_ptr, out_idx_ptr);
  }
}

template <typename T>
void MappedMaxUnpool2DWeighted(
    const int num_kernels, torch::Tensor grad_output, torch::Tensor idx_mask,
    torch::Tensor sample_map, torch::Tensor interp_weights, const int channels,
    const int orig_height, const int orig_width, const int pooled_height,
    const int pooled_width, const int kernel_size, const int interpolation,
    const int num_interp_pts, torch::Tensor grad_input) {
  const T *grad_output_ptr    = grad_output.data<T>();
  const int64_t *idx_mask_ptr = idx_mask.data<int64_t>();
  const T *sample_map_ptr     = sample_map.data<T>();
  const T *interp_weights_ptr = interp_weights.data<T>();
  T *grad_input_ptr           = grad_input.data<T>();
  int index;
#pragma omp parallel for shared(                                       \
    grad_output_ptr, idx_mask_ptr, sample_map_ptr, interp_weights_ptr, \
    grad_input_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::MappedMaxUnpool2DWeighted(
        index, grad_output_ptr, idx_mask_ptr, sample_map_ptr,
        interp_weights_ptr, channels, orig_height, orig_width, pooled_height,
        pooled_width, kernel_size, interpolation, num_interp_pts,
        grad_input_ptr);
  }
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv
#endif