#ifndef MAPPED_IM2COL_H_
#define MAPPED_IM2COL_H_

#include <omp.h>
#include <torch/extension.h>

#include "nn/common/mapped_im2col.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

template <typename T>
void MappedIm2Col2D(const int64_t num_kernels, torch::Tensor data_im,
                    torch::Tensor sample_map,  //  OH, OW, K, Z
                    const int64_t height_im, const int64_t width_im,
                    const int64_t width_out, const int64_t width_col,
                    const int64_t kernel_size, const int64_t interpolation,
                    torch::Tensor data_col) {
  T *data_col_ptr         = data_col.data<T>();
  const T *data_im_ptr    = data_im.data<T>();
  const T *sample_map_ptr = sample_map.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_col_ptr, sample_map_ptr, \
                                data_im_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::MappedIm2Col2D(index, data_im_ptr, sample_map_ptr, height_im,
                           width_im, width_out, width_col, kernel_size,
                           interpolation, data_col_ptr);
  }
}

template <typename T>
void MappedCol2Im2D(const int64_t num_kernels, torch::Tensor data_col,
                    torch::Tensor sample_map,  // OH, OW, kernel_size, 2
                    const int64_t height_im, const int64_t width_im,
                    const int64_t width_out, const int64_t width_col,
                    const int64_t kernel_size, const int64_t interpolation,
                    torch::Tensor data_im) {
  const T *data_col_ptr   = data_col.data<T>();
  const T *sample_map_ptr = sample_map.data<T>();
  T *data_im_ptr          = data_im.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_col_ptr, sample_map_ptr, \
                                data_im_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::MappedCol2Im2D(index, data_col_ptr,
                           sample_map_ptr,  // OH, OW, K, 2
                           height_im, width_im, width_out, width_col,
                           kernel_size, interpolation, data_im_ptr);
  }
}

// -----------------------------------------------------------
// -----------------------------------------------------------

template <typename T>
void MappedIm2Col2DWeighted(
    const int64_t num_kernels, torch::Tensor data_im,
    torch::Tensor sample_map,      // OH, OW, kernel_size, P, 2
    torch::Tensor interp_weights,  // OH, OW, kernel_size, P
    const int64_t height_im, const int64_t width_im, const int64_t width_out,
    const int64_t width_col, const int64_t kernel_size,
    const int64_t interpolation, const int64_t num_interp_pts,
    torch::Tensor data_col) {
  T *data_col_ptr             = data_col.data<T>();
  const T *data_im_ptr        = data_im.data<T>();
  const T *sample_map_ptr     = sample_map.data<T>();
  const T *interp_weights_ptr = interp_weights.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_col_ptr, sample_map_ptr, \
                                interp_weights_ptr,           \
                                data_im_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::MappedIm2Col2DWeighted(
        index, data_im_ptr, sample_map_ptr, interp_weights_ptr, height_im,
        width_im, width_out, width_col, kernel_size, interpolation,
        num_interp_pts, data_col_ptr);
  }
}

template <typename T>
void MappedCol2Im2DWeighted(
    const int64_t num_kernels, torch::Tensor data_col,
    torch::Tensor sample_map,      // OH, OW, kernel_size, P, 2
    torch::Tensor interp_weights,  // OH, OW, kernel_size, P
    const int64_t height_im, const int64_t width_im, const int64_t width_out,
    const int64_t width_col, const int64_t kernel_size,
    const int64_t interpolation, const int64_t num_interp_pts,
    torch::Tensor data_im) {
  const T *data_col_ptr       = data_col.data<T>();
  const T *sample_map_ptr     = sample_map.data<T>();
  const T *interp_weights_ptr = interp_weights.data<T>();
  T *data_im_ptr              = data_im.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_col_ptr, sample_map_ptr, \
                                interp_weights_ptr,           \
                                data_im_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::MappedCol2Im2DWeighted(index, data_col_ptr,
                                   sample_map_ptr,      // OH, OW, K, P, 2
                                   interp_weights_ptr,  // OH, OW, K, P
                                   height_im, width_im, width_out, width_col,
                                   kernel_size, interpolation, num_interp_pts,
                                   data_im_ptr);
  }
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv
#endif