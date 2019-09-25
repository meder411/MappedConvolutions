#ifndef CUBE2RECT_CUH_
#define CUBE2RECT_CUH_

#include <ATen/ATen.h>
#include <math.h>

#include "core/conversions.h"
#include "core/interpolation.h"
#include "cuda_helper.h"

namespace mapped_conv {
namespace util {
namespace cuda {

template <typename T>
__global__ void Cube2RectKernel(
    const int64_t n, const T *__restrict__ data_cube, const int64_t cube_dim,
    const int64_t rect_height, const int64_t rect_width,
    const int64_t channels, const int64_t interpolation,
    T *__restrict__ data_rect) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  // Location in the rectangular image
  const int64_t x_rect = index % rect_width;
  const int64_t y_rect = (index / rect_width) % rect_height;
  const int64_t c      = index / (rect_height * rect_width);

  // Compute the lon/lat
  const T lon = core::XToLongitude<T>(x_rect, rect_width);
  const T lat = core::YToLatitude<T>(y_rect, rect_height);

  // Convert lon/lat to xyz 3D points
  T x, y, z;
  core::SphericalToXYZ(lon, lat, x, y, z);

  // Convert XYZ to cube map coordinates in range [0,1]
  int64_t cube_idx;
  T u, v;
  core::ConvertXYZToCubeMap(x, y, z, cube_idx, u, v);

  // Adjust u/v to cubemap face dimension
  u *= (cube_dim - 1);
  v *= (cube_dim - 1);

  // Starting location for this face in the cubemap image
  const int64_t channel_stride = c * 6 * cube_dim * cube_dim;
  const int64_t row_stride     = 6 * cube_dim;
  const int64_t face_start     = cube_idx * cube_dim;

  // Sample the cubemap with desired interpolation
  switch (interpolation) {
    default:
    case 0:
      data_rect[index] = core::Interpolate2DNearest(
          u, v, cube_dim, row_stride, data_cube + channel_stride + face_start);
      break;

    case 1:
      data_rect[index] = core::Interpolate2DBilinear(
          u, v, cube_dim, row_stride, data_cube + channel_stride + face_start);
      break;
  }
}

// Expects the cubemap input as [back, left, front, right, top, bottom] or
// equivalently [-z, -x, +z, +x, +y, -y]
void Cube2RectLauncher(at::Tensor cubemap, const int64_t cube_dim,
                       const int64_t rect_height, const int64_t rect_width,
                       const int64_t channels, const int64_t interpolation,
                       at::Tensor rect) {
  const int64_t num_kernels = channels * rect_height * rect_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      cubemap.type(), "Cube2RectLauncher", ([&] {
        Cube2RectKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, cubemap.data<scalar_t>(), cube_dim, rect_height,
            rect_width, channels, interpolation, rect.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace util
}  // namespace mapped_conv
#endif