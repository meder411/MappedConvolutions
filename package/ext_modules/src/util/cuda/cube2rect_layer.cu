#include <ATen/ATen.h>

#include <vector>

#include "util/cuda/cube2rect.cuh"

namespace mapped_conv {
namespace util {
namespace cuda {

at::Tensor Cube2RectForward(at::Tensor cubemap,  // [-z, -x, +z, +x, +y, -y]
                            const int64_t rect_height,
                            const int64_t rect_width,
                            const int64_t interpolation) {
  // Useful dimensions to have
  const int64_t channels = cubemap.size(0);
  const int64_t cube_dim = cubemap.size(1);

  // Initialize output and index mask
  at::Tensor rect =
      at::zeros({channels, rect_height, rect_width}, cubemap.options());

  // Call the CUDA kernel
  Cube2RectLauncher(cubemap, cube_dim, rect_height, rect_width, channels,
                    interpolation, rect);

  return rect;
}
}  // namespace cuda
}  // namespace util
}  // namespace mapped_conv