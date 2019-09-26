#ifndef CUBE2RECT_LAYER_H_
#define CUBE2RECT_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace util {

#ifndef __NO_CUDA__
namespace cuda {

torch::Tensor Cube2RectForward(
    torch::Tensor cubemap,  // [-z, -x, +z, +x, +y, -y]
    const int64_t rect_height, const int64_t rect_width,
    const int64_t interpolation);

}  // namespace cuda
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor Cube2RectForward(
    torch::Tensor cubemap,  // [-z, -x, +z, +x, +y, -y]
    const int64_t rect_height, const int64_t rect_width,
    const int64_t interpolation) {
#ifndef __NO_CUDA__
  CHECK_CONTIGUOUS(cubemap);
  CHECK_CUDA(cubemap);

  return cuda::Cube2RectForward(cubemap, rect_height, rect_width,
                                interpolation);
#else
  printf("CUDA must be enabled to run cube2rect\n");
#endif
}

}  // namespace util
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cube2rect", &mapped_conv::util::Cube2RectForward,
        "Convert cubemape to equirectangular image");
}

#endif