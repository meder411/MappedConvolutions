#ifndef RESAMPLE_LAYER_H_
#define RESAMPLE_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace nn {

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
at::Tensor ResampleToMap(at::Tensor input, at::Tensor sample_map,
                         int outputHeight, int outputWidth, int interpolation);

at::Tensor ResampleFromMap(at::Tensor grad_output, at::Tensor sample_map,
                           int interpolation);
}  // namespace cuda
#endif

namespace cpu {
at::Tensor ResampleToMap(at::Tensor input, at::Tensor sample_map,
                         int outputHeight, int outputWidth, int interpolation);

at::Tensor ResampleFromMap(at::Tensor grad_output, at::Tensor sample_map,
                           int interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

at::Tensor ResampleToMap(at::Tensor input, at::Tensor sample_map,
                         int outputHeight, int outputWidth,
                         int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    return cuda::ResampleToMap(input, sample_map, outputHeight, outputWidth,
                               interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    return cpu::ResampleToMap(input, sample_map, outputHeight, outputWidth,
                              interpolation);
  }
}

at::Tensor ResampleFromMap(at::Tensor grad_output, at::Tensor sample_map,
                           int interpolation) {
  CHECK_CONTIGUOUS(grad_output);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.type().is_cuda()) {
    CHECK_CUDA(grad_output);
    CHECK_CUDA(sample_map);
    return cuda::ResampleFromMap(grad_output, sample_map, interpolation);
  } else
#endif
  {
    CHECK_CPU(grad_output);
    CHECK_CPU(sample_map);
    return cpu::ResampleFromMap(grad_output, sample_map, interpolation);
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("resample_to_map", &mapped_conv::nn::ResampleToMap,
        "Resampling operation");
  m.def("resample_from_map", &mapped_conv::nn::ResampleFromMap,
        "Unresampling operation");
}

#endif