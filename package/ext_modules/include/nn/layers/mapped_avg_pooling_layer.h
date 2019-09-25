#ifndef MAPPED_AVG_POOLING_LAYER_H_
#define MAPPED_AVG_POOLING_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace nn {

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
at::Tensor MappedAvgPoolForward(at::Tensor input, at::Tensor sample_map,
                                int kernel_size, int interpolation);

at::Tensor MappedAvgPoolBackward(at::Tensor input, at::Tensor sample_map,
                                 int inputHeight, int inputWidth,
                                 int kernel_size, int interpolation);
}  // namespace cuda
#endif

namespace cpu {
at::Tensor MappedAvgPoolForward(at::Tensor input, at::Tensor sample_map,
                                int kernel_size, int interpolation);

at::Tensor MappedAvgPoolBackward(at::Tensor input, at::Tensor sample_map,
                                 int inputHeight, int inputWidth,
                                 int kernel_size, int interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

at::Tensor MappedAvgPoolforward(at::Tensor input, at::Tensor sample_map,
                                int kernel_size, int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    return cuda::MappedAvgPoolForward(input, sample_map, kernel_size,
                                      interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    return cpu::MappedAvgPoolForward(input, sample_map, kernel_size,
                                     interpolation);
  }
}

at::Tensor MappedAvgPoolbackward(at::Tensor input, at::Tensor sample_map,
                                 int inputHeight, int inputWidth,
                                 int kernel_size, int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    return cuda::MappedAvgPoolBackward(input, sample_map, inputHeight,
                                       inputWidth, kernel_size, interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    return cpu::MappedAvgPoolBackward(input, sample_map, inputHeight,
                                      inputWidth, kernel_size, interpolation);
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapped_avg_pool", &mapped_conv::nn::MappedAvgPoolforward,
        "Mapped max pooling operation");
  m.def("mapped_avg_unpool", &mapped_conv::nn::MappedAvgPoolbackward,
        "Mapped max unpooling operation");
}

#endif