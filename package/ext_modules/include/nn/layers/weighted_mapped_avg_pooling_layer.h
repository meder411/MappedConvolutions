#ifndef WEIGHTED_MAPPED_AVG_POOLING_LAYER_H_
#define WEIGHTED_MAPPED_AVG_POOLING_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace nn {

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
at::Tensor WeightedMappedAvgPoolForward(at::Tensor input,
                                        at::Tensor sample_map,
                                        at::Tensor interp_weights,
                                        int kernel_size, int interpolation);

at::Tensor WeightedMappedAvgPoolBackward(at::Tensor input,
                                         at::Tensor sample_map,
                                         at::Tensor interp_weights,
                                         int inputHeight, int inputWidth,
                                         int kernel_size, int interpolation);
}  // namespace cuda
#endif

namespace cpu {
at::Tensor WeightedMappedAvgPoolForward(at::Tensor input,
                                        at::Tensor sample_map,
                                        at::Tensor interp_weights,
                                        int kernel_size, int interpolation);

at::Tensor WeightedMappedAvgPoolBackward(at::Tensor input,
                                         at::Tensor sample_map,
                                         at::Tensor interp_weights,
                                         int inputHeight, int inputWidth,
                                         int kernel_size, int interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

at::Tensor WeightedMappedAvgPoolForward(at::Tensor input,
                                        at::Tensor sample_map,
                                        at::Tensor interp_weights,
                                        int kernel_size, int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(interp_weights);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    CHECK_CUDA(interp_weights);
    return cuda::WeightedMappedAvgPoolForward(
        input, sample_map, interp_weights, kernel_size, interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    CHECK_CPU(interp_weights);
    return cpu::WeightedMappedAvgPoolForward(input, sample_map, interp_weights,
                                             kernel_size, interpolation);
  }
}

at::Tensor WeightedMappedAvgPoolBackward(at::Tensor input,
                                         at::Tensor sample_map,
                                         at::Tensor interp_weights,
                                         int inputHeight, int inputWidth,
                                         int kernel_size, int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(interp_weights);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    CHECK_CUDA(interp_weights);
    return cuda::WeightedMappedAvgPoolBackward(
        input, sample_map, interp_weights, inputHeight, inputWidth,
        kernel_size, interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    CHECK_CPU(interp_weights);
    return cpu::WeightedMappedAvgPoolBackward(
        input, sample_map, interp_weights, inputHeight, inputWidth,
        kernel_size, interpolation);
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weighted_mapped_avg_pool",
        &mapped_conv::nn::WeightedMappedAvgPoolForward,
        "Mapped max pooling operation");
  m.def("weighted_mapped_avg_unpool",
        &mapped_conv::nn::WeightedMappedAvgPoolBackward,
        "Mapped max unpooling operation");
}

#endif