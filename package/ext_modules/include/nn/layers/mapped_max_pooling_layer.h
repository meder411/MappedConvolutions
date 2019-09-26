#ifndef MAPPED_MAX_POOLING_LAYER_H_
#define MAPPED_MAX_POOLING_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace nn {

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
std::vector<torch::Tensor> MappedMaxPoolForward(torch::Tensor input,
                                                torch::Tensor sample_map,
                                                int kernel_size,
                                                int interpolation);

torch::Tensor MappedMaxPoolBackward(torch::Tensor input,
                                    torch::Tensor idx_mask,
                                    torch::Tensor sample_map, int inputHeight,
                                    int inputWidth, int kernel_size,
                                    int interpolation);
}  // namespace cuda
#endif

namespace cpu {
std::vector<torch::Tensor> MappedMaxPoolForward(torch::Tensor input,
                                                torch::Tensor sample_map,
                                                int kernel_size,
                                                int interpolation);

torch::Tensor MappedMaxPoolBackward(torch::Tensor input,
                                    torch::Tensor idx_mask,
                                    torch::Tensor sample_map, int inputHeight,
                                    int inputWidth, int kernel_size,
                                    int interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

std::vector<torch::Tensor> MappedMaxPoolForward(torch::Tensor input,
                                                torch::Tensor sample_map,
                                                int kernel_size,
                                                int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    return cuda::MappedMaxPoolForward(input, sample_map, kernel_size,
                                      interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    return cpu::MappedMaxPoolForward(input, sample_map, kernel_size,
                                     interpolation);
  }
}

torch::Tensor MappedMaxPoolBackward(torch::Tensor input,
                                    torch::Tensor idx_mask,
                                    torch::Tensor sample_map, int inputHeight,
                                    int inputWidth, int kernel_size,
                                    int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(idx_mask);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(idx_mask);
    CHECK_CUDA(sample_map);
    return cuda::MappedMaxPoolBackward(input, idx_mask, sample_map,
                                       inputHeight, inputWidth, kernel_size,
                                       interpolation);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(idx_mask);
    CHECK_CPU(sample_map);
    return cpu::MappedMaxPoolBackward(input, idx_mask, sample_map, inputHeight,
                                      inputWidth, kernel_size, interpolation);
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapped_max_pool", &mapped_conv::nn::MappedMaxPoolForward,
        "Mapped max pooling operation");
  m.def("mapped_max_unpool", &mapped_conv::nn::MappedMaxPoolBackward,
        "Mapped max unpooling operation");
}

#endif