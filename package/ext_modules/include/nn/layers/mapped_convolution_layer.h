#ifndef MAPPED_CONVOLUTION_LAYER_H_
#define MAPPED_CONVOLUTION_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace nn {

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// FORWARD DECLARATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
at::Tensor MappedConvForward(at::Tensor input, at::Tensor sample_map,
                             at::Tensor weight, at::Tensor bias,
                             int64_t kernel_size, int64_t interpolation);

at::Tensor MappedConvBackwardInput(at::Tensor grad_output,
                                   at::Tensor sample_map, at::Tensor weight,
                                   int64_t inputHeight, int64_t inputWidth,
                                   int64_t kernel_size, int64_t interpolation);

at::Tensor MappedConvBackwardWeight(at::Tensor grad_output,
                                    at::Tensor sample_map, at::Tensor input,
                                    int64_t kernel_size,
                                    int64_t interpolation);
}  // namespace cuda
#endif

namespace cpu {
at::Tensor MappedConvForward(at::Tensor input, at::Tensor sample_map,
                             at::Tensor weight, at::Tensor bias,
                             int64_t kernel_size, int64_t interpolation);

at::Tensor MappedConvBackwardInput(at::Tensor grad_output,
                                   at::Tensor sample_map, at::Tensor weight,
                                   int64_t inputHeight, int64_t inputWidth,
                                   int64_t kernel_size, int64_t interpolation);

at::Tensor MappedConvBackwardWeight(at::Tensor grad_output,
                                    at::Tensor sample_map, at::Tensor input,
                                    int64_t kernel_size,
                                    int64_t interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

at::Tensor MappedConvForward(at::Tensor input, at::Tensor sample_map,
                             at::Tensor weight, at::Tensor bias,
                             int64_t kernel_size, int64_t interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(sample_map);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    return cuda::MappedConvForward(input, sample_map, weight, bias,
                                   kernel_size, interpolation);
  } else
#endif
  {
    CHECK_CPU(sample_map);
    CHECK_CPU(weight);
    CHECK_CPU(bias);
    return cpu::MappedConvForward(input, sample_map, weight, bias, kernel_size,
                                  interpolation);
  }
}

std::vector<at::Tensor> MappedConvBackward(at::Tensor grad_output,
                                           at::Tensor sample_map,
                                           at::Tensor input, at::Tensor weight,
                                           at::Tensor bias,
                                           int64_t kernel_size,
                                           int64_t interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    at::Tensor grad_input = cuda::MappedConvBackwardInput(
        grad_output, sample_map, weight, input.size(2), input.size(3),
        kernel_size, interpolation);

    at::Tensor grad_weight = cuda::MappedConvBackwardWeight(
        grad_output, sample_map, input, kernel_size, interpolation);

    at::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    at::Tensor grad_input = cpu::MappedConvBackwardInput(
        grad_output, sample_map, weight, input.size(2), input.size(3),
        kernel_size, interpolation);

    at::Tensor grad_weight = cpu::MappedConvBackwardWeight(
        grad_output, sample_map, input, kernel_size, interpolation);

    at::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapped_conv_forward", &mapped_conv::nn::MappedConvForward,
        "Forward bilinear mapped convolution");
  m.def("mapped_conv_backward", &mapped_conv::nn::MappedConvBackward,
        "Backward bilinear mapped convolution");
}

#endif