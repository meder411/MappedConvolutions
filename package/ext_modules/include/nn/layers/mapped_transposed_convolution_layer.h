#ifndef MAPPED_TRANSPOSED_CONVOLUTION_LAYER_H_
#define MAPPED_TRANSPOSED_CONVOLUTION_LAYER_H_

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
at::Tensor MappedTransposedConvForward(at::Tensor input, at::Tensor sample_map,
                                       at::Tensor weight, at::Tensor bias,
                                       int outputHeight, int outputWidth,
                                       int kernel_size, int interpolation);

at::Tensor MappedTransposedConvBackwardInput(
    at::Tensor grad_output, at::Tensor sample_map, at::Tensor weight,
    int inputHeight, int inputWidth, int kernel_size, int interpolation);

at::Tensor MappedTransposedConvBackwardWeight(at::Tensor grad_output,
                                              at::Tensor sample_map,
                                              at::Tensor input,
                                              int kernel_size,
                                              int interpolation);
}  // namespace cuda
#endif

namespace cpu {
at::Tensor MappedTransposedConvForward(at::Tensor input, at::Tensor sample_map,
                                       at::Tensor weight, at::Tensor bias,
                                       int outputHeight, int outputWidth,
                                       int kernel_size, int interpolation);

at::Tensor MappedTransposedConvBackwardInput(
    at::Tensor grad_output, at::Tensor sample_map, at::Tensor weight,
    int inputHeight, int inputWidth, int kernel_size, int interpolation);

at::Tensor MappedTransposedConvBackwardWeight(at::Tensor grad_output,
                                              at::Tensor sample_map,
                                              at::Tensor input,
                                              int kernel_size,
                                              int interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

at::Tensor MappedTransposedConvForward(at::Tensor input, at::Tensor sample_map,
                                       at::Tensor weight, at::Tensor bias,
                                       int outputHeight, int outputWidth,
                                       int kernel_size, int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CUDA(sample_map);

    return cuda::MappedTransposedConvForward(input, sample_map, weight, bias,
                                             outputHeight, outputWidth,
                                             kernel_size, interpolation);
  } else
#endif
  {
    CHECK_CPU(sample_map);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    return cpu::MappedTransposedConvForward(input, sample_map, weight, bias,
                                            outputHeight, outputWidth,
                                            kernel_size, interpolation);
  }
}

std::vector<at::Tensor> MappedTransposedConvBackward(
    at::Tensor grad_output, at::Tensor sample_map, at::Tensor input,
    at::Tensor weight, at::Tensor bias, int kernel_size, int interpolation) {
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.type().is_cuda()) {
    CHECK_CUDA(sample_map);
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    at::Tensor grad_input = cuda::MappedTransposedConvBackwardInput(
        grad_output, sample_map, weight, input.size(2), input.size(3),
        kernel_size, interpolation);

    at::Tensor grad_weight = cuda::MappedTransposedConvBackwardWeight(
        grad_output, sample_map, input, kernel_size, interpolation);

    at::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  } else
#endif
  {
    CHECK_CPU(sample_map);
    CHECK_CPU(input);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    at::Tensor grad_input = cpu::MappedTransposedConvBackwardInput(
        grad_output, sample_map, weight, input.size(2), input.size(3),
        kernel_size, interpolation);

    at::Tensor grad_weight = cpu::MappedTransposedConvBackwardWeight(
        grad_output, sample_map, input, kernel_size, interpolation);

    at::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapped_transposed_conv_forward",
        &mapped_conv::nn::MappedTransposedConvForward, "Forward convolution");
  m.def("mapped_transposed_conv_backward",
        &mapped_conv::nn::MappedTransposedConvBackward,
        "Backward convolution");
}

#endif