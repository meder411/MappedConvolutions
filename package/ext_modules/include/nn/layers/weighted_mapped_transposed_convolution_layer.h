#ifndef WEIGHTED_MAPPED_TRANSPOSED_CONVOLUTION_LAYER_H_
#define WEIGHTED_MAPPED_TRANSPOSED_CONVOLUTION_LAYER_H_

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

torch::Tensor WeightedMappedTransposedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int outputHeight, int outputWidth, int kernel_size, int interpolation);

torch::Tensor WeightedMappedTransposedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, int inputHeight,
    int inputWidth, int kernel_size, int interpolation);

torch::Tensor WeightedMappedTransposedConvBackwardWeight(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor input, int kernel_size,
    int interpolation);

}  // namespace cuda
#endif

namespace cpu {

torch::Tensor WeightedMappedTransposedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int outputHeight, int outputWidth, int kernel_size, int interpolation);

torch::Tensor WeightedMappedTransposedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, int inputHeight,
    int inputWidth, int kernel_size, int interpolation);

torch::Tensor WeightedMappedTransposedConvBackwardWeight(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor input, int kernel_size,
    int interpolation);

}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor WeightedMappedTransposedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int outputHeight, int outputWidth, int kernel_size, int interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(interp_weights);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CUDA(sample_map);
    CHECK_CUDA(interp_weights);

    return cuda::WeightedMappedTransposedConvForward(
        input, sample_map, interp_weights, weight, bias, outputHeight,
        outputWidth, kernel_size, interpolation);
  } else
#endif
  {
    CHECK_CPU(sample_map);
    CHECK_CPU(interp_weights);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    return cpu::WeightedMappedTransposedConvForward(
        input, sample_map, interp_weights, weight, bias, outputHeight,
        outputWidth, kernel_size, interpolation);
  }
}

std::vector<torch::Tensor> WeightedMappedTransposedConvBackward(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor input, torch::Tensor weight,
    torch::Tensor bias, int kernel_size, int interpolation) {
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(interp_weights);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.type().is_cuda()) {
    CHECK_CUDA(sample_map);
    CHECK_CUDA(interp_weights);
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    // torch::Tensor grad_input =
    torch::Tensor grad_input = cuda::WeightedMappedTransposedConvBackwardInput(
        grad_output, sample_map, interp_weights, weight, input.size(2),
        input.size(3), kernel_size, interpolation);

    torch::Tensor grad_weight =
        cuda::WeightedMappedTransposedConvBackwardWeight(
            grad_output, sample_map, interp_weights, input, kernel_size,
            interpolation);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  } else
#endif
  {
    CHECK_CPU(sample_map);
    CHECK_CPU(interp_weights);
    CHECK_CPU(input);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    torch::Tensor grad_input = cpu::WeightedMappedTransposedConvBackwardInput(
        grad_output, sample_map, interp_weights, weight, input.size(2),
        input.size(3), kernel_size, interpolation);

    torch::Tensor grad_weight =
        cpu::WeightedMappedTransposedConvBackwardWeight(
            grad_output, sample_map, interp_weights, input, kernel_size,
            interpolation);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weighted_mapped_transposed_conv_forward",
        &mapped_conv::nn::WeightedMappedTransposedConvForward,
        "Forward convolution");
  m.def("weighted_mapped_transposed_conv_backward",
        &mapped_conv::nn::WeightedMappedTransposedConvBackward,
        "Backward convolution");
}

#endif