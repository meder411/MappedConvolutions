#ifndef WEIGHTED_MAPPED_CONVOLUTION_LAYER_H_
#define WEIGHTED_MAPPED_CONVOLUTION_LAYER_H_

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
torch::Tensor WeightedMappedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int64_t kernel_size, int64_t interpolation);

torch::Tensor WeightedMappedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, int64_t inputHeight,
    int64_t inputWidth, int64_t kernel_size, int64_t interpolation);

torch::Tensor WeightedMappedConvBackwardWeight(torch::Tensor grad_output,
                                               torch::Tensor sample_map,
                                               torch::Tensor interp_weights,
                                               torch::Tensor input,
                                               int64_t kernel_size,
                                               int64_t interpolation);
}  // namespace cuda
#endif

namespace cpu {

torch::Tensor WeightedMappedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int64_t kernel_size, int64_t interpolation);

torch::Tensor WeightedMappedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, int64_t inputHeight,
    int64_t inputWidth, int64_t kernel_size, int64_t interpolation);

torch::Tensor WeightedMappedConvBackwardWeight(torch::Tensor grad_output,
                                               torch::Tensor sample_map,
                                               torch::Tensor interp_weights,
                                               torch::Tensor input,
                                               int64_t kernel_size,
                                               int64_t interpolation);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor WeightedMappedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int64_t kernel_size, int64_t interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(interp_weights);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(sample_map);
    CHECK_CUDA(interp_weights);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    return cuda::WeightedMappedConvForward(input, sample_map, interp_weights,
                                           weight, bias, kernel_size,
                                           interpolation);
  } else
#endif
  {
    CHECK_CPU(sample_map);
    CHECK_CPU(interp_weights);
    CHECK_CPU(weight);
    CHECK_CPU(bias);
    return cpu::WeightedMappedConvForward(input, sample_map, interp_weights,
                                          weight, bias, kernel_size,
                                          interpolation);
  }
}

std::vector<torch::Tensor> WeightedMappedConvBackward(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor input, torch::Tensor weight,
    torch::Tensor bias, int64_t kernel_size, int64_t interpolation) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);
  CHECK_CONTIGUOUS(interp_weights);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    CHECK_CUDA(interp_weights);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    torch::Tensor grad_input = cuda::WeightedMappedConvBackwardInput(
        grad_output, sample_map, interp_weights, weight, input.size(2),
        input.size(3), kernel_size, interpolation);

    torch::Tensor grad_weight = cuda::WeightedMappedConvBackwardWeight(
        grad_output, sample_map, interp_weights, input, kernel_size,
        interpolation);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    CHECK_CPU(interp_weights);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    torch::Tensor grad_input = cpu::WeightedMappedConvBackwardInput(
        grad_output, sample_map, interp_weights, weight, input.size(2),
        input.size(3), kernel_size, interpolation);

    torch::Tensor grad_weight = cpu::WeightedMappedConvBackwardWeight(
        grad_output, sample_map, interp_weights, input, kernel_size,
        interpolation);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weighted_mapped_conv_forward",
        &mapped_conv::nn::WeightedMappedConvForward,
        "Forward bilinear mapped convolution");
  m.def("weighted_mapped_conv_backward",
        &mapped_conv::nn::WeightedMappedConvBackward,
        "Backward bilinear mapped convolution");
}

#endif