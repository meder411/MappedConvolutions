#ifndef CONVOLUTION_LAYER_H_
#define CONVOLUTION_LAYER_H_

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
torch::Tensor ConvForward(torch::Tensor input, torch::Tensor weight,
                          torch::Tensor bias, int kernel_h, int kernel_w,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          int dilation_h, int dilation_w);

torch::Tensor ConvBackwardInput(torch::Tensor grad_output,
                                torch::Tensor weight, int inputHeight,
                                int inputWidth, int kernel_h, int kernel_w,
                                int stride_h, int stride_w, int pad_h,
                                int pad_w, int dilation_h, int dilation_w);

torch::Tensor ConvBackwardWeight(torch::Tensor grad_output,
                                 torch::Tensor input, int kernel_h,
                                 int kernel_w, int stride_h, int stride_w,
                                 int pad_h, int pad_w, int dilation_h,
                                 int dilation_w);
}  // namespace cuda
#endif

namespace cpu {
torch::Tensor ConvForward(torch::Tensor input, torch::Tensor weight,
                          torch::Tensor bias, int kernel_h, int kernel_w,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          int dilation_h, int dilation_w);

torch::Tensor ConvBackwardInput(torch::Tensor grad_output,
                                torch::Tensor weight, int inputHeight,
                                int inputWidth, int kernel_h, int kernel_w,
                                int stride_h, int stride_w, int pad_h,
                                int pad_w, int dilation_h, int dilation_w);

torch::Tensor ConvBackwardWeight(torch::Tensor grad_output,
                                 torch::Tensor input, int kernel_h,
                                 int kernel_w, int stride_h, int stride_w,
                                 int pad_h, int pad_w, int dilation_h,
                                 int dilation_w);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor ConvForward(torch::Tensor input, torch::Tensor weight,
                          torch::Tensor bias, int kernel_h, int kernel_w,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          int dilation_h, int dilation_w) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    return cuda::ConvForward(input, weight, bias, kernel_h, kernel_w, stride_h,
                             stride_w, pad_h, pad_w, dilation_h, dilation_w);
  } else
#endif
  {
    CHECK_CPU(weight);
    CHECK_CPU(bias);
    return cpu::ConvForward(input, weight, bias, kernel_h, kernel_w, stride_h,
                            stride_w, pad_h, pad_w, dilation_h, dilation_w);
  }
}

std::vector<torch::Tensor> ConvBackward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
    torch::Tensor bias, int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, int dilation_h, int dilation_w) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    torch::Tensor grad_input = cuda::ConvBackwardInput(
        grad_output, weight, input.size(2), input.size(3), kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

    torch::Tensor grad_weight = cuda::ConvBackwardWeight(
        grad_output, input, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    torch::Tensor grad_input = cpu::ConvBackwardInput(
        grad_output, weight, input.size(2), input.size(3), kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

    torch::Tensor grad_weight = cpu::ConvBackwardWeight(
        grad_output, input, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_forward", &mapped_conv::nn::ConvForward, "Forward convolution");
  m.def("conv_backward", &mapped_conv::nn::ConvBackward,
        "Backward convolution");
}

#endif