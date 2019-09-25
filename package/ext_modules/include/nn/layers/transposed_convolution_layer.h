#ifndef TRANSPOSED_CONVOLUTION_LAYER_H_
#define TRANSPOSED_CONVOLUTION_LAYER_H_

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
at::Tensor TransposedConvForward(at::Tensor input, at::Tensor weight,
                                 at::Tensor bias, int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, int pad_h,
                                 int pad_w, int dilation_h, int dilation_w);

at::Tensor TransposedConvBackwardInput(at::Tensor grad_output,
                                       at::Tensor weight, int inputHeight,
                                       int inputWidth, int kernel_h,
                                       int kernel_w, int stride_h,
                                       int stride_w, int pad_h, int pad_w,
                                       int dilation_h, int dilation_w);

at::Tensor TransposedConvBackwardWeight(at::Tensor grad_output,
                                        at::Tensor input, int kernel_h,
                                        int kernel_w, int stride_h,
                                        int stride_w, int pad_h, int pad_w,
                                        int dilation_h, int dilation_w);
}  // namespace cuda
#endif

namespace cpu {

at::Tensor TransposedConvForward(at::Tensor input, at::Tensor weight,
                                 at::Tensor bias, int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, int pad_h,
                                 int pad_w, int dilation_h, int dilation_w);

at::Tensor TransposedConvBackwardInput(at::Tensor grad_output,
                                       at::Tensor weight, int inputHeight,
                                       int inputWidth, int kernel_h,
                                       int kernel_w, int stride_h,
                                       int stride_w, int pad_h, int pad_w,
                                       int dilation_h, int dilation_w);

at::Tensor TransposedConvBackwardWeight(at::Tensor grad_output,
                                        at::Tensor input, int kernel_h,
                                        int kernel_w, int stride_h,
                                        int stride_w, int pad_h, int pad_w,
                                        int dilation_h, int dilation_w);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

at::Tensor TransposedConvForward(at::Tensor input, at::Tensor weight,
                                 at::Tensor bias, int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, int pad_h,
                                 int pad_w, int dilation_h, int dilation_w) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    return cuda::TransposedConvForward(input, weight, bias, kernel_h, kernel_w,
                                       stride_h, stride_w, pad_h, pad_w,
                                       dilation_h, dilation_w);
  } else
#endif
  {
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    return cpu::TransposedConvForward(input, weight, bias, kernel_h, kernel_w,
                                      stride_h, stride_w, pad_h, pad_w,
                                      dilation_h, dilation_w);
  }
}

std::vector<at::Tensor> TransposedConvBackward(
    at::Tensor grad_output, at::Tensor input, at::Tensor weight,
    at::Tensor bias, int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, int dilation_h, int dilation_w) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);

    at::Tensor grad_input = cuda::TransposedConvBackwardInput(
        grad_output, weight, input.size(2), input.size(3), kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

    at::Tensor grad_weight = cuda::TransposedConvBackwardWeight(
        grad_output, input, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w);

    at::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(weight);
    CHECK_CPU(bias);

    at::Tensor grad_input = cpu::TransposedConvBackwardInput(
        grad_output, weight, input.size(2), input.size(3), kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

    at::Tensor grad_weight = cpu::TransposedConvBackwardWeight(
        grad_output, input, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w);

    at::Tensor grad_bias = grad_output.sum({0, 2, 3});

    return {grad_input, grad_weight, grad_bias};
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transposed_conv_forward", &mapped_conv::nn::TransposedConvForward,
        "Forward transposed convolution");
  m.def("transposed_conv_backward", &mapped_conv::nn::TransposedConvBackward,
        "Backward transposed convolution");
}

#endif