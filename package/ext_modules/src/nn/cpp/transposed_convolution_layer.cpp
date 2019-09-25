#include "nn/layers/transposed_convolution_layer.h"
#include "nn/cpp/im2col.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

/*
The transposed forward pass works by leveraging the same GEMM operation as the
backward pass w.r.t the inputs. The difference is in the arguments.

First, recall a traditional forward convolution operation. For each batch:
1. We form the matrix C from the image using the im2col function. C has
dimensions (ck^2, n) for c input channels, (square) filter size k, and n
filtering locations (e.g. pixels in the output)
2. We then perform the convolution as a matrix multiplication between our
weights W (dim: d, ck^2, for d output channels).
3. We get the resulting output O = WC, with dimensions (d, n). We can then
reshape n into the correct (h,w).


Now, recall the traiditonal backward pass w.r.t the inputs. For each batch:
1. Compute the matrix multiplication between the output gradients O (dim: d, n)
and the weights W (dim: d, ck^2). This gives the input gradients in column form
s.t. matrix C = W^T O
2. We then use the col2im function to decompose C back into the input shape


The transposed convolution takes the same arguments as the traditional
forward convolution, but the GEMM operation is the same as the traditional
backward convolution. To make this work, we first need to transpose the
dimensions of the weight matrix, hence the term "transposed convolution." This
transpose swaps the input and output channels, so the weights matrix goes from
(d, c, kh, kw) --> (c, d, kh, kw). Then, for each batch:
1. Compute the matrix multiplication between the input I (dim: c, n) and the
weight matrix (dim: c, dk^2, note the change here). This gives the output in
column form s.t. matrix C = W^T I (C dims: dk^2, n)
2. Now redistribute the column matrix C into the output O (dim: d, n) using
the col2im function.

In this function call, we assume the weight matrix has already been transposed.
*/

at::Tensor TransposedConvForward(at::Tensor input, at::Tensor weight,
                                 at::Tensor bias, int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, int pad_h,
                                 int pad_w, int dilation_h, int dilation_w) {
  // Useful dimensions to have
  const int64_t nInputPlanes  = weight.size(0);
  const int64_t nOutputPlanes = weight.size(1);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t outputHeight  = stride_h * (inputHeight - 1) + kernel_h +
                               (kernel_h - 1) * (dilation_h - 1) - 2 * pad_h;
  const int64_t outputWidth = stride_w * (inputWidth - 1) + kernel_w +
                              (kernel_w - 1) * (dilation_w - 1) - 2 * pad_w;
  const int64_t batchSize = input.size(0);

  // Initialize output and temporary columns
  at::Tensor output = at::zeros(
      {batchSize, nOutputPlanes, outputHeight, outputWidth}, input.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    // Use PyTorch for the initial matrix multiplication
    at::Tensor columns = weight.view({weight.size(0), -1})
                             .transpose(1, 0)
                             .mm(input[b].view({nInputPlanes, -1}));

    if (input.dtype() == at::kDouble) {
      Col2Im2D<double>(nOutputPlanes * outputHeight * outputWidth, columns,
                       outputHeight, outputWidth, inputHeight, inputWidth,
                       kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w, output[b]);
    } else if (input.dtype() == at::kFloat) {
      Col2Im2D<float>(nOutputPlanes * outputHeight * outputWidth, columns,
                      outputHeight, outputWidth, inputHeight, inputWidth,
                      kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                      dilation_h, dilation_w, output[b]);
    }

    // Use PyTorch to add the bias
    output[b] += bias.view({output[b].size(0), 1, 1});
  }

  return output;
}

at::Tensor TransposedConvBackwardInput(at::Tensor grad_output,
                                       at::Tensor weight, int inputHeight,
                                       int inputWidth, int kernel_h,
                                       int kernel_w, int stride_h,
                                       int stride_w, int pad_h, int pad_w,
                                       int dilation_h, int dilation_w) {
  // Useful dimensions to have
  const int64_t nInputPlanes  = weight.size(0);
  const int64_t nOutputPlanes = weight.size(1);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t batchSize     = grad_output.size(0);

  // Initialize output and temporary columns
  at::Tensor input_grad =
      at::zeros({batchSize, nInputPlanes, inputHeight, inputWidth},
                grad_output.options());
  at::Tensor columns = at::zeros(
      {kernel_w * kernel_h * nOutputPlanes, inputHeight * inputWidth},
      grad_output.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == at::kDouble) {
      Im2Col2D<double>(nOutputPlanes * columns.size(1), grad_output[b],
                       outputHeight, outputWidth, inputWidth, columns.size(1),
                       kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w, columns);
    } else if (grad_output.dtype() == at::kFloat) {
      Im2Col2D<float>(nOutputPlanes * columns.size(1), grad_output[b],
                      outputHeight, outputWidth, inputWidth, columns.size(1),
                      kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                      dilation_h, dilation_w, columns);
    }

    // Use PyTorch for the matrix multiplication
    input_grad[b] = weight.view({weight.size(0), -1})
                        .mm(columns)
                        .view({nInputPlanes, inputHeight, inputWidth});
  }

  return input_grad;
}

at::Tensor TransposedConvBackwardWeight(at::Tensor grad_output,
                                        at::Tensor input, int kernel_h,
                                        int kernel_w, int stride_h,
                                        int stride_w, int pad_h, int pad_w,
                                        int dilation_h, int dilation_w) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = grad_output.size(1);
  const int64_t nInputPlanes  = input.size(1);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t batchSize     = grad_output.size(0);

  // Initialize output and temporary columns
  at::Tensor weight_grad =
      at::zeros({nInputPlanes, nOutputPlanes, kernel_h, kernel_w},
                grad_output.options());
  at::Tensor columns = at::zeros(
      {kernel_w * kernel_h * nOutputPlanes, inputHeight * inputWidth},
      grad_output.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    // Create the column matrix from the grad output as we would for the input
    // in the standard conv_forward
    if (grad_output.dtype() == at::kDouble) {
      Im2Col2D<double>(nOutputPlanes * columns.size(1), grad_output[b],
                       outputHeight, outputWidth, inputWidth, columns.size(1),
                       kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w, columns);
    } else if (grad_output.dtype() == at::kFloat) {
      Im2Col2D<float>(nOutputPlanes * columns.size(1), grad_output[b],
                      outputHeight, outputWidth, inputWidth, columns.size(1),
                      kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                      dilation_h, dilation_w, columns);
    }

    // Use PyTorch for the final matrix multiplication
    weight_grad +=
        input[b]
            .view({input[b].size(0), -1})
            .mm(columns.transpose(1, 0))
            .view({nInputPlanes, nOutputPlanes, kernel_h, kernel_w});
  }

  return weight_grad;
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv