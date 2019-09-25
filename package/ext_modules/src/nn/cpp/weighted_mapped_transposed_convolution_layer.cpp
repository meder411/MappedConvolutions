#include "nn/layers/weighted_mapped_transposed_convolution_layer.h"
#include "nn/cpp/mapped_im2col.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

at::Tensor WeightedMappedTransposedConvForward(
    at::Tensor input, at::Tensor sample_map, at::Tensor interp_weights,
    at::Tensor weight, at::Tensor bias, int outputHeight, int outputWidth,
    int kernel_size, int interpolation) {
  // Useful dimensions to have
  const int64_t nInputPlanes   = weight.size(0);
  const int64_t nOutputPlanes  = weight.size(1);
  const int64_t inputWidth     = input.size(3);
  const int64_t num_interp_pts = interp_weights.size(3);
  const int64_t batchSize      = input.size(0);

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
      MappedCol2Im2DWeighted<double>(
          nOutputPlanes * columns.size(1), columns, sample_map, interp_weights,
          outputHeight, outputWidth, inputWidth, columns.size(1), kernel_size,
          interpolation, num_interp_pts, output[b]);
    } else if (input.dtype() == at::kFloat) {
      MappedCol2Im2DWeighted<float>(
          nOutputPlanes * columns.size(1), columns, sample_map, interp_weights,
          outputHeight, outputWidth, inputWidth, columns.size(1), kernel_size,
          interpolation, num_interp_pts, output[b]);
    }
    // Use PyTorch to add the bias
    output[b] += bias.view({output[b].size(0), 1, 1});
  }
  return output;
}

at::Tensor WeightedMappedTransposedConvBackwardInput(
    at::Tensor grad_output, at::Tensor sample_map, at::Tensor interp_weights,
    at::Tensor weight, int inputHeight, int inputWidth, int kernel_size,
    int interpolation) {
  // Useful dimensions to have
  const int64_t nInputPlanes   = weight.size(0);
  const int64_t nOutputPlanes  = weight.size(1);
  const int64_t outputHeight   = grad_output.size(2);
  const int64_t outputWidth    = grad_output.size(3);
  const int64_t num_interp_pts = interp_weights.size(3);
  const int64_t batchSize      = grad_output.size(0);

  // Initialize output and temporary columns
  at::Tensor input_grad =
      at::zeros({batchSize, nInputPlanes, inputHeight, inputWidth},
                grad_output.options());
  at::Tensor columns =
      at::zeros({kernel_size * nOutputPlanes, inputHeight * inputWidth},
                grad_output.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == at::kDouble) {
      MappedIm2Col2DWeighted<double>(nOutputPlanes * columns.size(1),
                                     grad_output[b], sample_map,
                                     interp_weights, outputHeight, outputWidth,
                                     inputWidth, columns.size(1), kernel_size,
                                     interpolation, num_interp_pts, columns);
    } else if (grad_output.dtype() == at::kFloat) {
      MappedIm2Col2DWeighted<float>(nOutputPlanes * columns.size(1),
                                    grad_output[b], sample_map, interp_weights,
                                    outputHeight, outputWidth, inputWidth,
                                    columns.size(1), kernel_size,
                                    interpolation, num_interp_pts, columns);
    }

    // Use PyTorch for the matrix multiplication
    input_grad[b] = weight.view({weight.size(0), -1})
                        .mm(columns)
                        .view({nInputPlanes, inputHeight, inputWidth});
  }

  return input_grad;
}

at::Tensor WeightedMappedTransposedConvBackwardWeight(
    at::Tensor grad_output, at::Tensor sample_map, at::Tensor interp_weights,
    at::Tensor input, int kernel_size, int interpolation) {
  // Useful dimensions to have
  const int64_t nOutputPlanes  = grad_output.size(1);
  const int64_t nInputPlanes   = input.size(1);
  const int64_t outputHeight   = grad_output.size(2);
  const int64_t outputWidth    = grad_output.size(3);
  const int64_t inputHeight    = input.size(2);
  const int64_t inputWidth     = input.size(3);
  const int64_t num_interp_pts = interp_weights.size(3);
  const int64_t batchSize      = grad_output.size(0);

  // Initialize output and temporary columns
  at::Tensor weight_grad = at::zeros(
      {nInputPlanes, nOutputPlanes, kernel_size}, grad_output.options());
  at::Tensor columns =
      at::zeros({kernel_size * nOutputPlanes, inputHeight * inputWidth},
                grad_output.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == at::kDouble) {
      MappedIm2Col2DWeighted<double>(nOutputPlanes * columns.size(1),
                                     grad_output[b], sample_map,
                                     interp_weights, outputHeight, outputWidth,
                                     inputWidth, columns.size(1), kernel_size,
                                     interpolation, num_interp_pts, columns);
    } else if (grad_output.dtype() == at::kFloat) {
      MappedIm2Col2DWeighted<float>(nOutputPlanes * columns.size(1),
                                    grad_output[b], sample_map, interp_weights,
                                    outputHeight, outputWidth, inputWidth,
                                    columns.size(1), kernel_size,
                                    interpolation, num_interp_pts, columns);
    }

    // Use PyTorch for the final matrix multiplication
    weight_grad += input[b]
                       .view({input[b].size(0), -1})
                       .mm(columns.transpose(1, 0))
                       .view({nInputPlanes, nOutputPlanes, kernel_size});
  }
  return weight_grad;
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv