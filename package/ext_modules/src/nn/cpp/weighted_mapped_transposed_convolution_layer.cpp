#include "nn/layers/weighted_mapped_transposed_convolution_layer.h"
#include "nn/cpp/mapped_im2col.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

torch::Tensor WeightedMappedTransposedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int outputHeight, int outputWidth, int kernel_size, int interpolation) {
  // Useful dimensions to have
  const int64_t nInputPlanes   = weight.size(0);
  const int64_t nOutputPlanes  = weight.size(1);
  const int64_t inputWidth     = input.size(3);
  const int64_t num_interp_pts = interp_weights.size(3);
  const int64_t batchSize      = input.size(0);

  // Initialize output and temporary columns
  torch::Tensor output = torch::zeros(
      {batchSize, nOutputPlanes, outputHeight, outputWidth}, input.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    // Use PyTorch for the initial matrix multiplication
    torch::Tensor columns = weight.view({weight.size(0), -1})
                                .transpose(1, 0)
                                .mm(input[b].view({nInputPlanes, -1}));

    if (input.dtype() == torch::kDouble) {
      MappedCol2Im2DWeighted<double>(
          nOutputPlanes * columns.size(1), columns, sample_map, interp_weights,
          outputHeight, outputWidth, inputWidth, columns.size(1), kernel_size,
          interpolation, num_interp_pts, output[b]);
    } else if (input.dtype() == torch::kFloat) {
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

torch::Tensor WeightedMappedTransposedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, int inputHeight,
    int inputWidth, int kernel_size, int interpolation) {
  // Useful dimensions to have
  const int64_t nInputPlanes   = weight.size(0);
  const int64_t nOutputPlanes  = weight.size(1);
  const int64_t outputHeight   = grad_output.size(2);
  const int64_t outputWidth    = grad_output.size(3);
  const int64_t num_interp_pts = interp_weights.size(3);
  const int64_t batchSize      = grad_output.size(0);

  // Initialize output and temporary columns
  torch::Tensor input_grad =
      torch::zeros({batchSize, nInputPlanes, inputHeight, inputWidth},
                   grad_output.options());
  torch::Tensor columns =
      torch::zeros({kernel_size * nOutputPlanes, inputHeight * inputWidth},
                   grad_output.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == torch::kDouble) {
      MappedIm2Col2DWeighted<double>(nOutputPlanes * columns.size(1),
                                     grad_output[b], sample_map,
                                     interp_weights, outputHeight, outputWidth,
                                     inputWidth, columns.size(1), kernel_size,
                                     interpolation, num_interp_pts, columns);
    } else if (grad_output.dtype() == torch::kFloat) {
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

torch::Tensor WeightedMappedTransposedConvBackwardWeight(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor input, int kernel_size,
    int interpolation) {
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
  torch::Tensor weight_grad = torch::zeros(
      {nInputPlanes, nOutputPlanes, kernel_size}, grad_output.options());
  torch::Tensor columns =
      torch::zeros({kernel_size * nOutputPlanes, inputHeight * inputWidth},
                   grad_output.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == torch::kDouble) {
      MappedIm2Col2DWeighted<double>(nOutputPlanes * columns.size(1),
                                     grad_output[b], sample_map,
                                     interp_weights, outputHeight, outputWidth,
                                     inputWidth, columns.size(1), kernel_size,
                                     interpolation, num_interp_pts, columns);
    } else if (grad_output.dtype() == torch::kFloat) {
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