#include "nn/layers/weighted_mapped_convolution_layer.h"
#include "nn/cpp/mapped_im2col.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

torch::Tensor WeightedMappedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int64_t kernel_size, int64_t interpolation) {
  // Useful dimensions to have
  const int64_t nOutputPlanes  = weight.size(0);
  const int64_t nInputPlanes   = weight.size(1);
  const int64_t inputHeight    = input.size(2);
  const int64_t inputWidth     = input.size(3);
  const int64_t outputHeight   = sample_map.size(0);
  const int64_t outputWidth    = sample_map.size(1);
  const int64_t num_interp_pts = interp_weights.size(3);
  const int64_t batchSize      = input.size(0);

  // Initialize output and resize temporary columns
  torch::Tensor output = torch::zeros(
      {batchSize, nOutputPlanes, outputHeight, outputWidth}, input.options());
  torch::Tensor columns =
      torch::zeros({kernel_size * nInputPlanes, outputHeight * outputWidth},
                   input.options());
  int64_t num_kernels = nInputPlanes * columns.size(1);

  // For each elt in batch, do:
  for (int64_t b = 0; b < batchSize; b++) {
    if (input.dtype() == torch::kDouble) {
      MappedIm2Col2DWeighted<double>(num_kernels, input[b], sample_map,
                                     interp_weights, inputHeight, inputWidth,
                                     outputWidth, columns.size(1), kernel_size,
                                     interpolation, num_interp_pts, columns);
    } else if (input.dtype() == torch::kFloat) {
      MappedIm2Col2DWeighted<float>(num_kernels, input[b], sample_map,
                                    interp_weights, inputHeight, inputWidth,
                                    outputWidth, columns.size(1), kernel_size,
                                    interpolation, num_interp_pts, columns);
    }

    // Use PyTorch for the rest
    // Compute the convolution output
    output[b] = weight.view({weight.size(0), weight.size(1) * weight.size(2)})
                    .mm(columns)
                    .view({weight.size(0), output.size(2), output.size(3)});

    // Add the bias
    output[b] += bias.view({output[b].size(0), 1, 1});
  }
  return output;
}

torch::Tensor WeightedMappedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, int64_t inputHeight,
    int64_t inputWidth, int64_t kernel_size, int64_t interpolation) {
  // Useful dimensions to have
  const int64_t nInputPlanes   = weight.size(1);
  const int64_t nOutputPlanes  = grad_output.size(1);
  const int64_t outputHeight   = grad_output.size(2);
  const int64_t outputWidth    = grad_output.size(3);
  const int64_t num_interp_pts = interp_weights.size(3);
  const int64_t batchSize      = grad_output.size(0);

  // Initialize output and temporary columns
  torch::Tensor input_grad =
      torch::zeros({batchSize, nInputPlanes, inputHeight, inputWidth},
                   grad_output.options());

  // For each elt in batch, do:
  for (int64_t b = 0; b < batchSize; b++) {
    torch::Tensor columns =
        weight.view({weight.size(0), weight.size(1) * weight.size(2)})
            .transpose(1, 0)
            .mm(grad_output[b].view(
                {nOutputPlanes, outputHeight * outputWidth}));

    if (grad_output.dtype() == torch::kDouble) {
      MappedCol2Im2DWeighted<double>(
          nInputPlanes * columns.size(1), columns, sample_map, interp_weights,
          inputHeight, inputWidth, outputWidth, outputHeight * outputWidth,
          kernel_size, interpolation, num_interp_pts, input_grad[b]);
    } else if (grad_output.dtype() == torch::kFloat) {
      MappedCol2Im2DWeighted<float>(
          nInputPlanes * columns.size(1), columns, sample_map, interp_weights,
          inputHeight, inputWidth, outputWidth, outputHeight * outputWidth,
          kernel_size, interpolation, num_interp_pts, input_grad[b]);
    }
  }
  return input_grad;
}

torch::Tensor WeightedMappedConvBackwardWeight(torch::Tensor grad_output,
                                               torch::Tensor sample_map,
                                               torch::Tensor interp_weights,
                                               torch::Tensor input,
                                               int64_t kernel_size,
                                               int64_t interpolation) {
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
      {nOutputPlanes, nInputPlanes, kernel_size}, grad_output.options());
  torch::Tensor columns =
      torch::zeros({kernel_size * nInputPlanes, outputHeight * outputWidth},
                   grad_output.options());

  // For each elt in batch, do:
  const int64_t num_kernels = nInputPlanes * columns.size(1);
  for (int64_t b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == torch::kDouble) {
      MappedIm2Col2DWeighted<double>(num_kernels, input[b], sample_map,
                                     interp_weights, inputHeight, inputWidth,
                                     outputWidth, columns.size(1), kernel_size,
                                     interpolation, num_interp_pts, columns);
    } else if (grad_output.dtype() == torch::kFloat) {
      MappedIm2Col2DWeighted<float>(num_kernels, input[b], sample_map,
                                    interp_weights, inputHeight, inputWidth,
                                    outputWidth, columns.size(1), kernel_size,
                                    interpolation, num_interp_pts, columns);
    }

    // Use PyTorch for the rest
    // Compute the convolution output
    weight_grad += grad_output[b]
                       .view({nOutputPlanes, outputHeight * outputWidth})
                       .mm(columns.transpose(1, 0))
                       .view({nOutputPlanes, nInputPlanes, kernel_size});
  }

  return weight_grad;
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv