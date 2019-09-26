#include "nn/layers/mapped_convolution_layer.h"
#include "nn/cpp/mapped_im2col.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

torch::Tensor MappedConvForward(torch::Tensor input, torch::Tensor map,
                                torch::Tensor weight, torch::Tensor bias,
                                int64_t kernel_size, int64_t interpolation) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = weight.size(0);
  const int64_t nInputPlanes  = weight.size(1);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t outputHeight  = map.size(0);
  const int64_t outputWidth   = map.size(1);
  const int64_t batchSize     = input.size(0);

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
      MappedIm2Col2D<double>(num_kernels, input[b], map, inputHeight,
                             inputWidth, outputWidth, columns.size(1),
                             kernel_size, interpolation, columns);
    } else if (input.dtype() == torch::kFloat) {
      MappedIm2Col2D<float>(num_kernels, input[b], map, inputHeight,
                            inputWidth, outputWidth, columns.size(1),
                            kernel_size, interpolation, columns);
    }

    // Use PyTorch for the rest
    // Compute the convolution output
    output[b] = weight.view({weight.size(0), -1})
                    .mm(columns)
                    .view({weight.size(0), output.size(2), output.size(3)});

    // Add the bias
    output[b] += bias.view({output[b].size(0), 1, 1});
  }
  return output;
}

torch::Tensor MappedConvBackwardInput(torch::Tensor grad_output,
                                      torch::Tensor map, torch::Tensor weight,
                                      int64_t inputHeight, int64_t inputWidth,
                                      int64_t kernel_size,
                                      int64_t interpolation) {
  // Useful dimensions to have
  const int64_t nInputPlanes  = weight.size(1);
  const int64_t nOutputPlanes = grad_output.size(1);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t batchSize     = grad_output.size(0);

  // Initialize output and temporary columns
  torch::Tensor input_grad =
      torch::zeros({batchSize, nInputPlanes, inputHeight, inputWidth},
                   grad_output.options());

  // For each elt in batch, do:
  for (int64_t b = 0; b < batchSize; b++) {
    torch::Tensor columns = weight.view({weight.size(0), -1})
                                .transpose(1, 0)
                                .mm(grad_output[b].view({nOutputPlanes, -1}));

    if (grad_output.dtype() == torch::kDouble) {
      MappedCol2Im2D<double>(nInputPlanes * columns.size(1), columns, map,
                             inputHeight, inputWidth, outputWidth,
                             outputHeight * outputWidth, kernel_size,
                             interpolation, input_grad[b]);
    } else if (grad_output.dtype() == torch::kFloat) {
      MappedCol2Im2D<float>(nInputPlanes * columns.size(1), columns, map,
                            inputHeight, inputWidth, outputWidth,
                            outputHeight * outputWidth, kernel_size,
                            interpolation, input_grad[b]);
    }
  }
  return input_grad;
}

torch::Tensor MappedConvBackwardWeight(torch::Tensor grad_output,
                                       torch::Tensor map, torch::Tensor input,
                                       int64_t kernel_size,
                                       int64_t interpolation) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = grad_output.size(1);
  const int64_t nInputPlanes  = input.size(1);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t batchSize     = grad_output.size(0);

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
      MappedIm2Col2D<double>(num_kernels, input[b], map, inputHeight,
                             inputWidth, outputWidth, columns.size(1),
                             kernel_size, interpolation, columns);
    } else if (grad_output.dtype() == torch::kFloat) {
      MappedIm2Col2D<float>(num_kernels, input[b], map, inputHeight,
                            inputWidth, outputWidth, columns.size(1),
                            kernel_size, interpolation, columns);
    }

    // Use PyTorch for the rest
    // Compute the convolution output
    weight_grad += grad_output[b]
                       .view({nOutputPlanes, -1})
                       .mm(columns.transpose(1, 0))
                       .view({nOutputPlanes, nInputPlanes, kernel_size});
  }

  return weight_grad;
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv