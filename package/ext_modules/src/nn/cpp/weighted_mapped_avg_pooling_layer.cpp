#include "nn/layers/weighted_mapped_avg_pooling_layer.h"
#include "nn/cpp/mapped_avg_pool.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

torch::Tensor WeightedMappedAvgPoolForward(torch::Tensor input,
                                           torch::Tensor sample_map,
                                           torch::Tensor interp_weights,
                                           int kernel_size,
                                           int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize      = input.size(0);
  const int64_t channels       = input.size(1);
  const int64_t inputHeight    = input.size(2);
  const int64_t inputWidth     = input.size(3);
  const int64_t outputHeight   = sample_map.size(0);
  const int64_t outputWidth    = sample_map.size(1);
  const int64_t num_interp_pts = interp_weights.size(3);

  // Initialize output and index mask
  torch::Tensor output = torch::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    if (input.dtype() == torch::kDouble) {
      MappedAvgPool2DWeighted<double>(
          channels * outputHeight * outputWidth, input[b], sample_map,
          interp_weights, channels, inputHeight, inputWidth, outputHeight,
          outputWidth, kernel_size, interpolation, num_interp_pts, output[b]);
    } else if (input.dtype() == torch::kFloat) {
      MappedAvgPool2DWeighted<float>(
          channels * outputHeight * outputWidth, input[b], sample_map,
          interp_weights, channels, inputHeight, inputWidth, outputHeight,
          outputWidth, kernel_size, interpolation, num_interp_pts, output[b]);
    }
  }

  return output;
}

torch::Tensor WeightedMappedAvgPoolBackward(torch::Tensor grad_output,
                                            torch::Tensor sample_map,
                                            torch::Tensor interp_weights,
                                            int inputHeight, int inputWidth,
                                            int kernel_size,
                                            int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize      = grad_output.size(0);
  const int64_t channels       = grad_output.size(1);
  const int64_t outputHeight   = grad_output.size(2);
  const int64_t outputWidth    = grad_output.size(3);
  const int64_t num_interp_pts = interp_weights.size(3);

  // Initialize output and index mask
  torch::Tensor grad_input = torch::zeros(
      {batchSize, channels, inputHeight, inputWidth}, grad_output.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == torch::kDouble) {
      MappedAvgUnpool2DWeighted<double>(
          channels * outputHeight * outputWidth, grad_output[b], sample_map,
          interp_weights, channels, inputHeight, inputWidth, outputHeight,
          outputWidth, kernel_size, interpolation, num_interp_pts,
          grad_input[b]);
    } else if (grad_output.dtype() == torch::kFloat) {
      MappedAvgUnpool2DWeighted<float>(
          channels * outputHeight * outputWidth, grad_output[b], sample_map,
          interp_weights, channels, inputHeight, inputWidth, outputHeight,
          outputWidth, kernel_size, interpolation, num_interp_pts,
          grad_input[b]);
    }
  }

  return grad_input;
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv