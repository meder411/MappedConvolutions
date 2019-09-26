#include <torch/extension.h>

#include <vector>

#include "nn/cuda/mapped_max_pool.cuh"

namespace mapped_conv {
namespace nn {
namespace cuda {

std::vector<torch::Tensor> WeightedMappedMaxPoolForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, int kernel_size, int interpolation) {
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
  torch::Tensor indices =
      torch::zeros({batchSize, channels, outputHeight, outputWidth},
                   input.options().dtype(torch::kLong));

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    MappedMaxPool2DWeightedLauncher(
        input[b], sample_map, interp_weights, channels, inputHeight,
        inputWidth, outputHeight, outputWidth, kernel_size, interpolation,
        num_interp_pts, output[b], indices[b]);
  }

  return {output, indices};
}

torch::Tensor WeightedMappedMaxPoolBackward(
    torch::Tensor grad_output, torch::Tensor idx_mask,
    torch::Tensor sample_map, torch::Tensor interp_weights, int inputHeight,
    int inputWidth, int kernel_size, int interpolation) {
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
    MappedMaxUnpool2DWeightedLauncher(
        grad_output[b], idx_mask[b], sample_map, interp_weights, channels,
        inputHeight, inputWidth, outputHeight, outputWidth, kernel_size,
        interpolation, num_interp_pts, grad_input[b]);
  }

  return grad_input;
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv