#include <ATen/ATen.h>

#include "nn/cuda/resample.cuh"

namespace mapped_conv {
namespace nn {
namespace cuda {

at::Tensor WeightedResampleToMap(at::Tensor input, at::Tensor sample_map,
                                 at::Tensor interp_weights, int outputHeight,
                                 int outputWidth, int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize      = input.size(0);
  const int64_t channels       = input.size(1);
  const int64_t inputHeight    = input.size(2);
  const int64_t inputWidth     = input.size(3);
  const int64_t num_interp_pts = interp_weights.size(2);

  // Initialize output and index mask
  at::Tensor output = at::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    ResampleToMap2DWeightedLauncher(input[b], sample_map, interp_weights,
                                    channels, inputHeight, inputWidth,
                                    outputHeight, outputWidth, interpolation,
                                    num_interp_pts, output[b]);
  }

  return output;
}

at::Tensor WeightedResampleFromMap(at::Tensor grad_output,
                                   at::Tensor sample_map,
                                   at::Tensor interp_weights,
                                   int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize      = grad_output.size(0);
  const int64_t channels       = grad_output.size(1);
  const int64_t inputHeight    = sample_map.size(0);
  const int64_t inputWidth     = sample_map.size(1);
  const int64_t outputHeight   = grad_output.size(2);
  const int64_t outputWidth    = grad_output.size(3);
  const int64_t num_interp_pts = interp_weights.size(2);

  // Initialize output and index mask
  at::Tensor input = at::zeros({batchSize, channels, inputHeight, inputWidth},
                               grad_output.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    ResampleFromMap2DWeightedLauncher(grad_output[b], sample_map,
                                      interp_weights, channels, inputHeight,
                                      inputWidth, outputHeight, outputWidth,
                                      interpolation, num_interp_pts, input[b]);
  }

  return input;
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv