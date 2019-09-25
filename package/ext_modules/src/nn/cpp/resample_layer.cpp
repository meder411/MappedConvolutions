#include "nn/layers/resample_layer.h"
#include "nn/cpp/resample.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

at::Tensor ResampleToMap(at::Tensor input, at::Tensor sample_map,
                         int outputHeight, int outputWidth,
                         int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize   = input.size(0);
  const int64_t channels    = input.size(1);
  const int64_t inputHeight = input.size(2);
  const int64_t inputWidth  = input.size(3);

  // Initialize output and index mask
  at::Tensor output = at::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    if (input.dtype() == at::kDouble) {
      ResampleToMap2D<double>(channels * inputHeight * inputWidth, input[b],
                              sample_map, channels, inputHeight, inputWidth,
                              outputHeight, outputWidth, interpolation,
                              output[b]);
    } else if (input.dtype() == at::kFloat) {
      ResampleToMap2D<float>(channels * inputHeight * inputWidth, input[b],
                             sample_map, channels, inputHeight, inputWidth,
                             outputHeight, outputWidth, interpolation,
                             output[b]);
    }
  }

  return output;
}

at::Tensor ResampleFromMap(at::Tensor grad_output, at::Tensor sample_map,
                           int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize    = grad_output.size(0);
  const int64_t channels     = grad_output.size(1);
  const int64_t inputHeight  = sample_map.size(0);
  const int64_t inputWidth   = sample_map.size(1);
  const int64_t outputHeight = grad_output.size(2);
  const int64_t outputWidth  = grad_output.size(3);

  // Initialize output and index mask
  at::Tensor input = at::zeros({batchSize, channels, inputHeight, inputWidth},
                               grad_output.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == at::kDouble) {
      ResampleFromMap2D<double>(channels * inputHeight * inputWidth,
                                grad_output[b], sample_map, channels,
                                inputHeight, inputWidth, outputHeight,
                                outputWidth, interpolation, input[b]);
    } else if (grad_output.dtype() == at::kFloat) {
      ResampleFromMap2D<float>(channels * inputHeight * inputWidth,
                               grad_output[b], sample_map, channels,
                               inputHeight, inputWidth, outputHeight,
                               outputWidth, interpolation, input[b]);
    }
  }

  return input;
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv