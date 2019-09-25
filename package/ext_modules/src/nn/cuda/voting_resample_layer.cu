#include <ATen/ATen.h>

#include "nn/cuda/resample.cuh"

namespace mapped_conv {
namespace nn {
namespace cuda {

at::Tensor VotingResampleToMap(at::Tensor input, at::Tensor sample_map,
                               int outputHeight, int outputWidth,
                               int numCandidates) {
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
    at::Tensor tmp = at::zeros(
        {channels, outputHeight, outputWidth, numCandidates}, input.options());
    ResampleToMap2DVotingLauncher(input[b], sample_map, channels, inputHeight,
                                  inputWidth, outputHeight, outputWidth,
                                  numCandidates, tmp);

    // Compute the index with the most votes
    at::Tensor argmax = tmp.argmax(-1);

    // Copy the selected indices to the output
    output[b].copy_(argmax);
  }

  return output;
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv