#include "nn/layers/voting_resample_layer.h"
#include "nn/cpp/resample.h"

namespace mapped_conv {
namespace nn {
namespace cpu {

at::Tensor VotingResampleToMap(at::Tensor input, at::Tensor sample_map,
                               int outputHeight, int outputWidth,
                               int numCandidates) {
  // Useful dimensions to have
  const int64_t batchSize   = input.size(0);
  const int64_t channels    = input.size(1);
  const int64_t inputHeight = input.size(2);
  const int64_t inputWidth  = input.size(3);

  // Initialize output and temporary tensor to accumulate votes
  at::Tensor output = at::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    at::Tensor tmp = at::zeros(
        {channels, outputHeight, outputWidth, numCandidates}, input.options());
    ResampleToMap2DVoting<int64_t>(channels * inputHeight * inputWidth,
                                   input[b], sample_map, channels, inputHeight,
                                   inputWidth, outputHeight, outputWidth,
                                   numCandidates, tmp);

    // Compute the index with the most votes
    at::Tensor argmax = tmp.argmax(-1);

    // Copy the selected indices to the output
    output[b].copy_(argmax);
  }

  return output;
}

}  // namespace cpu
}  // namespace nn
}  // namespace mapped_conv
