#include <ATen/ATen.h>

#include <vector>

#include "util/cuda/voting_resample_to_sphere.cuh"

namespace mapped_conv {
namespace util {
namespace cuda {

at::Tensor ResampleToSphereWithVoting(at::Tensor input, at::Tensor sample_map,
                                      int outputHeight, int outputWidth,
                                      int num_options) {
  // Useful dimensions to have
  const int64_t batchSize   = input.size(0);
  const int64_t channels    = input.size(1);
  const int64_t inputHeight = input.size(2);
  const int64_t inputWidth  = input.size(3);

  // Initialize output and index mask
  at::Tensor output = at::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());
  at::Tensor votes = at::zeros(
      {channels, outputHeight, outputWidth, num_options}, input.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    votes.fill_(0);
    ResampleToSphereWithVotingLauncher(input[b], sample_map, channels,
                                       inputHeight, inputWidth, outputHeight,
                                       outputWidth, num_options, votes);
    output[b] = votes.argmax(-1);
  }

  return output;
}

}  // namespace cuda
}  // namespace util
}  // namespace mapped_conv