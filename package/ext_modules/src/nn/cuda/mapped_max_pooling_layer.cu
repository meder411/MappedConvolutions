#include <ATen/ATen.h>

#include <vector>

#include "nn/cuda/mapped_max_pool.cuh"

namespace mapped_conv {
namespace nn {
namespace cuda {

std::vector<at::Tensor> MappedMaxPoolForward(at::Tensor input,
                                             at::Tensor sample_map,
                                             int kernel_size,
                                             int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize    = input.size(0);
  const int64_t channels     = input.size(1);
  const int64_t inputHeight  = input.size(2);
  const int64_t inputWidth   = input.size(3);
  const int64_t outputHeight = sample_map.size(0);
  const int64_t outputWidth  = sample_map.size(1);

  // Initialize output and index mask
  at::Tensor output = at::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());
  at::Tensor indices =
      at::zeros({batchSize, channels, outputHeight, outputWidth},
                input.options().dtype(at::kLong));

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    MappedMaxPool2DLauncher(input[b], sample_map, channels, inputHeight,
                            inputWidth, outputHeight, outputWidth, kernel_size,
                            interpolation, output[b], indices[b]);
  }

  return {output, indices};
}

at::Tensor MappedMaxPoolBackward(at::Tensor grad_output, at::Tensor idx_mask,
                                 at::Tensor sample_map, int inputHeight,
                                 int inputWidth, int kernel_size,
                                 int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize    = grad_output.size(0);
  const int64_t channels     = grad_output.size(1);
  const int64_t outputHeight = grad_output.size(2);
  const int64_t outputWidth  = grad_output.size(3);

  // Initialize output and index mask
  at::Tensor grad_input = at::zeros(
      {batchSize, channels, inputHeight, inputWidth}, grad_output.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    MappedMaxUnpool2DLauncher(grad_output[b], idx_mask[b], sample_map,
                              channels, inputHeight, inputWidth, outputHeight,
                              outputWidth, kernel_size, interpolation,
                              grad_input[b]);
  }

  return grad_input;
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv