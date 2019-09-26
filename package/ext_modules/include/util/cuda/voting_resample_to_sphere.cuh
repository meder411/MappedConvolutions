#ifndef VOTING_RESAMPLE_TO_SHERE_LAYER_CUH_
#define VOTING_RESAMPLE_TO_SHERE_LAYER_CUH_

#include <math.h>
#include <torch/extension.h>

#include "cuda_helper.h"

namespace mapped_conv {
namespace util {
namespace cuda {

template <typename T>
__global__ void ResampleToSphereWithVotingKernel(
    const int64_t n, const int64_t *__restrict__ data_in_ptr,
    const T *__restrict__ sample_map_ptr, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t num_options,
    int64_t *__restrict__ votes_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  const int64_t x_in = index % in_width;
  const int64_t y_in = (index / in_width) % in_height;
  const int64_t c    = (index / in_width / in_height) % channels;

  // Output image location given by the map
  const int64_t sample_map_idx = 2 * (y_in * in_width + x_in);

  // Data to uninterpolate
  const int64_t data = data_in_ptr[index];

  // Find the (fractional) location in the input image from which the mapped
  // convolution sampled at a given index of the kernel
  const T x = sample_map_ptr[sample_map_idx];
  const T y = sample_map_ptr[sample_map_idx + 1];

  // Find the nearest integer value
  int64_t x_int = llround(double(x));
  int64_t y_int = llround(double(y));

  // Check for validity of pixels
  const bool x_int_valid = x_int >= 0 && x_int <= out_width - 1;
  const bool y_int_valid = y_int >= 0 && y_int <= out_height - 1;

  // "Uninterpolate" the data value
  if (x_int_valid && y_int_valid) {
    votes_ptr[c * out_height * out_width * num_options +
              y_int * out_width * num_options + x_int * num_options + data] +=
        1;
  }
}

void ResampleToSphereWithVotingLauncher(
    torch::Tensor data_in,
    torch::Tensor sample_map,  // IH, IW, 2
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t num_options, torch::Tensor votes) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      sample_map.scalar_type(), "ResampleToSphereWithVotingKernel", ([&] {
        ResampleToSphereWithVotingKernel<scalar_t>
            <<<blocks, CUDA_NUM_THREADS>>>(
                num_kernels, data_in.data<int64_t>(),
                sample_map.data<scalar_t>(), channels, in_height, in_width,
                out_height, out_width, num_options, votes.data<int64_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace util
}  // namespace mapped_conv
#endif