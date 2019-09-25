#ifndef VOTING_RESAMPLE_LAYER_H_
#define VOTING_RESAMPLE_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace nn {

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {

at::Tensor VotingResampleToMap(at::Tensor input, at::Tensor sample_map,
                               int outputHeight, int outputWidth,
                               int numCandidates);
}  // namespace cuda
#endif

namespace cpu {

at::Tensor VotingResampleToMap(at::Tensor input, at::Tensor sample_map,
                               int outputHeight, int outputWidth,
                               int numCandidates);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

at::Tensor VotingResampleToMap(at::Tensor input, at::Tensor sample_map,
                               int outputHeight, int outputWidth,
                               int numCandidates) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sample_map);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.type().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(sample_map);
    return cuda::VotingResampleToMap(input, sample_map, outputHeight,
                                     outputWidth, numCandidates);
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(sample_map);
    return cpu::VotingResampleToMap(input, sample_map, outputHeight,
                                    outputWidth, numCandidates);
  }
}

}  // namespace nn
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voting_resample_to_map", &mapped_conv::nn::VotingResampleToMap,
        "Resampling operation");
}

#endif