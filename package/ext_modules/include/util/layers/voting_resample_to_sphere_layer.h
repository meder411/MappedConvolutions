#ifndef VOTING_RESAMPLE_TO_SHERE_LAYER_H_
#define VOTING_RESAMPLE_TO_SHERE_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace util {

#ifndef __NO_CUDA__
namespace cuda {
std::vector<torch::Tensor> ResampleToSphereWithVoting(torch::Tensor input,
                                                      torch::Tensor sample_map,
                                                      int outputHeight,
                                                      int outputWidth,
                                                      int num_options);
}  // namespace cuda
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

std::vector<torch::Tensor> ResampleToSphereWithVoting(torch::Tensor input,
                                                      torch::Tensor sample_map,
                                                      int outputHeight,
                                                      int outputWidth,
                                                      int num_options) {
#ifndef __NO_CUDA__
  CHECK_INPUT(input);
  CHECK_INPUT(sample_map);

  return cuda::ResampleToSphereWithVoting(input, sample_map, outputHeight,
                                          outputWidth, num_options);
#else
  printf("CUDA must be enabled to run ResampleToSphereWithVoting\n");
#endif
}

}  // namespace util
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("resample_to_sphere_with_voting",
        &mapped_conv::util::ResampleToSphereWithVoting,
        "Resample to sphere with voting");
}

#endif