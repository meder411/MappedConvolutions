#ifndef KNN_LAYER_H_
#define KNN_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace mapped_conv {
namespace util {

#ifndef __NO_CUDA__
namespace cuda {
std::vector<torch::Tensor> KNNForward(torch::Tensor ref,  // B x D x N
                                      torch::Tensor query, const int64_t k);
}  // namespace cuda
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

std::vector<torch::Tensor> KNNForward(torch::Tensor ref, torch::Tensor query,
                                      const int64_t k) {
#ifndef __NO_CUDA__
  CHECK_INPUT(ref);
  CHECK_INPUT(query);

  return cuda::KNNForward(ref, query, k);
#else
  printf("CUDA must be enabled to run KNN\n");
#endif
}

}  // namespace util
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn_forward", &mapped_conv::util::KNNForward, "K-Nearest Neighbors");
}

#endif