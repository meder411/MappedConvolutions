#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include "cuda_helper.h"
#include "util/cuda/knn.cuh"

namespace mapped_conv {
namespace util {
namespace cuda {

std::vector<at::Tensor> KNNForward(at::Tensor ref,  // B x D x N
                                   at::Tensor query, const int64_t k) {
  const int64_t batch_size    = ref.size(0);
  const int64_t dim           = ref.size(1);
  const int64_t num_ref_pts   = ref.size(2);
  const int64_t num_query_pts = query.size(2);

  at::Tensor idx  = at::zeros({batch_size, k, num_query_pts},
                             ref.options().dtype(at::kLong));
  at::Tensor dist = at::zeros({batch_size, k, num_query_pts}, ref.options());

  for (int i = 0; i < batch_size; i++) {
    KNNLauncher(ref[i], num_ref_pts, query[i], num_query_pts, dim, k, dist[i],
                idx[i], at::cuda::getCurrentCUDAStream());
    CUDA_CHECK(cudaGetLastError());
  }

  return {idx, dist};
}

}  // namespace cuda
}  // namespace util
}  // namespace mapped_conv