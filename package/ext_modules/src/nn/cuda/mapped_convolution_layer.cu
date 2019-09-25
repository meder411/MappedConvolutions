#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "nn/cuda/mapped_im2col.cuh"

namespace mapped_conv {
namespace nn {
namespace cuda {

at::Tensor MappedConvForward(at::Tensor input, at::Tensor sample_map,
                             at::Tensor weight, at::Tensor bias,
                             int64_t kernel_size, int64_t interpolation) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = weight.size(0);
  const int64_t nInputPlanes  = weight.size(1);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t outputHeight  = sample_map.size(0);
  const int64_t outputWidth   = sample_map.size(1);
  const int64_t batchSize     = input.size(0);

  // Initialize output and temporary columns
  at::Tensor output =
      at::zeros({batchSize, nOutputPlanes, outputHeight, outputWidth},
                input.options()) +
      10;
  at::Tensor columns =
      at::zeros({kernel_size * nInputPlanes, outputHeight * outputWidth},
                input.options());

  // For each elt in batch, do:
  const int64_t outputBatchStride = nOutputPlanes * outputHeight * outputWidth;
  for (int64_t b = 0; b < batchSize; b++) {
    // CUDA mapped_im2col
    MappedIm2Col2DLauncher(input[b], sample_map, nInputPlanes, inputHeight,
                           inputWidth, outputWidth, columns.size(1),
                           kernel_size, interpolation, columns);

    // Get cuda stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Call the GEMM function (note that it expects column major matrices)
    const int64_t m = columns.size(1);
    const int64_t k = weight.size(1) * weight.size(2);
    const int64_t n = weight.size(0);
    if (input.dtype() == at::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<double>(), m, weight.data<double>(), k, &beta,
                  output.data<double>() + b * outputBatchStride, m);
    } else if (input.dtype() == at::kFloat) {
      const float alpha = 1.0;
      const float beta  = 0.0;
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<float>(), m, weight.data<float>(), k, &beta,
                  output.data<float>() + b * outputBatchStride, m);
    }
    CUDA_CHECK(cudaGetLastError())

    // Use PyTorch to add the bias
    output[b] += bias.view({output[b].size(0), 1, 1});
  }

  return output;
}

at::Tensor MappedConvBackwardInput(at::Tensor grad_output,
                                   at::Tensor sample_map, at::Tensor weight,
                                   int64_t inputHeight, int64_t inputWidth,
                                   int64_t kernel_size,
                                   int64_t interpolation) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = weight.size(0);
  const int64_t nInputPlanes  = weight.size(1);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t batchSize     = grad_output.size(0);

  // Initialize output and temporary columns
  at::Tensor input_grad =
      at::zeros({batchSize, nInputPlanes, inputHeight, inputWidth},
                grad_output.options());
  at::Tensor columns =
      at::zeros({kernel_size * nInputPlanes, outputHeight * outputWidth},
                grad_output.options());

  // For each elt in batch, do:
  const int64_t outputBatchStride = nOutputPlanes * outputHeight * outputWidth;
  for (int64_t b = 0; b < batchSize; b++) {
    // Get cuda stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Call the GEMM function (note that it expects column major matrices)
    const int64_t m = columns.size(1);
    const int64_t n = weight.size(1) * weight.size(2);
    const int64_t k = weight.size(0);
    if (grad_output.dtype() == at::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                  grad_output.data<double>() + b * outputBatchStride,
                  m,                                  // lda=N
                  weight.data<double>(), n,           // ldb=ck^2
                  &beta, columns.data<double>(), m);  // ldc=N
    } else if (grad_output.dtype() == at::kFloat) {
      const float alpha = 1.0;
      const float beta  = 0.0;
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                  grad_output.data<float>() + b * outputBatchStride,
                  m,                                 // lda=N
                  weight.data<float>(), n,           // ldb=ck^2
                  &beta, columns.data<float>(), m);  // ldc=N
    }
    CUDA_CHECK(cudaGetLastError())

    MappedCol2Im2DLauncher(columns, sample_map, nInputPlanes, inputHeight,
                           inputWidth, outputWidth, columns.size(1),
                           kernel_size, interpolation, input_grad[b]);
  }

  return input_grad;
}

at::Tensor MappedConvBackwardWeight(at::Tensor grad_output,
                                    at::Tensor sample_map, at::Tensor input,
                                    int64_t kernel_size,
                                    int64_t interpolation) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = grad_output.size(1);
  const int64_t nInputPlanes  = input.size(1);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t batchSize     = grad_output.size(0);

  // Initialize output and temporary columns
  at::Tensor weight_grad = at::zeros(
      {nOutputPlanes, nInputPlanes, kernel_size}, grad_output.options());
  at::Tensor columns =
      at::zeros({kernel_size * nInputPlanes, outputHeight * outputWidth},
                grad_output.options());

  // For each elt in batch, do:
  const int64_t outputBatchStride = nOutputPlanes * outputHeight * outputWidth;
  for (int64_t b = 0; b < batchSize; b++) {
    // Create the column matrix from the input as we would in
    // mapped_conv_forward
    MappedIm2Col2DLauncher(input[b], sample_map, nInputPlanes, inputHeight,
                           inputWidth, outputWidth, columns.size(1),
                           kernel_size, interpolation, columns);

    // Get cuda stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Propagate the gradients from the outputs to the weights using GEMM
    // Note that GEMM expects column major matrices
    const int64_t m = weight_grad.size(1) * weight_grad.size(2);
    const int64_t n = weight_grad.size(0);
    const int64_t k = columns.size(1);
    if (grad_output.dtype() == at::kDouble) {
      const double alpha = 1.0;
      const double beta  = 1.0;
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<double>(), k,
                  grad_output.data<double>() + b * outputBatchStride, k, &beta,
                  weight_grad.data<double>(), m);
    }
    if (grad_output.dtype() == at::kFloat) {
      const float alpha = 1.0;
      const float beta  = 1.0;
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<float>(), k,
                  grad_output.data<float>() + b * outputBatchStride, k, &beta,
                  weight_grad.data<float>(), m);
    }
    CUDA_CHECK(cudaGetLastError())
  }

  return weight_grad;
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv