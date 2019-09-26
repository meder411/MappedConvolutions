#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "nn/cuda/mapped_im2col.cuh"

namespace mapped_conv {
namespace nn {
namespace cuda {

/*
The transposed forward pass works by leveraging the same GEMM operation as the
backward pass w.r.t the inputs. The difference is in the arguments.

First, recall a traditional forward convolution operation. For each batch:
1. We form the matrix C from the image using the mapped_im2col function. C has
dimensions (ck^2, n) for c input channels, (square) filter size k, and n
filtering locations (e.g. pixels in the output)
2. We then perform the convolution as a matrix multiplication between our
weights W (dim: d, ck^2, for d output channels).
3. We get the resulting output O = WC, with dimensions (d, n). We can then
reshape n into the correct (h,w).


Now, recall the traiditonal backward pass w.r.t the inputs. For each batch:
1. Compute the matrix multiplication between the output gradients O (dim: d, n)
and the weights W (dim: d, ck^2). This gives the input gradients in column form
s.t. matrix C = W^T O
2. We then use the mapped_col2im function to decompose C back into the input
shape


The transposed convolution takes the same arguments as the traditional
forward convolution, but the GEMM operation is the same as the traditional
backward convolution. To make this work, we first need to transpose the
dimensions of the weight matrix, hence the term "transposed convolution." This
transpose swaps the input and output channels, so the weights matrix goes from
(d, c, kh, kw) --> (c, d, kh, kw). Then, for each batch:
1. Compute the matrix multiplication between the input I (dim: c, n) and the
weight matrix (dim: c, dk^2, note the change here). This gives the output in
column form s.t. matrix C = W^T I (C dims: dk^2, n)
2. Now redistribute the column matrix C into the output O (dim: d, n) using
the mapped_col2im function.

In this function call, we assume the weight matrix has already been transposed.
*/

torch::Tensor WeightedMappedTransposedConvForward(
    torch::Tensor input, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, torch::Tensor bias,
    int outputHeight, int outputWidth, int kernel_size, int interpolation) {
  // Useful dimensions to have
  const int64_t nInputPlanes     = weight.size(0);
  const int64_t nOutputPlanes    = weight.size(1);
  const int64_t inputHeight      = input.size(2);
  const int64_t inputWidth       = input.size(3);
  const int64_t batchSize        = input.size(0);
  const int64_t num_interp_pts   = interp_weights.size(3);
  const int64_t inputBatchStride = nInputPlanes * inputHeight * inputWidth;

  // Initialize output and temporary columns
  torch::Tensor output = torch::zeros(
      {batchSize, nOutputPlanes, outputHeight, outputWidth}, input.options());
  torch::Tensor columns =
      torch::zeros({kernel_size * nOutputPlanes, inputHeight * inputWidth},
                   input.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    // Get cuda stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Call the GEMM function (note that it expects column major matrices)
    const int64_t m = columns.size(1);
    const int64_t n = weight.size(1) * weight.size(2);
    const int64_t k = weight.size(0);
    if (input.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                  input.data<double>() + b * inputBatchStride, m,
                  weight.data<double>(), n, &beta, columns.data<double>(), m);
    } else if (input.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 0.0;
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                  input.data<float>() + b * inputBatchStride, m,
                  weight.data<float>(), n, &beta, columns.data<float>(), m);
    }
    CUDA_CHECK(cudaGetLastError())

    MappedCol2Im2DWeightedLauncher(columns, sample_map, interp_weights,
                                   nOutputPlanes, outputHeight, outputWidth,
                                   inputWidth, columns.size(1), kernel_size,
                                   interpolation, num_interp_pts, output[b]);

    // Use PyTorch to add the bias
    output[b] += bias.view({output[b].size(0), 1, 1});
  }

  return output;
}

torch::Tensor WeightedMappedTransposedConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor weight, int inputHeight,
    int inputWidth, int kernel_size, int interpolation) {
  // Useful dimensions to have
  const int64_t nInputPlanes   = weight.size(0);
  const int64_t nOutputPlanes  = weight.size(1);
  const int64_t outputHeight   = grad_output.size(2);
  const int64_t outputWidth    = grad_output.size(3);
  const int64_t num_interp_pts = interp_weights.size(3);
  const int64_t batchSize      = grad_output.size(0);

  // Initialize output and temporary columns
  torch::Tensor input_grad =
      torch::zeros({batchSize, nInputPlanes, inputHeight, inputWidth},
                   grad_output.options());
  torch::Tensor columns =
      torch::zeros({kernel_size * nOutputPlanes, inputHeight * inputWidth},
                   grad_output.options());

  // For each elt in batch, do:
  const int64_t inputBatchStride = nInputPlanes * inputHeight * inputWidth;
  for (int b = 0; b < batchSize; b++) {
    MappedIm2Col2DWeightedLauncher(grad_output[b], sample_map, interp_weights,
                                   nOutputPlanes, outputHeight, outputWidth,
                                   inputWidth, columns.size(1), kernel_size,
                                   interpolation, num_interp_pts, columns);

    // Get cuda stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Call the GEMM function (note that it expects column major matrices)
    const int64_t m = columns.size(1);
    const int64_t k = weight.size(1) * weight.size(2);
    const int64_t n = weight.size(0);
    if (grad_output.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<double>(), m, weight.data<double>(), k, &beta,
                  input_grad.data<double>() + b * inputBatchStride, m);
    } else if (grad_output.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 0.0;
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<float>(), m, weight.data<float>(), k, &beta,
                  input_grad.data<float>() + b * inputBatchStride, m);
    }
    CUDA_CHECK(cudaGetLastError())
  }

  return input_grad;
}

torch::Tensor WeightedMappedTransposedConvBackwardWeight(
    torch::Tensor grad_output, torch::Tensor sample_map,
    torch::Tensor interp_weights, torch::Tensor input, int kernel_size,
    int interpolation) {
  // Useful dimensions to have
  const int64_t nOutputPlanes    = grad_output.size(1);
  const int64_t nInputPlanes     = input.size(1);
  const int64_t outputHeight     = grad_output.size(2);
  const int64_t outputWidth      = grad_output.size(3);
  const int64_t inputHeight      = input.size(2);
  const int64_t inputWidth       = input.size(3);
  const int64_t num_interp_pts   = interp_weights.size(3);
  const int64_t batchSize        = grad_output.size(0);
  const int64_t inputBatchStride = nInputPlanes * inputHeight * inputWidth;

  // Initialize output and temporary columns
  torch::Tensor weight_grad = torch::zeros(
      {nInputPlanes, nOutputPlanes, kernel_size}, grad_output.options());
  torch::Tensor columns =
      torch::zeros({kernel_size * nOutputPlanes, inputHeight * inputWidth},
                   grad_output.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    // Create the column matrix from the grad output as we would for the input
    // in the standard conv_forward
    MappedIm2Col2DWeightedLauncher(grad_output[b], sample_map, interp_weights,
                                   nOutputPlanes, outputHeight, outputWidth,
                                   inputWidth, columns.size(1), kernel_size,
                                   interpolation, num_interp_pts, columns);

    // Get cuda stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Propagate the gradients from the outputs to the weights using GEMM
    // Note that GEMM expects column major matrices
    const int64_t m = weight_grad.size(1) * weight_grad.size(2);
    const int64_t n = weight_grad.size(0);
    const int64_t k = columns.size(1);
    if (grad_output.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 1.0;
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<double>(), k,
                  input.data<double>() + b * inputBatchStride, k, &beta,
                  weight_grad.data<double>(), m);
    } else if (grad_output.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 1.0;
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<float>(), k,
                  input.data<float>() + b * inputBatchStride, k, &beta,
                  weight_grad.data<float>(), m);
    }
    CUDA_CHECK(cudaGetLastError())
  }

  return weight_grad;
}

}  // namespace cuda
}  // namespace nn
}  // namespace mapped_conv