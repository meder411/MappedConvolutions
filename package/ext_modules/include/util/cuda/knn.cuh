/** Modifed version of knn-CUDA from
 * https://github.com/vincentfpgarcia/kNN-CUDA The modifications are removed
 * texture memory usage removed split query KNN computation added feature
 * extraction with bilinear interpolation
 *
 * Originally modified by Christopher B. Choy <chrischoy@ai.stanford.edu>
 * 12/23/2016
 * Last modified by Marc Eder <meder@cs.unc.edu> 1/29/2019
 */

#ifndef KNN_CUH_
#define KNN_CUH_

// Includes
#include <cuda.h>
#include <cstdio>

#include "cuda_helper.h"

#define IDX2D(i, j, dj) (dj * i + j)
#define IDX3D(i, j, k, dj, dk) (IDX2D(IDX2D(i, j, dj), k, dk))

// Constants used by the program
#define BLOCK_DIM 16
#define DEBUG 0

namespace mapped_conv {
namespace util {
namespace cuda {

/**
 * Computes the distance between two matrix A (reference points) and
 * B (query points) containing respectively wA and wB points.
 *
 * @param A     pointer on the matrix A
 * @param wA    width of the matrix A = number of points in A
 * @param B     pointer on the matrix B
 * @param wB    width of the matrix B = number of points in B
 * @param dim   dimension of points = height of matrices A and B
 * @param AB    pointer on the matrix containing the wA*wB distances computed
 */
template <typename T>
__global__ void cuComputeDistanceGlobal(T *A, int wA, T *B, int wB, int dim,
                                        T *AB) {
  // Declaration of the shared memory arrays As and Bs used to store the
  // sub-matrix of A and B
  __shared__ T shared_A[BLOCK_DIM][BLOCK_DIM];
  __shared__ T shared_B[BLOCK_DIM][BLOCK_DIM];

  // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Other variables
  T tmp;
  T ssd = 0;

  // Loop parameters
  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A  = BLOCK_DIM * wA;
  step_B  = BLOCK_DIM * wB;
  end_A   = begin_A + (dim - 1) * wA;

  // Conditions
  int cond0 = (begin_A + tx < wA);  // used to write in shared memory
  int cond1 =
      (begin_B + tx < wB);  // used to write in shared memory & to
                            // computations and to write in output matrix
  int cond2 = (begin_A + ty <
               wA);  // used to computations and to write in output matrix

  // Loop over all the sub-matrices of A and B required to compute the block
  // sub-matrix
  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    // Load the matrices from device memory to shared memory; each thread loads
    // one element of each matrix
    if (a / wA + ty < dim) {
      shared_A[ty][tx] = (cond0) ? A[a + wA * ty + tx] : 0;
      shared_B[ty][tx] = (cond1) ? B[b + wB * ty + tx] : 0;
    } else {
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Compute the difference between the two matrixes; each thread computes
    // one element of the block sub-matrix
    if (cond2 && cond1) {
      for (int k = 0; k < BLOCK_DIM; ++k) {
        tmp = shared_A[k][ty] - shared_B[k][tx];
        ssd += tmp * tmp;
      }
    }

    // Synchronize to make sure that the preceding computation is done before
    // loading two new sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory; each thread writes one
  // element
  if (cond2 && cond1) AB[(begin_A + ty) * wB + begin_B + tx] = ssd;
}

/**
 * Gathers k-th smallest distances for each column of the distance matrix in
 * the top.
 *
 * @param dist        distance matrix
 * @param ind         index matrix
 * @param width       width of the distance matrix and of the index matrix
 * @param height      height of the distance matrix and of the index matrix
 * @param k           number of neighbors to consider
 */
template <typename T>
__global__ void cuInsertionSort(T *dist, long *ind, int width, int height,
                                int k) {
  // Variables
  int l, i, j;
  T *p_dist;
  long *p_ind;
  T curr_dist, max_dist;
  long curr_row, max_row;
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (xIndex < width) {
    // Pointer shift, initialization, and max value
    p_dist   = dist + xIndex;
    p_ind    = ind + xIndex;
    max_dist = p_dist[0];
    p_ind[0] = 1;

    // Part 1 : sort kth firt elementZ
    for (l = 1; l < k; l++) {
      curr_row  = l * width;
      curr_dist = p_dist[curr_row];
      if (curr_dist < max_dist) {
        i = l - 1;
        for (int a = 0; a < l - 1; a++) {
          if (p_dist[a * width] > curr_dist) {
            i = a;
            break;
          }
        }
        for (j = l; j > i; j--) {
          p_dist[j * width] = p_dist[(j - 1) * width];
          p_ind[j * width]  = p_ind[(j - 1) * width];
        }
        p_dist[i * width] = curr_dist;
        p_ind[i * width]  = l + 1;
      } else {
        p_ind[l * width] = l + 1;
      }
      max_dist = p_dist[curr_row];
    }

    // Part 2 : insert element in the k-th first lines
    max_row = (k - 1) * width;
    for (l = k; l < height; l++) {
      curr_dist = p_dist[l * width];
      if (curr_dist < max_dist) {
        i = k - 1;
        for (int a = 0; a < k - 1; a++) {
          if (p_dist[a * width] > curr_dist) {
            i = a;
            break;
          }
        }
        for (j = k - 1; j > i; j--) {
          p_dist[j * width] = p_dist[(j - 1) * width];
          p_ind[j * width]  = p_ind[(j - 1) * width];
        }
        p_dist[i * width] = curr_dist;
        p_ind[i * width]  = l + 1;
        max_dist          = p_dist[max_row];
      }
    }
  }
}

/**
 * Computes the square root of the first line (width-th first element)
 * of the distance matrix.
 *
 * @param dist    distance matrix
 * @param width   width of the distance matrix
 * @param k       number of neighbors to consider
 */
template <typename T>
__global__ void cuParallelSqrt(T *dist, int width, int k) {
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if (xIndex < width && yIndex < k)
    dist[yIndex * width + xIndex] = sqrt(dist[yIndex * width + xIndex]);
}

//---------------------------------------------------------------------------//
//                        K-th NEAREST NEIGHBORS                             //
//---------------------------------------------------------------------------//

/**
 * K nearest neighbor algorithm
 * - Initialize CUDA
 * - Allocate device memory
 * - Copy point sets (reference and query points) from host to device memory
 * - Compute the distances + indexes to the k nearest neighbors for each query
 * point
 * - Copy distances from device to host memory
 *
 * @param ref_host      reference points ; pointer to linear matrix
 * @param num_ref_pts   number of reference points ; width of the matrix
 * @param query_host    query points ; pointer to linear matrix
 * @param num_query_pts number of query points ; width of the matrix
 * @param dim           dimension of points ; height of the matrices
 * @param k             number of neighbor to consider
 * @param dist_host     distances to k nearest neighbors ; pointer to linear
 * matrix
 * @param dist_host     indexes of the k nearest neighbors ; pointer to linear
 * matrix
 *
 */
void KNNLauncher(at::Tensor ref, int num_ref_pts, at::Tensor query,
                 int num_query_pts, int dim, int k, at::Tensor dist,
                 at::Tensor idx, cudaStream_t stream) {
  // Grids ans threads
  dim3 g_16x16(num_query_pts / 16, num_ref_pts / 16, 1);
  dim3 t_16x16(16, 16, 1);
  if (num_query_pts % 16 != 0) g_16x16.x += 1;
  if (num_ref_pts % 16 != 0) g_16x16.y += 1;
  //
  dim3 g_256x1(num_query_pts / 256, 1, 1);
  dim3 t_256x1(256, 1, 1);
  if (num_query_pts % 256 != 0) g_256x1.x += 1;

  dim3 g_k_16x16(num_query_pts / 16, k / 16, 1);
  dim3 t_k_16x16(16, 16, 1);
  if (num_query_pts % 16 != 0) g_k_16x16.x += 1;
  if (k % 16 != 0) g_k_16x16.y += 1;

  // Kernel 1: Compute all the distances
  AT_DISPATCH_FLOATING_TYPES(
      ref.type(), "cuComputeDistanceGlobal", ([&] {
        cuComputeDistanceGlobal<scalar_t><<<g_16x16, t_16x16, 0, stream>>>(
            ref.data<scalar_t>(), num_ref_pts, query.data<scalar_t>(),
            num_query_pts, dim, dist.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())

  // Kernel 2: Sort each column
  AT_DISPATCH_FLOATING_TYPES(
      ref.type(), "cuInsertionSort", ([&] {
        cuInsertionSort<<<g_256x1, t_256x1, 0, stream>>>(
            dist.data<scalar_t>(), idx.data<int64_t>(), num_query_pts,
            num_ref_pts, k);
      }));
  CUDA_CHECK(cudaGetLastError())

  // Kernel 3: Compute square root of k first elements
  AT_DISPATCH_FLOATING_TYPES(
      ref.type(), "cuInsertionSort", ([&] {
        cuParallelSqrt<<<g_k_16x16, t_k_16x16, 0, stream>>>(
            dist.data<scalar_t>(), num_query_pts, k);
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace util
}  // namespace mapped_conv
#endif