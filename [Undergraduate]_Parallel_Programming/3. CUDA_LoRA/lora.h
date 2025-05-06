#include "cuda_runtime.h"

/*
  You can change the kernel name and the function signature.
*/
// __global__ void lora_kernel(const float *x, const float *W, const float *A,
//                             const float *B, float *y, int BATCH, int IN_DIM,
//                             int OUT_DIM, int RANK, float scale) {}

template<int TILE_ROW, int TILE_COL>
__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[TILE_ROW][TILE_COL];
    __shared__ float Bs[TILE_ROW][TILE_COL+1];

    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    const int thread_x_T = thread_x & (TILE_ROW - 1);

    const int block_y = blockIdx.y;
    const int block_x = blockIdx.x;

    const int row = block_y * TILE_ROW + thread_y;
    const int col = block_x * TILE_COL + thread_x;
    float acc = 0.0f;

#pragma unroll 8
    for (int t = 0; t < K; t += TILE_COL)
    {
        int a_col = t + thread_x;
        As[thread_y][thread_x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = block_x * TILE_COL + thread_y;
        int b_col = t + thread_x;
        Bs[thread_y][thread_x] = (b_row < N && b_col < K) ? B[b_row * K + b_col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_COL; ++k)
            acc += As[thread_y][k] * Bs[thread_x_T][k];

        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// New fused kernel for the final GEMM and scaling
template<int TILE_ROW, int TILE_COL>
__global__ void gemm_tiled_n_scale(const float* __restrict__ A, const float* __restrict__ B, 
                                const float* __restrict__ linear, float* __restrict__ C, 
                                float scale, int M, int N, int K)
{
    __shared__ float As[TILE_ROW][TILE_COL];
    __shared__ float Bs[TILE_ROW][TILE_COL+1];

    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    const int thread_x_T = thread_x & (TILE_ROW - 1);

    const int block_y = blockIdx.y;
    const int block_x = blockIdx.x;

    const int row = block_y * TILE_ROW + thread_y;
    const int col = block_x * TILE_COL + thread_x;
    float acc = 0.0f;

#pragma unroll 8
    for (int t = 0; t < K; t += TILE_COL)
    {
        int a_col = t + thread_x;
        As[thread_y][thread_x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = block_x * TILE_COL + thread_y;
        int b_col = t + thread_x;
        Bs[thread_y][thread_x] = (b_row < N && b_col < K) ? B[b_row * K + b_col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_COL; ++k)
            acc += As[thread_y][k] * Bs[thread_x_T][k];

        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = linear[idx] + scale * acc;
    }
}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y, int B,
          int in_dim, int out_dim, int r, float scale) {
  /*
   Call the kernel here.
  */
  float* d_linear;
  float* d_tmp;
  cudaMalloc(&d_linear, B * out_dim * sizeof(float));
  cudaMalloc(&d_tmp,    B * r       * sizeof(float));

  // Compute out_linear = x @ W_T
  {
      constexpr int TILE_ROW = 16;
      constexpr int TILE_COL = 16;
      dim3 block(TILE_COL, TILE_ROW);
      dim3 grid((out_dim + TILE_COL - 1) / TILE_COL, (B + TILE_ROW - 1) / TILE_ROW);
      gemm_tiled<TILE_ROW, TILE_COL><<<grid, block>>>(d_x, d_W, d_linear, B, out_dim, in_dim);
  }

  // Compute tmp = x @ A_T
  {
      constexpr int TILE_ROW = 8;
      constexpr int TILE_COL = 16;
      dim3 block(TILE_COL, TILE_ROW);
      dim3 grid((r + TILE_COL - 1) / TILE_COL, (B + TILE_ROW - 1) / TILE_ROW);
      gemm_tiled<TILE_ROW, TILE_COL><<<grid, block>>>(d_x, d_A, d_tmp, B, r, in_dim);
  }

  // Compute y = linear + scale * (tmp @ B_T)
  {
      constexpr int TILE_ROW = 16;
      constexpr int TILE_COL = 8;
      dim3 block(TILE_COL, TILE_ROW);
      dim3 grid((out_dim + TILE_COL - 1) / TILE_COL, (B + TILE_ROW - 1) / TILE_ROW);
      gemm_tiled_n_scale<TILE_ROW, TILE_COL><<<grid, block>>>(d_tmp, d_B, d_linear, d_y, scale, B, out_dim, r);
  }

  cudaFree(d_linear);
  cudaFree(d_tmp);
}