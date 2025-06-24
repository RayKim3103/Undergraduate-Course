#include "cuda_runtime.h"

#define TILE_IM2COL_X 16        // im2col x dim TILE SIZE
#define TILE_IM2COL_Y 16        // im2col y dim TILE SIZE
#define TILE_MATMUL 8           // matmul TILE SIZE (square)
#define TILE_MATMUL_SHIFT 3     // matmul TILE shift (2^3 = 8)
#define TILE_CONV2D_X 16        // conv2d_direct x dim TILE SIZE
#define TILE_CONV2D_Y 16        // conv2d_direct y dim TILE SIZE
#define MAX_FILTER_SIZE 7       // Maximum filter size for conv2d_direct (e.g., 7x7)


/***** im2col not using shared memory *****/
// __global__ void im2col_kernel(const float* X, float* X_col,
//                               int N, int C, int H, int W,
//                               int kH, int kW, int padH, int padW,
//                               int strideH, int strideW, int dilH, int dilW,
//                               int H_out, int W_out) {
//     // compute thread index
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_output_positions = N * H_out * W_out;

//     if (idx >= total_output_positions) return;

//     // compute output coordinates, (n, h_out, w_out)
//     int n = idx / (H_out * W_out);
//     int h_out = (idx / W_out) % H_out;
//     int w_out = idx % W_out;

//     // start position of the input data (considering padding and stride)
//     int h_start = h_out * strideH - padH;
//     int w_start = w_out * strideW - padW;

//     // iterate over each channel and filter position
//     for (int c = 0; c < C; c++) {
//         for (int p = 0; p < kH; p++) {
//             for (int q = 0; q < kW; q++) {
//                 // coordinates of the input data (considering dilation)
//                 int h_idx = h_start + p * dilH;
//                 int w_idx = w_start + q * dilW;

//                 // compute the row index of the output matrix & column index
//                 int filter_length = kH * kW;
//                 int C_base = c * filter_length;
//                 int y = C_base + p * kW + q;
//                 int x = idx;

//                 // check the boundary of the input data
//                 if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
//                     X_col[y * total_output_positions + x] = X[n * (C * H * W) + c * (H * W) + h_idx * W + w_idx];
//                 } 
//                 else {
//                     X_col[y * total_output_positions + x] = 0.0f;
//                 }
//             }
//         }
//     }
// }

/***** im2col using shared memory *****/
__global__ void im2col_kernel_shared(const float* __restrict__ X, float* __restrict__ X_col,
                                     int N, int C, int H, int W,
                                     int kH, int kW, int padH, int padW,
                                     int strideH, int strideW, int dilH, int dilW,
                                     int H_out, int W_out) {
    // use shared memory for input data reuse
    __shared__ float X_sh[TILE_IM2COL_Y][TILE_IM2COL_X+1];

    // thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // compute output coordinates
    const int h_out = blockIdx.y * (TILE_IM2COL_Y - kH + 1) + ty;
    const int w_out = blockIdx.x * (TILE_IM2COL_X - kW + 1) + tx;
    const int n = blockIdx.z;

    // compute input coordinates (considering padding and stride)
    // but in this assignment, we don't need to use these values
    const int h_in_base = h_out * strideH - padH;
    const int w_in_base = w_out * strideW - padW;

    // load input tile into shared memory
    for (int c = 0; c < C; c++) {
        int h_in = h_in_base;       // h_in is height index of input matrix
        int w_in = w_in_base;       // w_in is width index of input matrix

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            X_sh[ty][tx] = X[(n * C + c) * H * W + h_in * W + w_in];
        } 
        else {
            X_sh[ty][tx] = 0.0f;
        }
        __syncthreads();

        // compute im2col for only valid output threads, considering filter size
        if (ty < (TILE_IM2COL_Y - kH + 1) && tx < (TILE_IM2COL_X - kW + 1) && h_out < H_out && w_out < W_out) {
            int idx = n * H_out * W_out + h_out * W_out + w_out;
            for (int p = 0; p < kH; p++) {
                for (int q = 0; q < kW; q++) {
                    int h_idx = ty + p * dilH;          // h_idx is height index of shared memory
                    int w_idx = tx + q * dilW;          // w_idx is width index of shared memory

                    int filter_length = kH * kW;
                    int C_base = c * filter_length;     // base index for filter, it computes the offset for each filter
                    int y = C_base + p * kW + q;        // filter index

                    // read data from shared memory and store it in output matrix
                    X_col[y * (N * H_out * W_out) + idx] = X_sh[h_idx][w_idx];
                }
            }
        }
        // __syncthreads();
    }
}

/***** Matrix Multiplication Kernel using shared memory *****/
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int M, int N, int K)
{
    __shared__ float As[TILE_MATMUL][TILE_MATMUL];
    __shared__ float Bs[TILE_MATMUL][TILE_MATMUL+1];    // avoid bank conflict

    // compute thread indices and coordinates of the output matrix
    const int row = (blockIdx.y * TILE_MATMUL) + threadIdx.y; // (blockIdx.y << TILE_MATMUL_SHIFT) + threadIdx.y;
    const int col = (blockIdx.x * TILE_MATMUL) + threadIdx.x; // (blockIdx.x << TILE_MATMUL_SHIFT) + threadIdx.x;

    float acc = 0.f;

    for (int t = 0; t < K; t += TILE_MATMUL) {
        // Load A, B TILE into shared memory
        As[threadIdx.y][threadIdx.x] = (row < M && t + threadIdx.x < K) ? A[row * K + (t + threadIdx.x)] : 0.f;

        Bs[threadIdx.y][threadIdx.x] = (col < N && t + threadIdx.y < K) ? B[(t + threadIdx.y) * N + col] : 0.f;

        __syncthreads();

        // Partial MAC operation
        #pragma unroll
        for (int k = 0; k < TILE_MATMUL; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

/***** conv2d_direct not using shared memory *****/
// __global__ void conv2d_direct_kernel(const float* __restrict__ x,
//                                      const float* __restrict__ w,
//                                      float*       __restrict__ y,
//                                      int N, int C, int H, int W,
//                                      int K_out,
//                                      int kH, int kW,
//                                      int padH, int padW,
//                                      int strideH, int strideW,
//                                      int dilH,   int dilW,
//                                      int H_out,  int W_out)
// {
//     // compute coordinates of output matrix
//     const int h_out = blockIdx.y * TILE_CONV2D_Y + threadIdx.y;
//     const int w_out = blockIdx.x * TILE_CONV2D_X + threadIdx.x;
//     const int nk    = blockIdx.z;                   // Fused (n, k) dimension
//     const int n     = nk / K_out;                   // Batch index, input image index
//     const int k     = nk % K_out;                   // Output channel index, filter index; K_out = 1 for this AS
// //    const int n = nk;                             // Batch index, input image index
// //    const int k = 0;                              // Output channel index, filter index; K_out = 1 for this AS

//     if (h_out >= H_out || w_out >= W_out) return;   // check boundaries

//     float acc = 0.f;

//     // all input channels & multiply + accumulation
//     for (int c = 0; c < C; ++c) {                                       // C=1 for this AS, only 1 channel
//         for (int p = 0; p < kH; ++p) {
//             const int h_in = h_out * strideH - padH + p * dilH;         // padH = 0, dilH = 1
//             if (h_in < 0 || h_in >= H) continue;                        // H = 2048

//             for (int q = 0; q < kW; ++q) {
//                 const int w_in = w_out * strideW - padW + q * dilW;     // padW = 0, dilW = 1
//                 if (w_in < 0 || w_in >= W) continue;                    // W = 2048

//                 const float x_val = x[((n * C + c) * H + h_in) * W + w_in];
//                 const float w_val = w[(((k * C + c) * kH + p) * kW) + q];
//                 // const float w_val = w[((c * kH + p) * kW) + q];
//                 acc += x_val * w_val;
//             }
//         }
//     }
//     y[((n * K_out + k) * H_out + h_out) * W_out + w_out] = acc;
// }

/***** conv2d_direct using shared memory *****/
__global__ void conv2d_direct_kernel_shared(const float* __restrict__ x,
                                    const float* __restrict__ w,
                                    float* __restrict__ y,
                                    int N, int C, int H, int W,
                                    int K_out,
                                    int kH, int kW,
                                    int padH, int padW,
                                    int strideH, int strideW,
                                    int dilH, int dilW,
                                    int H_out, int W_out)
{
    // Static shared memory declarations
    __shared__ float X_sh[TILE_CONV2D_Y][TILE_CONV2D_X + 1]; // Input tile
    __shared__ float W_sh[MAX_FILTER_SIZE][MAX_FILTER_SIZE + 1]; // Weight tile

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Output coordinates
    const int h_out = blockIdx.y * (TILE_CONV2D_Y - kH + 1) + ty * dilH;
    const int w_out = blockIdx.x * (TILE_CONV2D_X - kW + 1) + tx * dilW;
    const int nk = blockIdx.z;  // Fused (n, k) dimension
    const int n = nk / K_out;   // Batch index
    const int k = nk % K_out;   // Output channel index

    // Input coordinates adjusted for filter size
    const int h_in = h_out * strideH - padH;
    const int w_in = w_out * strideW - padW;

    float acc = 0.0f;

    // Load input tile into shared memory
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        X_sh[ty][tx] = x[((n * C) * H + h_in) * W + w_in];
    } 
    else {
        X_sh[ty][tx] = 0.0f;
    }

    // Load weight tile into shared memory
    if (ty < kH && tx < kW) {
        W_sh[ty][tx] = w[(k * kH + ty) * kW + tx];
    }

    __syncthreads();

    // Compute convolution only for valid output threads
    if (ty < (TILE_CONV2D_Y - kH + 1) && tx < (TILE_CONV2D_X - kW + 1) && h_out < H_out && w_out < W_out) {
        #pragma unroll
        for (int p = 0; p < kH; ++p) {
            for (int q = 0; q < kW; ++q) {
                acc += X_sh[ty + p][tx + q] * W_sh[p][q];
            }
        }
        __syncthreads();

        // Store result
        y[((n * K_out + k) * H_out + h_out) * W_out + w_out] = acc;
    }
}


/**
 * @brief Launches the im2col operation, which rearranges image blocks into columns.
 *
 * This function is typically used in convolutional neural networks (CNNs) to
 * transform input image data into a format suitable for matrix multiplication.
 *
 * @param x        Pointer to the input tensor of shape (N, C, H, W), where:
 *                 - N: Batch size
 *                 - C: Number of channels
 *                 - H: Height of the input
 *                 - W: Width of the input
 * @param N        Number of images in the batch.
 * @param C        Number of channels in the input tensor.
 * @param H        Height of the input tensor.
 * @param W        Width of the input tensor.
 * @param kH       Height of the convolution kernel (filter).
 * @param kW       Width of the convolution kernel (filter).
 * @param padH     Padding applied to the height dimension.
 * @param padW     Padding applied to the width dimension.
 * @param strideH  Stride along the height dimension.
 * @param strideW  Stride along the width dimension.
 * @param dilH     Dilation factor for the height dimension.
 * @param dilW     Dilation factor for the width dimension.
 * @param out      Pointer to the output tensor, which stores the rearranged
 *                 image blocks in column format.
 */
void launch_im2col(const float* x, int N, int C, int H, int W,
                   int kH, int kW, int padH, int padW, int strideH, int strideW,
                   int dilH, int dilW, float* out)
{
    /***** kernel launch when using im2col kernel not using shared memory *****/
    // const int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    // const int W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

    // int total_output_positions = N * H_out * W_out;
    // int threadsPerBlock = TILE_IM2COL_X * TILE_IM2COL_Y;
    // int blocksPerGrid = (total_output_positions + threadsPerBlock - 1) / threadsPerBlock;

    // im2col_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, out,
    //                                                   N, C, H, W,
    //                                                   kH, kW, padH, padW,
    //                                                   strideH, strideW,
    //                                                   dilH, dilW,
    //                                                   H_out, W_out);

    /***** kernel launch when using im2col kernel using shared memory *****/
    int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    int W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

    // thread block size is (TILE_IM2COL_X, TILE_IM2COL_Y) which fits to shared memory
    // grid size is (output_width/(TILE_SIZE_X - kW), output_height/(TILE_SIZE_Y - kH), N)
    dim3 threadsPerBlock(TILE_IM2COL_X, TILE_IM2COL_Y);
    dim3 blocksPerGrid((W_out + TILE_IM2COL_X - kW) / (TILE_IM2COL_X - kW + 1),
                       (H_out + TILE_IM2COL_Y - kH) / (TILE_IM2COL_Y - kH + 1),
                       N);

    // call im2col_kernel_shared kernel
    im2col_kernel_shared<<<blocksPerGrid, threadsPerBlock>>>(
        x, out, N, C, H, W, kH, kW, padH, padW, strideH, strideW, dilH, dilW, H_out, W_out);
}


/**
 * @brief Launches a matrix multiplication operation on the provided matrices.
 *
 * This function performs the matrix multiplication operation C = A * B, where:
 * - A is an MxK matrix.
 * - B is a KxN matrix.
 * - C is the resulting MxN matrix.
 *
 * @param A Pointer to the first input matrix (MxK).
 * @param B Pointer to the second input matrix (KxN).
 * @param C Pointer to the output matrix (MxN) where the result will be stored.
 * @param M Number of rows in matrix A and matrix C.
 * @param N Number of columns in matrix B and matrix C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 block(TILE_MATMUL, TILE_MATMUL);
    dim3 grid((N + TILE_MATMUL - 1) / TILE_MATMUL,
              (M + TILE_MATMUL - 1) / TILE_MATMUL);

    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

/** 
 * @brief Launches a 2D convolution operation using the direct convolution method.
 *
 * @param x Pointer to the input tensor of shape (N, C, H, W), where:
 *          - N: Batch size
 *          - C: Number of input channels
 *          - H: Height of the input
 *          - W: Width of the input
 * @param w Pointer to the weight tensor of shape (K, C, kH, kW), where:
 *          - K: Number of output channels
*          - C: Number of input channels
 *          - kH: Height of the kernel
 *          - kW: Width of the kernel
 * @param y Pointer to the output tensor of shape (N, K, outH, outW), where:
 *          - outH: Computed output height
 *          - outW: Computed output width
 * @param N Number of input batches.
 * @param C Number of input channels.
 * @param H Height of the input tensor.
 * @param W Width of the input tensor.
 * @param K Number of output channels.
 * @param kH Height of the convolution kernel.
 * @param kW Width of the convolution kernel.
 * @param padH Padding applied to the height dimension.
 * @param padW Padding applied to the width dimension.
 * @param strideH Stride applied to the height dimension.
 * @param strideW Stride applied to the width dimension.
 * @param dilH Dilation applied to the height dimension of the kernel.
 * @param dilW Dilation applied to the width dimension of the kernel.
 */
void launch_conv2d_direct(const float* x, const float* w, float* y,
                          int N, int C, int H, int W,
                          int K, int kH, int kW,
                          int padH, int padW, int strideH, int strideW,
                          int dilH, int dilW)
{
    /***** kernel launch when using conv2d_direct kernel not using shared memory *****/
    // const int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1; // compute output height considering filter size
    // const int W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1; // compute output width considering filter size

    // dim3 block(TILE_CONV2D_X, TILE_CONV2D_Y);
    // dim3 grid((W_out + TILE_CONV2D_X - 1) / TILE_CONV2D_X,
    //           (H_out + TILE_CONV2D_Y - 1) / TILE_CONV2D_Y,
    //           N * K);                         // fused (n,k) dimension → grid.z

    // conv2d_direct_kernel<<<grid, block>>>(x, w, y,
    //                                       N, C, H, W,
    //                                       K, kH, kW,
    //                                       padH, padW,
    //                                       strideH, strideW,
    //                                       dilH,  dilW,
    //                                       H_out, W_out);

    /***** kernel launch when using conv2d_direct kernel using shared memory *****/
    const int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    const int W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

    dim3 block(TILE_CONV2D_X , TILE_CONV2D_Y);
    dim3 grid((W_out + TILE_CONV2D_X - kW - 1) / (TILE_CONV2D_X - kW + 1),
              (H_out + TILE_CONV2D_Y -kH - 1) / (TILE_CONV2D_Y -kH + 1),
              N * K);

    conv2d_direct_kernel_shared<<<grid, block>>>(x, w, y,
                                          N, C, H, W,
                                          K, kH, kW,
                                          padH, padW,
                                          strideH, strideW,
                                          dilH, dilW,
                                          H_out, W_out);
}