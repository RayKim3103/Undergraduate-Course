#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "reduction.h"

////////////////////////////////////////////////////
// Define which kernel to execute (0 for reduce0, 1 for reduce1, ..., 6 for reduce6)
// Change this value to test different kernels
#define WHICH_KERNEL 0
////////////////////////////////////////////////////

void allocateDeviceMemory(void** M, int size)
{
    cudaError_t err = cudaMalloc(M, size);
    assert(err==cudaSuccess);
}


void deallocateDeviceMemory(void* M)
{
    cudaError_t err = cudaFree(M);
    assert(err==cudaSuccess);
}

void cudaMemcpyToDevice(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyHostToDevice);
    assert(err==cudaSuccess);
}

void cudaMemcpyToHost(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToHost);
    assert(err==cudaSuccess);
}

void reduce_ref(const int* const g_idata, int* const g_odata, const int n) {
    for (int i = 0; i < n; i++)
        g_odata[0] += g_idata[i];
}

// Reduction #1: Interleaved Addressing
__global__ void reduce0(const int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += ((tid + s) < blockDim.x) ? sdata[tid + s] : 0;
            // sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction #2: Interleaved Addressing With strided index and non-divergent branch
__global__ void reduce1(const int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += ((index + s) < blockDim.x) ? sdata[index + s] : 0;
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction #3: Sequential Addressing
__global__ void reduce2(const int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += (tid + s < blockDim.x) ? sdata[tid + s] : 0;
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction #4: First Add During Load
__global__ void reduce3(const int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = (i < n && i + blockDim.x < n) ? g_idata[i] + g_idata[i + blockDim.x] : (i < n ? g_idata[i] : 0);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += (tid + s < blockDim.x) ? sdata[tid + s] : 0;
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction #5: Unroll the Last Warp
__global__ void reduce4(const int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = (i < n && i + blockDim.x < n) ? g_idata[i] + g_idata[i + blockDim.x] : (i < n ? g_idata[i] : 0);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += (tid + s < blockDim.x) ? sdata[tid + s] : 0;
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int *sdata_volatile = sdata;
        sdata_volatile[tid] += sdata_volatile[tid + 32];
        sdata_volatile[tid] += sdata_volatile[tid + 16];
        sdata_volatile[tid] += sdata_volatile[tid + 8];
        sdata_volatile[tid] += sdata_volatile[tid + 4];
        sdata_volatile[tid] += sdata_volatile[tid + 2];
        sdata_volatile[tid] += sdata_volatile[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction #6: Completely Unrolled
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce5(const int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    sdata[tid] = (i < n && i + blockSize < n) ? g_idata[i] + g_idata[i + blockSize] : (i < n ? g_idata[i] : 0);
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction #7: Multiple Adds / Thread
template <unsigned int blockSize>
__global__ void reduce6(const int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += (i < n && i + blockSize < n) ? g_idata[i] + g_idata[i + blockSize] : (i < n ? g_idata[i] : 0);
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// invocation functions of the reduction kernels
void invoke_reduce0(int threads, int blocks, int smemSize, const int *d_idata, int *d_odata, unsigned int n) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    reduce0<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n);
}

void invoke_reduce1(int threads, int blocks, int smemSize, const int *d_idata, int *d_odata, unsigned int n) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    reduce1<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n);
}

void invoke_reduce2(int threads, int blocks, int smemSize, const int *d_idata, int *d_odata, unsigned int n) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    reduce2<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n);
}

void invoke_reduce3(int threads, int blocks, int smemSize, const int *d_idata, int *d_odata, unsigned int n) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    reduce3<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n);
}

void invoke_reduce4(int threads, int blocks, int smemSize, const int *d_idata, int *d_odata, unsigned int n) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    reduce4<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n);
}

void invoke_reduce5(int threads, int blocks, int smemSize, const int *d_idata, int *d_odata, unsigned int n) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    switch (threads) {
        case 512: reduce5<512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 256: reduce5<256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 128: reduce5<128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 64: reduce5<64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 32: reduce5<32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 16: reduce5<16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 8: reduce5<8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 4: reduce5<4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 2: reduce5<2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 1: reduce5<1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
    }
}

void invoke_reduce6(int threads, int blocks, int smemSize, const int *d_idata, int *d_odata, unsigned int n) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    switch (threads) {
        case 512: reduce6<512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 256: reduce6<256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 128: reduce6<128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 64: reduce6<64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 32: reduce6<32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 16: reduce6<16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 8: reduce6<8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 4: reduce6<4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 2: reduce6<2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
        case 1: reduce6<1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n); break;
    }
}

void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, int* const d_odata, const int n) {
    // TODO: Implement your CUDA code
    // Reduction result must be stored in d_odata[0]
    // You should run the best kernel in here but you must remain other kernels as evidence.
    
    const int threads = 256; // threads per block
    int blocks;
    
    // Calculate blocks based on WHICH_KERNEL
    if (WHICH_KERNEL >= 0 && WHICH_KERNEL <= 2) {
        // reduce0~reduce2: Each block processes 'threads' elements
        blocks = (n + threads - 1) / threads;
    } 
    else {
        // reduce3~reduce6: Each block processes 'threads * 2' elements
        blocks = (n + threads * 2 - 1) / (threads * 2);
    }

    int smemSize = threads * sizeof(int);

    // Allocate temporary buffer for intermediate results
    int *d_input = const_cast<int *>(d_idata); // Read-only input buffer
    int *d_output = d_odata; // Output buffer
    int current_n = n;

    // Select which kernel to execute based on WHICH_KERNEL
    switch (WHICH_KERNEL) {
        case 0: // reduce0
            while (blocks > 1) {
                invoke_reduce0(threads, blocks, smemSize, d_input, d_output, static_cast<unsigned int>(current_n));
                // Swap input and output buffers for the next iteration
                int *temp = d_input;
                d_input = d_output;
                d_output = temp;
                current_n = blocks;
                blocks = (current_n + threads - 1) / threads; // Adjusted for reduce0
            }
            invoke_reduce0(threads, 1, smemSize, d_input, d_odata, static_cast<unsigned int>(current_n));
            break;

        case 1: // reduce1
            while (blocks > 1) {
                invoke_reduce1(threads, blocks, smemSize, d_input, d_output, static_cast<unsigned int>(current_n));
                int *temp = d_input;
                d_input = d_output;
                d_output = temp;
                current_n = blocks;
                blocks = (current_n + threads - 1) / threads; // Adjusted for reduce1
            }
            invoke_reduce1(threads, 1, smemSize, d_input, d_odata, static_cast<unsigned int>(current_n));
            break;

        case 2: // reduce2
            while (blocks > 1) {
                invoke_reduce2(threads, blocks, smemSize, d_input, d_output, static_cast<unsigned int>(current_n));
                int *temp = d_input;
                d_input = d_output;
                d_output = temp;
                current_n = blocks;
                blocks = (current_n + threads - 1) / threads; // Adjusted for reduce2
            }
            invoke_reduce2(threads, 1, smemSize, d_input, d_odata, static_cast<unsigned int>(current_n));
            break;

        case 3: // reduce3
            while (blocks > 1) {
                invoke_reduce3(threads, blocks, smemSize, d_input, d_output, static_cast<unsigned int>(current_n));
                int *temp = d_input;
                d_input = d_output;
                d_output = temp;
                current_n = blocks;
                blocks = (current_n + threads * 2 - 1) / (threads * 2);
            }
            invoke_reduce3(threads, 1, smemSize, d_input, d_odata, static_cast<unsigned int>(current_n));
            break;

        case 4: // reduce4
            while (blocks > 1) {
                invoke_reduce4(threads, blocks, smemSize, d_input, d_output, static_cast<unsigned int>(current_n));
                int *temp = d_input;
                d_input = d_output;
                d_output = temp;
                current_n = blocks;
                blocks = (current_n + threads * 2 - 1) / (threads * 2);
            }
            invoke_reduce4(threads, 1, smemSize, d_input, d_odata, static_cast<unsigned int>(current_n));
            break;

        case 5: // reduce5
            while (blocks > 1) {
                invoke_reduce5(threads, blocks, smemSize, d_input, d_output, static_cast<unsigned int>(current_n));
                int *temp = d_input;
                d_input = d_output;
                d_output = temp;
                current_n = blocks;
                blocks = (current_n + threads * 2 - 1) / (threads * 2);
            }
            invoke_reduce5(threads, 1, smemSize, d_input, d_odata, static_cast<unsigned int>(current_n));
            break;

        case 6: // reduce6
        default:
            // Multi-kernel invocation for reduce6
            while (blocks > 1) {
                invoke_reduce6(threads, blocks, smemSize, d_input, d_output, static_cast<unsigned int>(current_n));
                int *temp = d_input;
                d_input = d_output;
                d_output = temp;
                current_n = blocks;
                blocks = (current_n + threads * 2 - 1) / (threads * 2);
            }
            invoke_reduce6(threads, 1, smemSize, d_input, d_odata, static_cast<unsigned int>(current_n));
            break;
    }
}