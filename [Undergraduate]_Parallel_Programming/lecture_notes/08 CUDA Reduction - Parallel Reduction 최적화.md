---
title: "08 CUDA Reduction - Parallel Reduction 최적화"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - reduction
  - optimization
---

# 08 CUDA Reduction - Parallel Reduction 최적화

이전: [[07 CUDA DNN - Convolution과 im2col]]  
다음: [[09 CUDA Others - TensorCore와 CUDA Libraries]]

## 핵심 요약

이 강의는 CUDA parallel reduction을 7단계로 최적화하며 GPU 성능 병목을 읽는 방법을 보여준다. Reduction은 arithmetic intensity가 낮아 compute-bound가 아니라 bandwidth-bound에 가깝다. 따라서 목표 metric은 GFLOP/s보다 memory bandwidth가 된다.

## Reduction 문제

배열의 모든 원소를 하나의 값으로 합치는 연산이다.

큰 배열에서는 여러 block이 각각 partial sum을 만들고, 다시 partial sum을 reduce해야 한다. CUDA에는 모든 block을 한 번에 동기화하는 global synchronization이 없으므로 kernel decomposition이 필요하다.

## Kernel Decomposition

CUDA block 간 global sync가 없기 때문에, partial result를 global memory에 저장하고 다음 kernel launch에서 다시 reduce한다. Kernel launch 자체가 단계 사이의 synchronization 역할을 한다.

## 최적화 목표

Reduction은 원소 하나를 읽어 한 번 정도 더하는 낮은 arithmetic intensity를 가진다. 따라서 peak FLOPS보다 peak memory bandwidth에 가까워지는 것이 목표다.

## 단계별 최적화

| 단계 | 아이디어 | 해결하는 문제 |
|---|---|---|
| 1 | Interleaved addressing with divergent branch | 기본 shared memory reduction |
| 2 | Index 계산으로 modulo 제거 | branch divergence 감소 |
| 3 | Sequential addressing | bank conflict와 divergence 개선 |
| 4 | First add during global load | idle thread 감소, global read 절반화 |
| 5 | Last warp unrolling | warp 내부 sync/branch overhead 제거 |
| 6 | Complete unrolling with templates | loop overhead 제거, compile-time specialization |
| 7 | Multiple adds per thread | Brent's theorem 기반 algorithm cascading |

## Interleaved Addressing 문제

초기 reduction은 `tid % (2*s) == 0` 같은 조건을 사용한다. 이 방식은 modulo 연산 overhead와 warp divergence를 만든다.

개선된 interleaved addressing은 `index = 2 * s * tid`를 사용해 modulo를 제거한다.

## Sequential Addressing

`s = blockDim.x / 2`에서 시작해 절반씩 줄이며 `if (tid < s)`인 thread가 `sdata[tid] += sdata[tid+s]`를 수행한다. 이 방식은 shared memory 접근이 더 규칙적이고 bank conflict가 줄어든다.

## First Add During Load

처음부터 thread 하나가 global memory에서 두 원소를 읽어 더한 뒤 shared memory에 저장한다.

```cpp
sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
```

초기 iteration에서 절반 thread가 idle인 문제를 줄이고, block당 처리 원소 수도 증가한다.

## Last Warp Unrolling

`s <= 32`가 되면 하나의 warp 안에서만 reduction이 일어난다. 전통적 warp lockstep 가정에서는 `__syncthreads()`가 불필요하므로 마지막 warp를 unroll하여 instruction overhead를 줄인다.

단, Volta 이후 independent thread scheduling에서는 warp-level synchronization 가정이 약해져 `__syncwarp()` 같은 명시적 동기화를 고려해야 한다.

## Complete Unrolling과 Template

Block size를 template parameter로 넘기면 compiler가 compile time에 branch를 제거하고 loop를 unroll할 수 있다.

```cpp
template <unsigned int blockSize>
__global__ void reduce(...)
```

Runtime에는 switch문으로 block size별 specialized kernel을 호출한다.

## Algorithm Cascading

Brent's theorem 관점에서 너무 많은 thread가 너무 적은 일을 하면 cost가 커진다. 각 thread가 여러 원소를 sequential하게 더한 뒤 shared memory tree reduction에 참여하면 cost efficiency가 좋아진다.

## 정리

Reduction 최적화는 CUDA 성능 최적화의 축소판이다. divergent branch, bank conflict, idle thread, instruction overhead, launch decomposition, algorithmic cost를 모두 보여준다. 좋은 reduction kernel은 단순히 병렬 step 수를 줄이는 것이 아니라 memory bandwidth와 instruction overhead를 함께 최적화한다.
