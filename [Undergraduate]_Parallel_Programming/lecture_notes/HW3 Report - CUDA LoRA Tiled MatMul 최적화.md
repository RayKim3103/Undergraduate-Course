---
title: "HW3 Report - CUDA LoRA Tiled MatMul 최적화"
course: "Parallel Programming"
type: "report"
tags:
  - parallel-programming
  - report
  - cuda
  - lora
  - shared-memory
---

# HW3 Report - CUDA LoRA Tiled MatMul 최적화

이전: [[HW3 Assignment - CUDA LoRA]]  
다음: [[HW4 Assignment - CUDA Conv2d]]

## 핵심 요약

이 보고서는 CUDA로 LoRA 연산을 구현하며 tiled matrix multiplication, shared memory, coalescing, bank conflict padding, non-square tiling, occupancy, branch 제거를 고려한 과정을 정리한다. 최종 LoRA는 `x @ W.T`, `x @ A.T`, 중간 결과 `@ B.T`, scaling/addition kernel로 구성된다.

## 목표

- CUDA GEMM 병렬 구현
- LoRA 연산 최적화
- RTX 3090 기준 hardware specification, register, shared memory, occupancy 고려

## Hardware Specification 기반 설계

보고서는 성능 분석을 위해 다음을 고려했다.

- thread 수 per SM
- shared memory usage per block
- register usage per thread/block
- max threads per block
- occupancy

Compile option으로 register usage를 확인했고, thread당 약 40 register 사용을 기준으로 block size별 occupancy를 비교했다.

## Tiled MatMul Kernel

Input A와 B를 받아 `A x B.T = C`를 계산하는 kernel을 작성했다. Tile size는 template으로 조절 가능하게 두었다.

적용 최적화:

- A/B tile을 shared memory에 저장
- B shared memory에는 bank conflict 완화를 위해 padding 추가
- global memory load에서 coalescing 고려
- thread가 C의 element 또는 tile 위치를 담당

## Coalescing 개선

B tile을 load할 때 `threadIdx.y`가 column을 담당하면 warp 내 thread들이 연속 주소를 읽지 못해 non-coalesced access가 된다. 이를 `threadIdx.x`가 연속 column을 담당하도록 바꾸어 load coalescing을 개선했다.

핵심 형태:

```cpp
int b_row = block_x * TILE_COL + thread_y;
int b_col = t + thread_x;
Bs[thread_y][thread_x] = B[b_row * K + b_col];
```

## Shared Memory B 접근 주의점

Load할 때와 compute할 때 B의 index 의미가 다르다. 잘못해서 `Bs[thread_y][k]`를 사용하면 A의 한 row가 B의 같은 row 일부하고만 곱해져 잘못된 결과가 나온다.

올바른 접근은 `Bs[thread_x][k]`처럼 B의 여러 row와 곱해지도록 구성하는 것이다.

## Non-Square Tiling

Non-square tile은 occupancy와 matrix shape에 맞춰 유리할 수 있지만 index 계산이 어려워진다.

문제점:

- tile shape와 block shape가 달라질 수 있음
- `threadIdx.x`, `threadIdx.y`가 boundary를 넘어 redundancy 발생
- transpose된 B 접근에서 load/compute index가 꼬이기 쉬움
- 일부 계산 결과가 버려지는 비효율 가능

실험 결과 occupancy를 100%로 유지하고 일부 redundant computation을 감수하는 방식이 근소하게 빨랐다.

## Scaling and Sum Kernel

LoRA 최종 합산은 element-wise operation이다.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
y[i] = a[i] + scale * b[i];
```

각 thread가 독립 element를 처리하므로 구조는 단순하다.

## 성능 최적화 판단

보고서에서 강조한 성능 요인:

- Shared memory로 global memory latency 완화
- Bank conflict padding
- Coalesced load/store
- Register와 shared memory 사용량 확인
- Branch divergence를 만들 수 있는 if-else 최소화
- Transpose kernel launch overhead를 줄이기 위해 matmul 내부에서 transpose 고려

## 남은 개선 여지

`x @ A.T @ B.T` 부분에서 non-square tiling이 완벽하지 않아 일부 redundant computation이 남았다. 결과 store 시 boundary check를 위한 if-else도 완전히 제거하지 못해 추가 performance margin이 존재한다고 분석했다.

## 정리

HW3 보고서의 핵심은 CUDA 최적화가 단일 기법이 아니라 memory access, shared memory, index mapping, occupancy, branch를 동시에 맞추는 일이라는 점이다. LoRA는 수식은 간단하지만 여러 GEMM shape가 달라 non-square tiling과 transpose 처리에서 실질적인 어려움이 발생한다.
