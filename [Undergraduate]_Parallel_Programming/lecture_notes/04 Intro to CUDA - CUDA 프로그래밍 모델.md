---
title: "04 Intro to CUDA - CUDA 프로그래밍 모델"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - gpu
  - warp
---

# 04 Intro to CUDA - CUDA 프로그래밍 모델

이전: [[03 Matrix Multiplication - CPU Cache와 병렬 행렬곱]]  
다음: [[05 CUDA Matrix Multiplication - Shared Memory Tiling]]

## 핵심 요약

이 강의는 NVIDIA GPU를 C/C++ 기반 CUDA로 프로그래밍하는 기본 흐름을 다룬다. host와 device memory를 구분하고, `cudaMalloc`, `cudaMemcpy`, kernel launch, grid/block/thread index 계산, warp scheduling, branch divergence, latency hiding, occupancy를 이해하는 것이 핵심이다.

## CPU와 GPU 역할

| 장치 | 잘하는 일 |
|---|---|
| CPU | branch 많고 control-heavy한 작업, random access, OS 제어 |
| GPU | 대량의 data-parallel 작업, 같은 연산을 많은 데이터에 반복 |

실제 프로그램은 CPU와 GPU를 함께 사용한다. CPU host가 device memory를 할당하고 데이터를 복사한 뒤 kernel을 launch하고, GPU가 대규모 병렬 계산을 수행한다.

## CUDA 기본 흐름

1. Host memory에 입력 준비
2. `cudaMalloc`으로 device memory 할당
3. `cudaMemcpy(..., cudaMemcpyHostToDevice)`로 입력 복사
4. Kernel launch
5. `cudaMemcpy(..., cudaMemcpyDeviceToHost)`로 결과 복사
6. Device memory 해제

```cpp
vecAddKernel<<<ceil(N / 256.0), 256>>>(A_d, B_d, C_d, N);
```

위 launch는 block당 256 threads를 사용하고, 전체 N개 element를 덮을 만큼 grid를 만든다.

## Kernel과 Index

```cpp
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
```

`__global__` 함수는 host에서 호출되고 device에서 실행된다. `threadIdx`, `blockIdx`, `blockDim`, `gridDim`으로 각 thread의 전역 index를 계산한다.

## Grid와 Block

| 개념 | 의미 |
|---|---|
| Thread | 실제 작업 단위 |
| Thread block | 함께 scheduling되며 shared memory와 `__syncthreads()`를 공유하는 단위 |
| Grid | kernel launch 전체 block 집합 |
| Warp | SM 내부 scheduling 단위, 32 threads |

Block size는 보통 warp size의 배수로 잡는다. 최대 thread block size는 일반적으로 1024이다.

## 2D 이미지 예시

Color to grayscale, image blur 같은 image kernel은 2D grid/block을 사용한다.

```cpp
int Col = threadIdx.x + blockIdx.x * blockDim.x;
int Row = threadIdx.y + blockIdx.y * blockDim.y;
```

Boundary check는 이미지 크기가 block tile로 나누어떨어지지 않을 때 out-of-bounds 접근을 막는다.

## CUDA Function Qualifier

| qualifier | 호출 위치 | 실행 위치 |
|---|---|---|
| `__host__` | host | host |
| `__global__` | host | device |
| `__device__` | device | device |

## SM과 Warp Scheduling

SM(Streaming Multiprocessor)은 block을 받아 warp 단위로 실행한다. Warp selector는 operand가 준비된 warp를 골라 실행한다. 준비된 warp가 없으면 SM이 idle 상태가 된다.

GPU는 context switching overhead 없이 많은 warp를 유지하여 memory latency를 숨긴다. 이것이 latency hiding이다.

## Branch Divergence

Warp 안 thread들이 서로 다른 branch를 선택하면 branch path가 시간적으로 나뉘어 실행되어 utilization이 낮아진다. `tid % 2`처럼 warp 내부에서 절반이 다른 branch를 타는 코드는 좋지 않다.

## Occupancy와 Resource Limit

Occupancy는 SM에 동시에 resident할 수 있는 active warp/thread 비율이다. 영향을 주는 요소:

- block당 thread 수
- register 사용량
- shared memory 사용량
- SM당 최대 threads/warps/blocks

`--ptxas-options=-v`를 사용하면 register 사용량 등 kernel resource 정보를 확인할 수 있다.

## 정리

CUDA의 핵심은 host-device 구조, grid/block/thread 계층, warp 단위 실행을 이해하는 것이다. 성능은 단순히 thread를 많이 만드는 것으로 결정되지 않고, branch divergence를 줄이고 occupancy를 확보하며 memory latency를 숨기는 방식으로 결정된다.
