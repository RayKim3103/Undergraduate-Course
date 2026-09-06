---
title: "05 CUDA Matrix Multiplication - Shared Memory Tiling"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - matrix-multiplication
  - shared-memory
---

# 05 CUDA Matrix Multiplication - Shared Memory Tiling

이전: [[04 Intro to CUDA - CUDA 프로그래밍 모델]]  
다음: [[06 CUDA Transpose and Bank Conflict - Shared Memory 심화]]

## 핵심 요약

이 강의는 CUDA에서 matrix multiplication을 구현하고 shared memory tiling으로 global memory traffic을 줄이는 방법을 다룬다. 핵심은 각 thread가 C의 한 element를 계산하되, A와 B의 tile을 shared memory에 올려 block 내부 thread들이 재사용하게 만드는 것이다.

## CUDA Memory Model

| Memory | 범위 | 특징 |
|---|---|---|
| Register | thread private | 가장 빠름, 자동 변수 대부분 |
| Shared memory | block shared | 빠른 SRAM, 명시적 관리 |
| Global memory | grid 전체 | 크지만 느림 |
| Constant/Texture | 특수 read path | 접근 패턴에 따라 유리 |

GPU cache는 spatial locality는 어느 정도 잡지만, 많은 thread가 동시에 실행되므로 per-thread cache capacity가 작다. Matrix multiplication처럼 같은 데이터를 여러 번 재사용하는 경우 shared memory를 직접 쓰는 것이 중요하다.

## Simple CUDA MatMul

각 thread가 `P[row][col]` 하나를 계산한다.

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

float value = 0;
for (int k = 0; k < width; k++) {
    value += M[row * width + k] * N[k * width + col];
}
P[row * width + col] = value;
```

이 방식은 구현은 단순하지만 같은 M/N element를 여러 thread가 global memory에서 반복적으로 읽는다.

## 2D Block Strategy

`TILE_WIDTH x TILE_WIDTH` thread block 하나가 결과 matrix의 같은 크기 tile을 계산한다.

| 구성 | 의미 |
|---|---|
| `dimBlock(TILE_WIDTH, TILE_WIDTH)` | block 안 thread 배열 |
| `dimGrid(width/TILE_WIDTH, width/TILE_WIDTH)` | 결과 tile 개수 |
| `threadIdx.y` | 결과 row 내부 위치 |
| `threadIdx.x` | 결과 column 내부 위치 |

## Shared Memory Blocking

각 phase마다 A tile과 B tile을 shared memory에 load한다.

```cpp
__shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
__shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
```

흐름:

1. 각 thread가 global memory에서 A/B element 하나씩 load
2. `__syncthreads()`로 tile load 완료 대기
3. shared memory tile을 이용해 partial dot product 계산
4. `__syncthreads()`로 tile 사용 완료 대기
5. 다음 tile phase로 진행

## `__syncthreads()`

`__syncthreads()`는 같은 thread block 안의 barrier다. 모든 thread가 도착해야 다음으로 넘어간다. Tiled algorithm에서는 tile load와 tile consume 사이의 correctness를 보장한다.

주의:

- block 내부에서만 동작한다.
- 다른 block과는 synchronization하지 않는다.
- 조건문 안에서 일부 thread만 도달하면 deadlock이 될 수 있다.

## Memory Traffic 분석

Tile 크기가 `b`이면 각 tile element가 block 내부에서 `b`번 재사용된다. 전체 global memory load는 naive 대비 대략 `b`배 줄어들 수 있다.

단, tile을 크게 잡는다고 항상 좋은 것은 아니다.

제약:

- block당 최대 thread 수
- SM shared memory 용량
- register pressure
- occupancy

예를 들어 8x8, 16x16, 32x32 tile은 모두 occupancy 100%가 가능할 수 있지만, 32x32는 block당 1024 threads라 scheduling 유연성이 떨어질 수 있다.

## Coalescing

Global memory access는 warp의 thread들이 연속 주소를 읽을 때 효율적이다. Matrix multiplication에서 A와 B tile을 load할 때 `threadIdx.x`가 연속 주소를 담당하도록 배치해야 coalescing이 잘 일어난다.

## Corner Turning

데이터를 shared memory에 넣을 때는 coalesced read로 읽고, shared memory 내부에서 access pattern을 바꾸어 write 또는 compute에서도 효율을 얻는 방식이다. 이후 transpose 강의에서 bank conflict와 함께 더 자세히 다룬다.

## 정리

CUDA matmul 최적화의 핵심은 global memory에서 직접 dot product를 계산하지 않고, shared memory tile을 통해 데이터를 block 내부에서 재사용하는 것이다. 성능은 tile 크기, coalescing, bank conflict, occupancy, synchronization overhead의 균형으로 결정된다.
