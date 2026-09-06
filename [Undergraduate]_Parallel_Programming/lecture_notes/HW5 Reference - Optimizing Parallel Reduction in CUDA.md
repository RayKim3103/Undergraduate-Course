---
title: "HW5 Reference - Optimizing Parallel Reduction in CUDA"
course: "Parallel Programming"
type: "reference"
tags:
  - parallel-programming
  - cuda
  - reduction
  - nvidia
---

# HW5 Reference - Optimizing Parallel Reduction in CUDA

이전: [[HW5 Assignment - CUDA Sum Reduction]]  
다음: [[HW6 Assignment - Triton ResNet]]

## 핵심 요약

이 참고자료는 Mark Harris의 CUDA parallel reduction 최적화 자료다. Reduction kernel을 여러 version으로 발전시키며 memory bandwidth, divergent branching, bank conflict, loop unrolling, Brent's theorem 기반 cascading을 설명한다. HW5의 이론적 기준 자료다.

## Global Synchronization 문제

CUDA는 block 간 global synchronization을 kernel 내부에서 제공하지 않는다. 따라서 큰 배열 reduction은 block별 partial reduction을 수행한 뒤, 여러 kernel invocation으로 recursive하게 줄이는 방식이 필요하다.

## 성능 Metric

Reduction은 원소 하나를 읽고 덧셈 하나를 수행하는 낮은 arithmetic intensity를 가진다. 따라서 compute throughput보다 memory bandwidth가 적절한 성능 지표다.

## Version별 발전

| Version | 핵심 변화 | 효과 |
|---|---|---|
| reduce0 | interleaved addressing, divergent branch | baseline |
| reduce1 | modulo 제거, index 계산 | branch/instruction overhead 감소 |
| sequential addressing | stride 감소 방향 변경 | bank conflict와 divergence 완화 |
| first add during load | global load 시 두 원소 합산 | idle thread 감소 |
| last warp unroll | 마지막 warp에서 sync 제거 | loop/sync overhead 감소 |
| complete unroll | template block size | compile-time 최적화 |
| multiple adds per thread | thread당 여러 원소 처리 | Brent's theorem, cost efficiency |

## Warp Reduce

마지막 32개 thread만 남으면 warp 내부 reduction을 unroll한다.

```cpp
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
```

이 방식은 instruction overhead를 줄이지만, 최신 independent thread scheduling에서는 warp 동기화 안전성을 고려해야 한다.

## Template Unrolling

`blockSize`를 template parameter로 넘기면 compiler가 compile time에 branch를 평가하고 불필요한 코드를 제거한다.

```cpp
template <unsigned int blockSize>
__global__ void reduce(...)
```

Runtime에는 switch문으로 block size별 specialization을 호출한다.

## Parallel Complexity

Reduction은 work complexity가 `O(N)`이어야 sequential algorithm과 같은 총 작업량을 가진다. 단순히 `O(log N)` parallel step을 얻기 위해 너무 많은 processor/thread를 쓰면 cost가 `O(N log N)`이 되어 비효율적일 수 있다.

Brent's theorem은 각 thread가 `O(log N)` 정도의 sequential work를 하고, 전체 thread 수를 줄여 cost efficiency를 높이는 방향을 제시한다.

## Algorithmic vs Code Optimization

자료는 algorithmic optimization이 loop unrolling 같은 code optimization보다 더 큰 speedup을 만들 수 있음을 보여준다.

- Algorithmic: addressing 변경, cascading
- Code: loop unrolling, template specialization

## 정리

이 참고자료의 핵심은 CUDA 성능 최적화가 병목 유형에 따라 달라진다는 점이다. Reduction은 memory-bound이고 instruction overhead도 중요하므로, memory access pattern과 algorithmic work efficiency를 먼저 잡고, 이후 unrolling 같은 code optimization을 적용해야 한다.
