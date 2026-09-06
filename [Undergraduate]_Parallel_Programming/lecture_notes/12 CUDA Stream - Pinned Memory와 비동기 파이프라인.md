---
title: "12 CUDA Stream - Pinned Memory와 비동기 파이프라인"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - stream
  - pinned-memory
---

# 12 CUDA Stream - Pinned Memory와 비동기 파이프라인

이전: [[11 Triton Introduction - Triton DSL과 Kernel Fusion]]  
다음: [[12 CUDA Stream Updated - 업데이트본 요약]]

## 핵심 요약

이 강의는 host-device memory copy 병목을 줄이기 위한 pinned memory와 CUDA stream을 다룬다. GPU kernel이 빠르더라도 PCIe/NVLink를 통한 H2D/D2H copy가 오래 걸리면 전체 시간이 copy에 지배된다. Pinned memory와 `cudaMemcpyAsync`, multiple stream을 사용하면 copy와 kernel 실행을 overlap하여 pipeline을 만들 수 있다.

## Memcpy 병목

8192x8192 transpose 예시에서 kernel 자체는 약 2.3ms인데, host-device copy는 각각 44ms 수준으로 더 크다. 이 경우 kernel만 최적화해도 전체 실행시간은 크게 줄지 않는다.

## Pinned Memory

Pinned memory는 OS가 page out하지 않도록 고정한 host memory다. GPU DMA가 안정적으로 접근할 수 있어 pageable memory보다 transfer가 빠르다.

사용 API:

```cpp
cudaMallocHost(&h_ptr, bytes);
```

또는 기존 malloc memory를 등록:

```cpp
h_ptr = malloc(bytes);
cudaHostRegister(h_ptr, bytes, 0);
```

예시에서는 H2D/D2H copy가 44ms에서 20ms 정도로 줄어드는 효과가 제시된다.

## Stream

CUDA stream은 비동기 작업 queue다. 서로 다른 stream의 copy와 kernel은 resource가 겹치지 않으면 overlap될 수 있다.

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
```

Kernel launch도 stream을 지정할 수 있다.

```cpp
MatMulKernel<<<dimGrid, dimBlock, 0, stream1>>>(...);
```

## Pipeline 구조

여러 matrix를 처리할 때 각 matrix에 대해 다음 흐름을 하나의 stream에 배치한다.

```text
H2D copy -> kernel -> D2H copy
```

그리고 여러 stream을 동시에 사용하면:

```text
stream1: M1 H2D -> M1 kernel -> M1 D2H
stream2: M2 H2D -> M2 kernel -> M2 D2H
stream3: M3 H2D -> M3 kernel -> M3 D2H
stream4: M4 H2D -> M4 kernel -> M4 D2H
```

Copy engine과 compute engine이 동시에 일할 수 있으면 전체 throughput이 좋아진다.

## NVVP 확인

NVIDIA Visual Profiler에서는 timeline을 통해 stream별 H2D, kernel, D2H가 실제로 overlap되는지 확인할 수 있다. Zoom in해서 copy와 kernel 사이의 빈 공간이 줄었는지 보는 것이 중요하다.

## 정리

CUDA stream 최적화는 kernel 내부 최적화가 아니라 application-level scheduling 최적화다. Copy가 kernel보다 훨씬 긴 workload에서는 pinned memory와 async stream pipeline이 전체 성능을 크게 좌우한다.
