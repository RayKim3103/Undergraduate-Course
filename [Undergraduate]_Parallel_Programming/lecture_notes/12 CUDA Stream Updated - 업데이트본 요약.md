---
title: "12 CUDA Stream Updated - 업데이트본 요약"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - stream
  - updated
---

# 12 CUDA Stream Updated - 업데이트본 요약

이전: [[12 CUDA Stream - Pinned Memory와 비동기 파이프라인]]  
다음: [[13 CUDA Debug and Profiling - cuda-gdb와 nvprof]]

## 핵심 요약

이 업데이트본은 CUDA stream 강의와 같은 주제인 pinned memory, asynchronous copy, stream pipeline, NVVP timeline 확인을 다룬다. 추출된 내용 기준으로 기존 stream 자료와 구조가 거의 같으므로, 이 노트는 업데이트본에서 확인해야 할 실무 포인트를 중심으로 정리한다.

## 확인해야 할 핵심 개념

| 개념 | 내용 |
|---|---|
| Pageable memory | 일반 host memory, DMA 전송 전 staging 비용 가능 |
| Pinned memory | page-locked host memory, H2D/D2H 전송 가속 |
| `cudaMemcpyAsync` | stream에 비동기 copy enqueue |
| CUDA stream | 작업 순서를 보존하는 비동기 queue |
| Pipeline | copy와 kernel을 겹쳐 전체 처리량 향상 |

## Pinned Memory API

```cpp
cudaMallocHost(&h_ptr, bytes);
cudaHostRegister(h_ptr, bytes, 0);
```

Pinned memory는 많이 쓰면 OS memory management에 부담이 되므로 필요한 buffer에 제한적으로 사용해야 한다.

## Stream Pipeline 작성 패턴

각 stream 안에서는 순서가 보장된다.

```cpp
cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_in, d_out);
cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream);
```

여러 stream 간에는 독립적으로 실행될 수 있으므로, data chunk 또는 여러 matrix를 stream별로 분배하면 overlap이 가능하다.

## 업데이트본 관점의 체크리스트

- Host buffer가 pinned memory인지 확인한다.
- `cudaMemcpyAsync`에 stream argument를 명시한다.
- Kernel launch에도 같은 stream을 지정해 한 data chunk의 순서를 유지한다.
- 불필요한 global device synchronization을 줄인다.
- NVVP 또는 profiler timeline에서 실제 overlap을 확인한다.

## 정리

업데이트본의 핵심은 stream 최적화를 코드에 넣었다고 끝나는 것이 아니라, profiler timeline에서 copy/compute overlap이 실제 발생하는지 검증해야 한다는 점이다.
