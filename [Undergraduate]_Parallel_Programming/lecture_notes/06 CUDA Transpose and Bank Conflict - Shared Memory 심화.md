---
title: "06 CUDA Transpose and Bank Conflict - Shared Memory 심화"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - transpose
  - bank-conflict
---

# 06 CUDA Transpose and Bank Conflict - Shared Memory 심화

이전: [[05 CUDA Matrix Multiplication - Shared Memory Tiling]]  
다음: [[07 CUDA DNN - Convolution과 im2col]]

## 핵심 요약

이 강의는 dynamic shared memory, matrix transpose, shared memory bank conflict를 다룬다. Matrix transpose는 read와 write 중 하나가 비연속 접근이 되기 쉬워 coalescing 문제가 발생한다. Shared memory를 이용한 corner turning으로 global memory read/write를 모두 coalesced하게 만들 수 있지만, shared memory 내부 bank conflict를 padding으로 해결해야 한다.

## Dynamic Shared Memory

정적 shared memory:

```cpp
__shared__ float tile[32][32];
```

동적 shared memory:

```cpp
extern __shared__ float buffer[];
SomeKernel<<<grid, block, shared_bytes>>>(...);
```

특징:

- kernel launch 시 shared memory 크기를 지정한다.
- 1D array로 선언되므로 index 계산이 필요하다.
- 하나의 buffer를 여러 영역으로 나누어 사용할 수 있다.
- block size를 여러 값으로 바꿔 실험할 때 유용하다.

## Shared Memory Bank

Shared memory는 32개 bank로 구성되어 warp의 32 threads가 동시에 접근할 수 있게 한다. 각 bank는 보통 4B 단위로 mapping된다.

| 접근 패턴 | 결과 |
|---|---|
| 32 threads가 서로 다른 bank 접근 | 병렬 처리 |
| 여러 threads가 같은 bank의 다른 address 접근 | bank conflict, serialization |
| 여러 threads가 같은 address broadcast | architecture에 따라 broadcast 가능 |

## Matrix Transpose 문제

Transpose는 `B[col][row] = A[row][col]` 형태다. Row-major 배열에서는 read와 write 중 하나가 stride 접근이 된다.

| 시도 | read | write | 문제 |
|---|---|---|---|
| Try 0 | coalesced | not coalesced | write가 비연속 |
| Try 1 | not coalesced | coalesced | read가 비연속 |
| Try 2 | 2D block | 한쪽 비연속 | index는 명확하지만 한계 존재 |
| Try 3 | shared memory corner turning | global read/write 개선 | shared memory bank conflict 가능 |
| Try 4 | skew/padding 추가 | coalescing + conflict 완화 | shared memory index 계산 필요 |

## Corner Turning

Corner turning의 목적은 global memory 접근을 모두 coalesced하게 만드는 것이다.

1. 원본 matrix를 row-wise로 읽어 shared memory tile에 저장한다.
2. Shared memory tile 내부에서 transpose된 위치로 접근한다.
3. 결과 matrix에 row-wise로 write한다.

Global memory 관점에서는 read와 write가 모두 연속 접근이 된다.

## Bank Conflict와 Padding

Shared memory tile을 `[TILE][TILE]`로 두고 transpose access를 하면 warp threads가 같은 bank를 동시에 건드릴 수 있다. 해결책은 한 column의 폭을 1만큼 늘리는 padding이다.

```cpp
const int SKEW = 1;
extern __shared__ float buffer[];
// logical tile width = TILE + SKEW
```

이렇게 stride가 32의 배수에서 벗어나 bank mapping이 분산된다.

## 정리

Transpose는 memory coalescing과 bank conflict를 동시에 보여주는 대표 예제다. 좋은 CUDA kernel은 global memory access만 보는 것이 아니라, shared memory 내부 bank mapping까지 고려해야 한다. “coalescing을 만들고, bank conflict를 피하라”가 이 강의의 결론이다.
