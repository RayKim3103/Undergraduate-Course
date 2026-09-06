---
title: "HW2 Report - Parallel GEMM과 Freivalds 최적화"
course: "Parallel Programming"
type: "report"
tags:
  - parallel-programming
  - report
  - gemm
  - gemv
  - freivalds
---

# HW2 Report - Parallel GEMM과 Freivalds 최적화

이전: [[HW2 Assignment - Matrix Verification Challenge]]  
다음: [[HW3 Assignment - CUDA LoRA]]

## 핵심 요약

이 보고서는 GEMM과 Freivalds algorithm을 CPU multi-threading으로 구현하고 최적화한 내용을 정리한다. 주요 최적화는 thread 수 조정, row-wise 병렬화, B transpose, blocked matrix multiplication, cache locality 개선, false sharing 회피, loop unrolling이다.

## 목표

- GEMM과 GEMV 기반 Freivalds algorithm 병렬 구현
- Deep learning과 computer vision에서 중요한 행렬 연산의 성능 개선
- Cache와 thread scheduling을 고려한 CPU 최적화
- GEMM 접근과 Freivalds 접근의 성능 및 정확도 tradeoff 비교

## GEMM 구현 전략

### Row-wise 병렬화

`C[i][j]` 계산에서 row index `i`를 thread별로 나누었다. 이유는 다음과 같다.

- thread 간 dependency가 없다.
- 같은 C element를 여러 thread가 쓰지 않아 race condition이 없다.
- atomic 연산이 필요 없다.
- row-major access로 spatial locality가 좋다.

### Transpose와 Blocking

B matrix를 transpose하여 원래 column-wise 접근을 row-wise 접근으로 바꾸었다. 이후 block 단위로 계산해 A, B, C의 tile이 L1 cache에 들어가도록 했다.

Block size 결정 관점:

- L1d cache를 core당 약 32KiB로 보고 계산
- local A, local B, local C가 cache에 들어가야 함
- 실험적으로 `16 x 16` block size가 좋은 성능을 보임

## Freivalds / GEMV 구현 전략

Freivalds는 `A * (B * v) == C * v`를 검사한다. 핵심 계산은 GEMV다.

적용한 최적화:

- multi-threading
- B vector 또는 partial vector를 thread-local하게 복사해 L1 cache 활용
- blocked GEMV
- loop unrolling

`init_vec()` 실행 시간은 grading 성능 평가에 포함되지 않으므로 단순 single-thread loop로 0/1 vector를 생성했다.

## Freivalds가 확률적인 이유

만약 `AB=C`이면 Freivalds는 항상 참을 반환한다. 그러나 `AB != C`인 경우에도 특정 random vector가 차이를 숨길 수 있다. 한 번의 검사에서 false positive 확률은 최대 `1/2`이고, `k`번 독립 반복하면 error bound는 `(1/2)^k`가 된다.

## 두 접근 비교

| 기준 | Parallel GEMM | Freivalds |
|---|---|---|
| 정확성 | 결정적 | 확률적 |
| 계산량 | `O(N^3)` | 여러 GEMV, 대략 `O(kN^2)` |
| 검증 목적 성능 | 느릴 수 있음 | 훨씬 빠름 |
| 결과 행렬 필요 여부 | 계산 결과가 필요하면 적합 | 검증만 필요할 때 적합 |

검증만 목적이라면 Freivalds가 더 적은 연산으로 높은 확률의 correctness를 제공하므로 더 적합하다고 판단했다.

## 추가 성능 고찰

보고서는 GEMV advanced performance 기준을 달성하지 못한 이유도 분석한다.

고려한 요소:

- thread 수가 너무 많으면 scheduling/context switching overhead 증가
- cache locality가 부족하면 memory access가 병목
- false sharing 회피를 위해 local matrix/vector 사용
- loop unrolling으로 compiler SIMD 유도
- copy overhead와 cache hit gain 사이 균형 필요

## 정리

HW2 보고서의 핵심은 CPU 병렬화에서 thread 수만 늘리는 것이 답이 아니라는 점이다. 행렬곱은 memory layout, cache line, false sharing, block size, transpose 여부가 성능을 지배한다. 검증 문제에서는 Freivalds처럼 문제 목적에 맞는 algorithmic optimization이 가장 큰 차이를 만든다.
