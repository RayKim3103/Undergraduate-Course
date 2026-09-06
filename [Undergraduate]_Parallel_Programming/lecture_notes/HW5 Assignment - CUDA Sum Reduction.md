---
title: "HW5 Assignment - CUDA Sum Reduction"
course: "Parallel Programming"
type: "assignment"
tags:
  - parallel-programming
  - homework
  - cuda
  - reduction
---

# HW5 Assignment - CUDA Sum Reduction

이전: [[HW4 Report - Direct Convolution과 im2col GEMM]]  
다음: [[HW5 Reference - Optimizing Parallel Reduction in CUDA]]

## 핵심 요약

HW5는 CUDA sum reduction을 구현하는 과제다. 강의자료에서 제공한 7가지 reduction kernel version을 실행할 수 있도록 kernel invocation을 작성하고, 최종 제출에서는 `2^24 = 16777216`개 item reduction에 대해 가장 빠른 version을 사용해야 한다.

## Reduction 정의

Parallel reduction은 배열 원소를 결합해 하나의 값을 만드는 알고리즘이다.

예:

```text
[a0, a1, a2, ..., an] -> sum
```

## 제공 변수

| 변수 | 의미 |
|---|---|
| `g_idata` | host input sequence |
| `g_odata` | host output buffer |
| `d_idata` | device input sequence |
| `d_odata` | device output buffer, 최종 결과는 `d_odata[0]` |

입력 sequence의 총합은 32bit signed integer 범위를 넘지 않도록 보장된다.

## 구현 범위

`main()`은 수정할 수 없고, 필요한 추가 allocation은 `reduction_optimized()` 내부에서 수행해야 한다. Host-device copy와 기본 device memory allocation/deallocation은 main에서 처리된다.

## 실행 및 실험

제공된 Makefile로 여러 version을 실행한다.

```bash
make 1
make 2
make 3
make 4
make run
```

목표는 7가지 version을 모두 이해하고, 최종적으로 가장 빠른 version을 선택하는 것이다.

## 채점 조건

- 출력 형식 변경 금지
- CUDA library 사용 금지
- deadline 이후 timestamp 변경 주의
- grading server에서 5회 실행 중 maximum 기준
- local congestion 또는 server congestion을 고려해 여유 있는 성능 필요

## 관련 강의 연결

- [[08 CUDA Reduction - Parallel Reduction 최적화]]

## 정리

HW5는 CUDA reduction 최적화의 실습 과제다. 단순 합계 계산이지만, divergent branch, bank conflict, idle thread, loop unrolling, multiple adds per thread, kernel decomposition을 모두 확인해야 한다.
