---
title: "HW4 Assignment - CUDA Conv2d"
course: "Parallel Programming"
type: "assignment"
tags:
  - parallel-programming
  - homework
  - cuda
  - convolution
---

# HW4 Assignment - CUDA Conv2d

이전: [[HW3 Report - CUDA LoRA Tiled MatMul 최적화]]  
다음: [[HW4 Report - Direct Convolution과 im2col GEMM]]

## 핵심 요약

HW4는 CUDA로 convolution을 두 방식으로 구현하는 과제다. 첫째는 direct Conv2d kernel이고, 둘째는 im2col로 convolution을 matrix multiplication으로 변환한 뒤 matmul을 수행하는 방식이다. 제출 대상은 `conv.h`이며, filter size가 구현에 미치는 영향을 신중히 고려해야 한다.

## 구현 방식

| 방식 | 설명 |
|---|---|
| Parallel Conv2d | GPU에서 convolution을 직접 계산 |
| im2col + Matmul | input patch를 column matrix로 펼친 뒤 GEMM 수행 |

## 문제 조건

- Grading 중 input size는 고정
- input value는 달라질 수 있음
- batch size는 항상 1
- filter size가 memory access와 output size에 미치는 영향을 고려해야 함

## 구현 대상

`conv.h` 안의 세 함수를 구현한다.

예상 구성:

- direct convolution
- im2col 변환
- matmul 기반 convolution

## 채점 포인트

- Correctness
- CUDA kernel 성능
- memory access pattern 최적화
- strict performance benchmark 만족

## 규칙

- 지정된 HW4 directory 구조 유지
- `make run` 정상 실행
- 출력 문구 변경 금지
- deadline 이후 timestamp 변경 주의
- CUDA library 사용 금지
- file permission 유지

## 관련 강의 연결

- [[05 CUDA Matrix Multiplication - Shared Memory Tiling]]
- [[06 CUDA Transpose and Bank Conflict - Shared Memory 심화]]
- [[07 CUDA DNN - Convolution과 im2col]]

## 정리

HW4는 convolution 자체와 GEMM으로 변환한 convolution을 비교하는 과제다. Direct 방식은 중간 memory가 적지만 최적화가 어렵고, im2col+GEMM은 matrix multiplication 최적화를 재사용할 수 있지만 중간 데이터 변환 비용이 생긴다.
