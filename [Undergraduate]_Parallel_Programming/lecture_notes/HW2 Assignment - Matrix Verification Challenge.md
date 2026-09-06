---
title: "HW2 Assignment - Matrix Verification Challenge"
course: "Parallel Programming"
type: "assignment"
tags:
  - parallel-programming
  - homework
  - gemm
  - freivalds
---

# HW2 Assignment - Matrix Verification Challenge

이전: [[15 More Notes - DL Compiler와 LLM Inference]]  
다음: [[HW2 Report - Parallel GEMM과 Freivalds 최적화]]

## 핵심 요약

HW2는 행렬곱 검증 문제를 두 방식으로 구현하는 과제다. 하나는 병렬 GEMM으로 `A * B`를 직접 계산해 `C`와 비교하는 방식이고, 다른 하나는 Freivalds algorithm으로 확률적 검증을 수행하는 방식이다. 구현 파일은 `hw2/parallel.h`이며, 성능과 correctness가 모두 중요하다.

## 문제 배경

세 개의 `N x N` 행렬 `A`, `B`, `C`가 있을 때 `A * B == C`인지 검증해야 한다.

| 접근 | 아이디어 |
|---|---|
| Parallel GEMM | `A * B` 전체를 병렬 계산한 뒤 `C`와 비교 |
| Freivalds Algorithm | random vector `v`를 이용해 `A(Bv) == Cv`인지 검사 |

## 구현 대상

- `GEMV`: General Matrix-Vector Multiplication
- `GEMM`: General Matrix-Matrix Multiplication
- `init_vec`: Freivalds 검증에 사용할 vector 초기화
- 병렬 programming technique을 사용해 multi-thread 성능 개선

## Freivalds Algorithm

기본 절차:

1. 0/1 random vector `v` 생성
2. `Bv` 계산
3. `A(Bv)` 계산
4. `Cv` 계산
5. 두 vector가 같으면 `AB=C`일 가능성이 높다고 판단

`AB != C`인데 통과할 확률은 한 번 검사에서 최대 `1/2`이고, `k`번 반복하면 최대 `(1/2)^k`로 감소한다.

## 보고서 요구사항

보고서에는 다음을 포함해야 한다.

- Parallel algorithm 구현 방식
- 두 접근법 중 어떤 방식이 더 나은지
- Freivalds algorithm이 probabilistic인 이유와 error bound
- 프로그램이 최상의 성능을 내는 이유와 추가 여지가 있는지
- 표 또는 그래프를 포함한 evaluation
- 추가 분석

## 제출 및 채점 조건

- grading 시 `hw2/parallel.h`만 복사해 사용
- grading server에서 5회 실행 중 maximum 기준으로 speed 측정
- 출력 형식을 바꾸면 안 됨
- deadline 이후 timestamp 변경 시 late 처리 가능
- local에서만 동작했다는 이유로 재채점되지 않음

## 관련 강의 연결

- [[02 Thread Programming - C++ Thread와 동기화]]
- [[03 Matrix Multiplication - CPU Cache와 병렬 행렬곱]]

## 정리

HW2의 핵심은 같은 검증 문제를 완전 계산 방식과 확률적 방식으로 비교하는 것이다. GEMM은 정확하지만 `O(N^3)` 비용이 크고, Freivalds는 확률적 error를 허용하는 대신 GEMV 중심으로 계산량을 크게 줄인다.
