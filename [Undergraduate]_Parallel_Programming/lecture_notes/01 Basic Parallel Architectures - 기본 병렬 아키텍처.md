---
title: "01 Basic Parallel Architectures - 기본 병렬 아키텍처"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - architecture
  - simd
  - mimd
---

# 01 Basic Parallel Architectures - 기본 병렬 아키텍처

이전: [[00 Course Overview - 병렬 프로그래밍 개요]]  
다음: [[02 Thread Programming - C++ Thread와 동기화]]

## 핵심 요약

이 강의는 병렬 컴퓨터 구조를 SISD, MIMD, SIMD 관점으로 소개한다. 핵심은 병렬성이 여러 층위에 존재한다는 점이다. Superscalar CPU는 명령어 수준 병렬성(ILP)을 자동으로 찾고, multi-core CPU는 thread 수준 병렬성을 요구하며, vector processor는 하나의 명령으로 여러 데이터를 처리한다.

## 병렬성의 층위

| 구조 | 분류 | 병렬성 | 프로그래머 관점 |
|---|---|---|---|
| Scalar / Superscalar CPU | SISD | ILP | 대부분 자동 |
| Multi-core CPU | MIMD | thread-level parallelism | thread 또는 OpenMP로 명시 |
| Vector processor | SIMD | data-level parallelism | compiler vectorization 또는 intrinsic |

## ILP와 Superscalar

Processor는 원래 program counter가 가리키는 instruction을 순서대로 실행하는 기계다. Superscalar processor는 여러 execution unit을 사용해 서로 독립적인 instruction을 같은 cycle에 실행한다.

예를 들어 `x*x`, `y*y`, `z*z`는 서로 독립적이면 동시에 실행될 수 있다. 그러나 dependency가 있는 instruction은 앞 instruction 결과를 기다려야 한다. 따라서 ILP에는 프로그램 내부 dependency가 만드는 한계가 있다.

## Multi-core Processor

Multi-core는 core를 여러 개 두어 서로 다른 instruction stream을 동시에 실행한다. 각 core 하나의 frequency가 single-core보다 낮을 수 있어도, 여러 core를 잘 활용하면 총 처리량이 증가한다.

하지만 기존 single-thread loop는 자동으로 여러 core에서 실행되지 않는다. multi-core를 쓰려면 thread로 일을 나누거나, OpenMP 같은 data parallel 표현을 사용해야 한다.

## Thread

Process는 실행 중인 프로그램 전체를 뜻하고 code, data, stack, register, PC 등을 가진다. Thread는 process 안의 실행 흐름 단위다. 같은 process의 thread들은 주소 공간을 공유하면서 각자 PC, register, stack을 가진다.

예시:

- 배열 `c[i] = k[i] * a[i] + k[i] * b[i]`는 각 `i`가 독립적이면 thread별로 index 범위를 나누기 좋다.
- loop iteration 사이 dependency가 없으면 data parallelism이 있다.

## Vector Processing

Vector processor는 하나의 instruction으로 여러 데이터를 동시에 처리한다. SSE, AVX2, AVX-512는 register 폭이 다르다.

| ISA | register 폭 | 32bit float 처리량 |
|---|---:|---:|
| SSE | 128bit | 4개 |
| AVX2 | 256bit | 8개 |
| AVX-512 | 512bit | 16개 |

Compiler가 vectorization을 자동 수행할 수도 있고, `<immintrin.h>` 같은 intrinsic을 직접 사용할 수도 있다.

## 조건문과 Predication

SIMD 구조에서 lane마다 branch 방향이 다르면 모든 lane을 같은 instruction stream으로 처리하기 어렵다. 이때 predication으로 조건별 결과를 mask 처리할 수 있지만, 실제 계산 자원 활용률이 떨어질 수 있다. 이 개념은 CUDA warp divergence와 직접 연결된다.

## 용어 정리

| 용어 | 의미 |
|---|---|
| SISD | Single Instruction, Single Data |
| SIMD | Single Instruction, Multiple Data |
| MIMD | Multiple Instruction, Multiple Data |
| ILP | Instruction Level Parallelism |
| TLP | Thread Level Parallelism |
| DLP | Data Level Parallelism |

## 정리

이 강의의 핵심은 병렬화를 하나의 기법으로 보지 않고, 하드웨어 계층별 병렬성으로 보는 것이다. Superscalar는 자동, multi-core는 thread 분할, SIMD는 데이터 벡터화가 중심이며, 이후 CUDA warp와 GPU thread block도 이 관점의 확장으로 이해할 수 있다.
