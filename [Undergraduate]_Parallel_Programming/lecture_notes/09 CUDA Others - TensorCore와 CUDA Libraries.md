---
title: "09 CUDA Others - TensorCore와 CUDA Libraries"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - tensorcore
  - libraries
---

# 09 CUDA Others - TensorCore와 CUDA Libraries

이전: [[08 CUDA Reduction - Parallel Reduction 최적화]]  
다음: [[10 Prefix Sum - GPU Scan 알고리즘]]

## 핵심 요약

이 강의는 TensorCore, CUDA library, 그리고 최신 GPU에서 기존 설명이 단순화였던 부분들을 보완한다. 핵심은 모든 kernel을 직접 작성할 필요는 없으며, GEMM/DNN/parallel primitive는 cuBLAS, cuDNN, Thrust 같은 library가 매우 강력하다는 점이다.

## TensorCore

Matrix multiplication은 작은 matrix multiply-accumulate의 반복으로 볼 수 있다. TensorCore는 이런 작은 matrix 연산을 hardware에서 매우 빠르게 처리하는 특수 unit이다.

특징:

- 작은 tile 단위 matrix multiply-accumulate 수행
- 16bit multiplication과 32bit accumulation 지원
- GPU 세대별 지원 precision이 다름
- deep learning GEMM/conv 성능의 핵심 hardware

## CUDA Libraries

| Library | 역할 |
|---|---|
| cuBLAS | BLAS, GEMM/GEMV 등 dense linear algebra |
| cuDNN | convolution, pooling, normalization 등 DNN primitive |
| Thrust | C++ STL 스타일 parallel algorithms |

직접 kernel을 쓰는 것은 학습과 특수 최적화에는 중요하지만, 실제 production에서는 library kernel이 더 빠르고 안정적인 경우가 많다.

## Independent Thread Scheduling

초기 CUDA 설명에서는 warp가 lockstep으로 움직인다고 단순화한다. 하지만 Volta 이후 GPU는 thread마다 program counter를 갖고 더 독립적으로 scheduling될 수 있다.

결과:

- 더 유연한 scheduling 가능
- 일부 warp-synchronous trick이 더 이상 안전하지 않을 수 있음
- warp 내부 통신이나 reduction에서는 `__syncwarp()` 필요 가능

## Large L2 Cache

최신 GPU는 더 큰 L2 cache를 가진다. 예를 들어 A100은 40MB L2 cache를 갖지만 SM 수로 나누면 SM당 절대적으로 무한히 큰 것은 아니다. 그래도 높은 bandwidth와 cache 효과는 성능 분석에서 중요하다.

## 정리

이 강의의 핵심은 “직접 kernel 작성”과 “library 사용”의 균형이다. CUDA를 잘하려면 low-level 최적화를 이해해야 하지만, 동시에 TensorCore와 cuBLAS/cuDNN/Thrust 같은 검증된 고성능 primitive를 언제 사용할지 판단할 수 있어야 한다.
