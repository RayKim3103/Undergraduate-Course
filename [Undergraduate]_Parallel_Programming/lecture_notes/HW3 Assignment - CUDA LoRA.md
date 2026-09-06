---
title: "HW3 Assignment - CUDA LoRA"
course: "Parallel Programming"
type: "assignment"
tags:
  - parallel-programming
  - homework
  - cuda
  - lora
---

# HW3 Assignment - CUDA LoRA

이전: [[HW2 Report - Parallel GEMM과 Freivalds 최적화]]  
다음: [[HW3 Report - CUDA LoRA Tiled MatMul 최적화]]

## 핵심 요약

HW3는 LoRA 연산을 CUDA C++로 구현하는 과제다. LoRA 자체의 세부 이론보다, LoRA가 matrix multiplication에 크게 의존한다는 점에 초점을 둔다. 제출 대상은 `lora.h`이며, CUDA matmul과 scaling/addition을 정확하고 빠르게 구현해야 한다.

## LoRA 연산

NumPy 형태:

```python
def lora(x, W, A, B, alpha):
    out_linear = x @ W.T
    out_lora = x @ A.T @ B.T * scale
    return out_linear + out_lora
```

Shape:

| Tensor | Shape | 의미 |
|---|---|---|
| `x` | `[B, in_dim]` | input |
| `W` | `[out_dim, in_dim]` | original weight |
| `A` | `[r, in_dim]` | down projection |
| `B` | `[out_dim, r]` | up projection |
| output | `[B, out_dim]` | LoRA 적용 결과 |

## 구현 요구사항

- `lora.h`에 CUDA C++ 구현
- CUDA 기반 matrix multiplication
- LoRA scaling과 addition
- 정확한 output 형식 유지
- OpenMP 사용 금지

## 채점 기준

성능 순위 기반 점수:

| 점수 | 조건 |
|---:|---|
| 10 | top 10% |
| 9 | top 30% |
| 8 | top 50% |
| 7 | correctness 만족 |

## 제출 규칙

- HW3 directory를 지정 경로에 배치
- `make run`이 정상 작동해야 함
- `std::cout` 출력 변경 금지
- deadline 이후 timestamp 변경 주의
- file permission 유지

## 관련 강의 연결

- [[04 Intro to CUDA - CUDA 프로그래밍 모델]]
- [[05 CUDA Matrix Multiplication - Shared Memory Tiling]]
- [[06 CUDA Transpose and Bank Conflict - Shared Memory 심화]]

## 정리

HW3는 CUDA matmul을 실제 deep learning layer 계산에 적용하는 과제다. 핵심은 LoRA 식을 여러 GEMM과 element-wise operation으로 분해하고, shared memory, coalescing, tiling, occupancy를 고려해 구현하는 것이다.
