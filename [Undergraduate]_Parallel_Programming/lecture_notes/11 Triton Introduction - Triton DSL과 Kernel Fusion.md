---
title: "11 Triton Introduction - Triton DSL과 Kernel Fusion"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - triton
  - gpu
  - kernel-fusion
---

# 11 Triton Introduction - Triton DSL과 Kernel Fusion

이전: [[10 Prefix Sum NVIDIA Supplement - Work-Efficient Scan]]  
다음: [[12 CUDA Stream - Pinned Memory와 비동기 파이프라인]]

## 핵심 요약

이 강의는 Triton DSL을 소개한다. PyTorch의 operator 조합은 각 operator마다 memory load/store와 kernel launch가 발생해 memory-bound workload에서 비효율이 생길 수 있다. Triton은 Python 기반 DSL로 block 단위 GPU kernel을 직접 작성하여 operation fusion과 tunable kernel configuration을 가능하게 한다.

## 왜 Triton인가

Softmax 예시에서 naive PyTorch 구현은 여러 operator로 나뉘어 input을 반복해서 읽고 중간 결과를 저장한다. Triton은 필요한 연산을 하나의 custom kernel로 fusion하여 global memory traffic을 줄일 수 있다.

이론적 비교:

- Naive PyTorch: 여러 operator로 인해 `5N^2 + 2N` 수준의 load, 중간 store 발생
- Triton fusion: 필요한 데이터를 한 번 읽고 최종 결과를 저장하는 형태로 감소
- Memory-bound kernel에서는 memory traffic 감소가 곧 큰 speedup으로 연결된다.

## CUDA Software Stack 속 Triton

Application은 보통 CUDA library(cuBLAS, cuDNN), CUDA runtime API, device API 등을 통해 GPU를 사용한다. Triton은 CUDA보다 높은 수준에서 GPU kernel을 작성하지만, PyTorch보다 낮은 수준에서 block, index, mask, tile size를 직접 설계할 수 있는 위치에 있다.

## Triton Kernel 구성 요소

| 요소 | 의미 |
|---|---|
| `@triton.jit` | Python 함수를 GPU kernel로 JIT compile |
| `tl.program_id(axis)` | 현재 program instance의 block id |
| `tl.arange` | block 내부 vector index 생성 |
| mask | out-of-bounds load/store 방지 |
| strides | non-contiguous tensor layout 접근 |
| constexpr | compile-time tunable parameter |

Triton의 program instance는 CUDA thread block과 비슷한 추상화로 볼 수 있다. 각 program이 tile 단위 data를 처리한다.

## Index와 Mask

Triton에서는 직접 pointer offset을 만들어 load/store한다.

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offs < n
x = tl.load(x_ptr + offs, mask=mask)
tl.store(y_ptr + offs, out, mask=mask)
```

Mask는 block 크기가 입력 크기를 딱 나누지 않을 때 잘못된 memory 접근을 막는다.

## Autotune과 Benchmark

| 기능 | 역할 |
|---|---|
| `@triton.autotune` | 여러 block size, num warps 등 config를 자동 실험 |
| key | 어떤 입력 shape 변화에서 다시 튜닝할지 결정 |
| `@triton.benchmark` | kernel 성능 측정 |

Triton은 kernel parameter를 실험하며 shape별 최적 configuration을 찾기 좋다.

## PyTorch Integration

Triton은 PyTorch와 함께 custom operator처럼 사용될 수 있다. Liger Kernel, PyTorch tutorial 등에서 transformer/LLM 연산을 Triton으로 최적화하는 예시를 제공한다.

## 정리

Triton의 강점은 CUDA보다 생산성이 높고 PyTorch보다 kernel-level 제어가 가능하다는 점이다. 특히 memory-bound operator에서 fusion으로 global memory traffic을 줄일 때 효과적이다. 다만 CUDA처럼 thread, shared memory를 세밀하게 직접 제어하는 모델은 아니므로 최적화 관점의 추상화 차이를 이해해야 한다.
