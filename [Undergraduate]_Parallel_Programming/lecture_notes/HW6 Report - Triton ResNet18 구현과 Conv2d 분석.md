---
title: "HW6 Report - Triton ResNet18 구현과 Conv2d 분석"
course: "Parallel Programming"
type: "report"
tags:
  - parallel-programming
  - report
  - triton
  - resnet18
  - conv2d
---

# HW6 Report - Triton ResNet18 구현과 Conv2d 분석

이전: [[HW6 Assignment - Triton ResNet]]  
다음: 없음

## 핵심 요약

이 보고서는 Triton으로 ResNet18을 구현하면서 각 layer의 병렬화 방식과 Conv2d 성능 한계를 분석한다. ResNet18에서 Conv2d가 대부분의 계산량을 차지하므로, direct convolution보다 im2col + GEMM 방식으로 구현했다. 하지만 PyTorch가 사용하는 cuDNN/cuBLAS/CUTLASS 수준의 low-level 최적화에는 미치지 못해 Conv2d 성능이 torch보다 낮았다.

## 목표

- Triton으로 ResNet18 구조 가속
- Triton program instance, block size, mask, index 계산 이해
- Conv2d, BatchNorm, Linear, ReLU 등의 layer 구현
- PyTorch 대비 latency와 병목 분석

## ResNet18 연산 구조

ResNet18은 convolution, batch normalization, ReLU, max pooling, residual connection, linear layer로 구성된다.

보고서에서 중요하게 본 부분:

- 초기 7x7 Conv
- 여러 개의 3x3 Conv block
- channel 증가 시 1x1 Conv
- 최종 Linear

연산량 관점에서는 Conv2d가 가장 큰 비중을 차지하므로 최적화 우선순위가 높다.

## Triton Kernel 사고방식

Triton에서는 CUDA thread 하나가 아니라 program instance가 block/tile 단위 작업을 담당한다.

주요 개념:

| Triton 요소 | 의미 |
|---|---|
| `@triton.jit` | GPU kernel compile |
| `tl.program_id(0)` | 현재 program block id |
| `tl.arange` | vectorized offset 생성 |
| `tl.load` / `tl.store` | pointer + offset 기반 memory 접근 |
| mask | out-of-bounds 방지 |
| `tl.constexpr` | compile-time block parameter |

CUDA와 달리 shared memory를 명시적으로 선언하거나 thread 단위로 제어하는 방식은 제한적이다. Triton compiler가 접근 패턴에 따라 register/shared memory 사용을 결정한다.

## BatchNorm2d

BatchNorm은 channel별 running mean, running variance, scale, bias를 사용한다. 따라서 index 계산에서 channel을 정확히 복원하는 것이 중요하다.

일반 형태:

```text
y = (x - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]
```

## Conv2d 구현

처음에는 direct convolution을 시도했지만, output element 하나를 매우 작은 kernel/program이 계산하는 식으로 구성되어 지나치게 느렸다. 이후 im2col + GEMM 방식으로 바꾸었다.

구성:

1. im2col 변환 kernel
2. weight shape 변환 kernel
3. GEMM kernel
4. 결과 reshape kernel

## im2col Kernel

보고서의 im2col은 transpose된 형태의 결과를 만들도록 작성되었다. row-wise `tl.load`와 `tl.store`를 통해 연속적인 memory 접근을 얻으려는 목적이다.

하지만 reshape 과정에서 일부 load 또는 store가 coalesced하지 않아 병목이 남았다.

## Weight Transpose

Weight transpose는 store는 coalesced하게 만들었지만, load에서 output channel 기준 접근이 되어 coalescing이 깨질 수 있다고 분석했다. Load coalescing까지 개선하면 추가 성능 향상이 가능하다.

## GEMM Kernel

GEMM은 이전 CUDA 실습의 tiling/blocked matrix multiplication 아이디어를 Triton으로 옮겼다.

주요 고려:

- block size
- program 수
- mask 처리
- accumulator dtype
- memory access coalescing

## Linear와 ReLU

Linear는 input이 float16으로 들어오는 점을 고려해 compute dtype과 output dtype을 정했다. ReLU는 skeleton 제공 방식으로 block size 1024를 사용하며 `tl.where`로 element-wise 처리한다. Offsets가 연속적이라 load/store coalescing이 잘 지켜진다.

## Latency Evaluation

초기 direct convolution 구현은 너무 느려 실행 완료를 기다리기 어려웠다. im2col + GEMM으로 바꾼 뒤 Triton 결과와 PyTorch 결과를 비교했다. PyTorch가 더 빠른 이유는 NVIDIA library 기반 최적화 kernel을 사용하기 때문이다.

## Conv2d가 Torch보다 느린 이유

분석된 이유:

- Triton은 CUDA보다 thread 단위/explicit shared memory 제어가 제한적이다.
- PyTorch는 cuDNN, cuBLAS, CUTLASS 같은 vendor-optimized kernel을 활용한다.
- im2col 변환과 reshape 과정에서 추가 kernel launch와 memory traffic이 발생한다.
- 일부 reshape/transpose에서 coalescing이 깨진다.
- TensorCore, mixed precision, specialized convolution algorithm 활용이 부족하다.

## 개선 방향

- 32bit 대신 16bit 사용으로 bandwidth와 TensorCore 활용 가능성 개선
- im2col/reshape kernel fusion
- weight transpose load coalescing 개선
- Conv2d direct 또는 implicit GEMM 방식 재설계
- Triton autotune으로 block size, num warps, stages 탐색

## 정리

HW6 보고서의 핵심은 Triton이 생산성과 kernel fusion 측면에서 강력하지만, Conv2d처럼 vendor library가 극도로 최적화한 연산에서는 직접 구현이 쉽게 이기기 어렵다는 점이다. 성능을 끌어올리려면 im2col memory traffic, coalescing, dtype, TensorCore 활용, kernel fusion을 함께 개선해야 한다.
