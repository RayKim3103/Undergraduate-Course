---
title: "HW4 Report - Direct Convolution과 im2col GEMM"
course: "Parallel Programming"
type: "report"
tags:
  - parallel-programming
  - report
  - cuda
  - convolution
  - im2col
---

# HW4 Report - Direct Convolution과 im2col GEMM

이전: [[HW4 Assignment - CUDA Conv2d]]  
다음: [[HW5 Assignment - CUDA Sum Reduction]]

## 핵심 요약

이 보고서는 CUDA로 direct convolution과 im2col+GEMM 방식을 구현하고 최적화한 내용을 정리한다. Shared memory, tile overlap, coalescing, bank conflict, occupancy, `__syncthreads()` overhead가 주요 분석 대상이다.

## 2D Convolution

입력 matrix `I`와 filter `K`를 사용하여 output `O`를 만든다. Output의 각 element는 filter window 내부의 multiply-accumulate로 계산된다.

일반 CNN convolution에서는 batch, output channel, input channel, height, width, kernel height, kernel width가 함께 관여한다.

## im2col

im2col은 input image의 sliding window patch를 column 형태로 재배치하여 convolution을 GEMM으로 바꾸는 기법이다.

장점:

- GEMM kernel 최적화를 활용 가능
- GPU가 잘 처리하는 matrix multiplication으로 문제 변환

단점:

- 겹치는 patch가 중복 저장되어 memory overhead 증가
- im2col 변환 kernel 자체의 비용 존재

## Direct Convolution 최적화 포인트

### Shared Memory

인접 output pixel은 input window를 많이 공유한다. 따라서 input tile과 filter를 shared memory에 올리면 global memory 접근을 줄일 수 있다.

### Tile Overlap

Filter size가 `K x K`이면 output tile을 계산하기 위해 input tile 주변의 halo 영역도 필요하다. Load tile은 compute tile보다 커질 수 있다.

### Bank Conflict

Input tile과 filter를 shared memory에 저장하면 warp threads가 같은 bank를 동시에 접근할 수 있다. Padding을 고려하면 conflict를 줄일 수 있다.

### Synchronization

Shared memory를 쓰면 tile load 완료 후 `__syncthreads()`가 필요하다. 이 barrier는 correctness에는 필수지만, 모든 thread가 가장 늦은 thread를 기다리므로 성능 overhead가 된다.

## GEMM Kernel

im2col 이후 matmul은 강의안의 shared memory tiled matrix multiplication 구조를 사용했다.

특징:

- shared memory 사용
- load/store coalescing 고려
- temporal locality 향상
- tile size와 occupancy 균형 필요

## Implementation 관점

보고서는 RTX 3090의 hardware specification을 기준으로 thread block size 128, 256, 512 등을 검토했다. Occupancy 100%가 가능하더라도 실제 성능은 memory access와 synchronization overhead의 영향을 함께 받는다.

## 성능 저하 요인

- Shared memory tile load에 참여하지만 실제 compute에 사용되지 않는 thread 발생
- Boundary와 filter size 때문에 idle thread 증가
- `__syncthreads()` overhead
- Filter load에서 같은 값 접근이 많아 bank conflict 가능성
- im2col 변환으로 추가 memory traffic 발생

## Evaluation

보고서는 shared memory 적용 여부, tile size, direct convolution과 im2col+GEMM의 실행 시간을 비교했다. 성능 평가는 단순 correctness가 아니라 어떤 optimization이 실제 benchmark에서 이득을 주는지 확인하는 방식으로 진행되었다.

## 정리

HW4 보고서의 핵심은 convolution이 matrix multiplication보다 data reuse와 boundary 처리가 더 복잡하다는 점이다. Direct convolution은 중복 read를 shared memory로 줄여야 하고, im2col+GEMM은 GEMM의 장점을 얻는 대신 변환 비용과 memory overhead를 감수해야 한다.
