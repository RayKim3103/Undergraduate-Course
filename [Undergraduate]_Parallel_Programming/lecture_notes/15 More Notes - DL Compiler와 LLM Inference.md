---
title: "15 More Notes - DL Compiler와 LLM Inference"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - dl-compiler
  - llm
  - inference
---

# 15 More Notes - DL Compiler와 LLM Inference

이전: [[14 Multi GPU - 단일 노드와 MPI]]  
다음: [[HW2 Assignment - Matrix Verification Challenge]]

## 핵심 요약

마지막 강의는 deep learning compiler와 LLM inference 최적화를 소개한다. TensorFlow XLA, TensorRT, TVM, Glow, Triton 같은 compiler/runtime은 computation graph를 IR로 보고 graph rewrite, operator fusion, code generation을 수행한다. LLM inference에서는 prefill은 GEMM, decode는 GEMV가 중요하며, FlashAttention과 vLLM 같은 기법이 memory와 scheduling 병목을 줄인다.

## Deep Learning Compilers

대표 시스템:

- TensorFlow XLA
- NVIDIA TensorRT
- TVM
- Glow
- Triton

DL framework는 보통 network를 operator 단위로 실행한다. 각 operator는 input을 읽고 output을 만들며, operator 사이에 intermediate tensor가 생긴다. 이 방식은 사용하기 쉽지만 interpreted execution, kernel launch overhead, 불필요한 memory traffic이 생길 수 있다.

## Compiler 관점

Compiler는 framework의 computation을 intermediate representation(IR)로 보고 hardware별 code를 생성한다.

흐름:

```text
Framework graph
-> IR
-> graph optimization
-> operator fusion / layout transform
-> target-specific code generation
-> optimized binary/runtime
```

## XLA

XLA는 Accelerated Linear Algebra로, TensorFlow computation을 HLO(High Level Operations) IR로 표현한다. Map, broadcast, reduce, convolution 같은 tensor primitive를 최적화하고 target에 맞는 실행 binary를 생성한다.

## TensorRT

NVIDIA TensorRT는 DNN inference 최적화 플랫폼이다.

주요 기능:

- Weight quantization
- Kernel fusion
- Vertical/horizontal fusion
- NVIDIA GPU에 최적화된 inference execution

## TVM

TVM은 다양한 hardware target을 위한 machine learning compiler와 runtime이다. High-level graph rewriting부터 schedule/code generation까지 포함하며 GPU, CPU, mobile, FPGA 같은 target을 지원한다.

## LLM Inference

Decode-only LLM inference는 크게 두 단계로 나뉜다.

| 단계 | 설명 | 주요 연산 |
|---|---|---|
| Prefill | prompt 전체를 처리해 첫 token 상태 생성 | GEMM 중심 |
| Decode | token을 하나씩 autoregressive하게 생성 | GEMV 중심 |

Context length가 길어질수록 prefill latency에서 GEMM 비중이 커지고, decode에서는 매 token마다 batch/sequence 구조 때문에 GEMV가 중요해진다.

## FlashAttention

Standard attention은 memory traffic이 크다. FlashAttention은 tiling과 fusion으로 attention 중간 matrix를 global memory에 크게 저장하지 않고 block 단위로 처리하여 memory traffic을 줄인다.

발전 흐름:

- FlashAttention v1: tiling/fusion으로 memory efficient attention
- v2/v3: work partitioning과 hardware utilization 개선

## vLLM과 PagedAttention

vLLM은 PagedAttention으로 KV cache를 memory page처럼 관리한다. LLM serving에서 요청별 sequence 길이가 다르고 KV cache가 커지는 문제를 줄여 throughput과 memory utilization을 개선한다.

## 정리

마지막 강의의 메시지는 병렬 프로그래밍의 원리가 CPU/GPU kernel을 넘어 DL compiler와 LLM serving에도 그대로 적용된다는 것이다. 연산을 fusion하고, memory traffic을 줄이고, GEMM/GEMV/attention 같은 핵심 primitive를 target hardware에 맞게 최적화하는 것이 현대 AI system 성능의 중심이다.
