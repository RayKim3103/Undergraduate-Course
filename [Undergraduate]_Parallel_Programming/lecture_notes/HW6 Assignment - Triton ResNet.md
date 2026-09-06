---
title: "HW6 Assignment - Triton ResNet"
course: "Parallel Programming"
type: "assignment"
tags:
  - parallel-programming
  - homework
  - triton
  - resnet
---

# HW6 Assignment - Triton ResNet

이전: [[HW5 Reference - Optimizing Parallel Reduction in CUDA]]  
다음: [[HW6 Report - Triton ResNet18 구현과 Conv2d 분석]]

## 핵심 요약

HW6는 final assignment로 Triton을 이용해 ResNet18을 직접 구현하는 과제다. PyTorch와 CUDA library가 이미 고성능 kernel을 제공하지만, 특수 상황에서 custom GPU kernel을 작성하는 능력을 기르기 위해 Triton kernel을 구현한다.

## 과제 목표

- ResNet18 구조를 Triton kernel로 구현
- `TritonMGP/kernel` directory 안 kernel 구현
- Grading 시 kernel directory만 사용
- PyTorch/library 대비 custom kernel 성능과 한계 분석

## 왜 Triton인가

Custom kernel 작성은 비용이 있지만, operator fusion이나 특정 shape 최적화가 필요할 때 유리할 수 있다. Triton은 CUDA보다 높은 수준에서 GPU tile/program 단위 kernel을 작성할 수 있게 한다.

## 보고서 요구사항

보고서는 매우 중요하며 10점 배점이다. 특히 다음 질문에 대한 분석이 필요하다.

- 왜 Conv2d가 torch 대비 효율적인 성능을 보이지 않는가?
- 직접 구현한 Triton kernel의 병목은 무엇인가?
- PyTorch가 사용하는 cuBLAS, cuDNN, CUTLASS 같은 library kernel과 어떤 차이가 있는가?

## 채점 및 규칙

- Final assignment라 late submission 없음
- grading server 5회 실행 중 maximum 기준
- output 변경 금지
- 충분한 file permission 유지
- 보고서에 GPT 사용 시 큰 penalty 명시

## 관련 강의 연결

- [[11 Triton Introduction - Triton DSL과 Kernel Fusion]]
- [[15 More Notes - DL Compiler와 LLM Inference]]
- [[07 CUDA DNN - Convolution과 im2col]]

## 정리

HW6는 강의의 CUDA 최적화 지식을 Triton과 DNN 전체 모델 구현으로 확장하는 과제다. 핵심은 ResNet18의 각 layer를 Triton kernel로 구현하면서, high-level DSL의 생산성과 low-level CUDA/library 대비 성능 한계를 함께 분석하는 것이다.
