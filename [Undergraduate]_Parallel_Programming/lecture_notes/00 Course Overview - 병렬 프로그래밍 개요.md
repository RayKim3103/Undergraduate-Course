---
title: "00 Course Overview - 병렬 프로그래밍 개요"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - course-overview
  - multicore
  - gpu
---

# 00 Course Overview - 병렬 프로그래밍 개요

이전: 없음  
다음: [[01 Basic Parallel Architectures - 기본 병렬 아키텍처]]

## 핵심 요약

이 강의는 멀티코어 CPU와 GPU를 대상으로 병렬 프로그래밍 모델, 병렬 아키텍처, 성능 최적화 사례를 다룬다. 단순히 문법을 배우는 과목이 아니라, 하드웨어 구조를 이해하고 그 구조에 맞게 프로그램을 나누어 실행하도록 만드는 것이 중심이다.

## 수업 정보

| 항목 | 내용 |
|---|---|
| 과목 | Multicore and GPU Programming |
| 학기 | 2025 Spring |
| 담당 | Yongjun Park |
| 선수 지식 | C++ OOP, 자료구조, Linux 기본 |
| 권장 지식 | 컴퓨터 구조, 운영체제, 알고리즘 |
| 평가 | 중간고사, 과제, 기타 평가 요소 |

## 왜 병렬 프로그래밍인가

과거에는 단일 thread 프로그램도 시간이 지나면 자동으로 빨라졌다. Moore's Law와 Dennard scaling 덕분에 transistor 수와 clock frequency가 증가했기 때문이다. 하지만 2000년대 초반 이후 power wall로 인해 frequency scaling이 멈췄고, 더 이상 같은 코드를 기다리기만 해도 빨라지는 시대가 아니다.

## Moore's Law와 Dennard Scaling

| 개념 | 의미 | 병렬화와의 관계 |
|---|---|---|
| Moore's Law | 집적회로의 transistor 수가 약 2년마다 2배 증가 | 더 많은 자원을 칩에 넣을 수 있음 |
| Dennard Scaling | 공정이 작아질수록 전압과 전력이 같이 줄어 frequency 증가 가능 | 2000년대 이후 한계 도달 |
| Power wall | 동적/정적 전력 증가로 clock을 더 올리기 어려움 | multi-core, GPU 같은 병렬 구조로 전환 |

동적 전력은 대략 `P ∝ C * V^2 * f`로 표현된다. voltage와 frequency를 계속 올릴 수 없게 되면서, 남는 transistor는 단일 core를 더 빠르게 하기보다 여러 core와 병렬 실행 자원으로 배치되었다.

## 병렬 아키텍처의 등장

대표 예시:

- Intel Pentium Extreme Edition 840: 초기 dual-core processor
- Intel Coffee Lake: multi-core CPU와 integrated GPU
- Intel Xeon Phi: many-core architecture
- NVIDIA Tesla V100: 대규모 병렬 GPU

중요한 변화는 이제 하드웨어가 병렬적이어도 프로그램이 자동으로 빨라지지 않는다는 점이다. 병렬성을 드러내고 작업을 나누는 책임이 프로그래머에게 넘어왔다.

## Sum Reduction 예시

배열 합계를 구하는 단순 코드는 병렬화의 좋은 예시다.

```cpp
int sum = 0;
for (int i = 0; i < N; i++) {
    sum += numbers[i];
}
```

병렬화하면 각 processor가 partial sum을 만들고, 마지막에 partial sum을 합친다. 단순 master가 모든 partial sum을 모으는 방식은 병목이 생긴다. 더 좋은 방식은 hierarchical reduce로, tree 구조처럼 단계적으로 합쳐 network contention과 serial bottleneck을 줄인다.

## 이 과목에서 다루는 것

- Parallel programming model
- Multi-core CPU programming
- GPU programming
- Architecture-aware optimization
- Matrix multiplication, reduction, scan, convolution 같은 핵심 kernel
- CUDA, Triton, library, compiler 관점의 성능 최적화

## 정리

이 강의의 출발점은 power wall 이후 컴퓨터 성능 향상이 “자동 clock 증가”에서 “명시적 병렬성 활용”으로 바뀌었다는 사실이다. 이후 모든 주제는 작업을 어떻게 나눌지, 데이터를 어떻게 배치할지, 하드웨어 병목을 어떻게 피할지로 이어진다.
