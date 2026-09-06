---
title: "03 Matrix Multiplication - CPU Cache와 병렬 행렬곱"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - matrix-multiplication
  - cache
  - false-sharing
---

# 03 Matrix Multiplication - CPU Cache와 병렬 행렬곱

이전: [[02 Thread Programming - C++ Thread와 동기화]]  
다음: [[04 Intro to CUDA - CUDA 프로그래밍 모델]]

## 핵심 요약

이 강의는 CPU에서 행렬곱을 병렬화할 때 thread 분할만으로는 충분하지 않다는 점을 다룬다. column-wise 분할은 false sharing과 cache 비효율을 만들 수 있고, row-wise 분할, transpose, padding, blocked matrix multiplication으로 spatial locality와 temporal locality를 개선해야 한다.

## 기본 행렬곱

정방행렬 기준:

```cpp
for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < N; k++)
      C[i][j] += A[i][k] * B[k][j];
```

연산량은 `O(N^3)`이다. 따라서 cache miss와 memory bandwidth 문제가 전체 성능에 큰 영향을 준다.

## 병렬화 전략

| 전략 | 방식 | 장점 | 문제 |
|---|---|---|---|
| Column-wise block striping | C의 column을 thread에 분배 | 중복 계산 없음 | C write에서 false sharing, B column 접근 비효율 |
| Row-wise block striping | C의 row를 thread에 분배 | lock 불필요, row-major 접근에 유리 | B는 여전히 column 방향 접근 |

일반적인 row-major 배열에서는 row-wise로 C를 나누는 편이 locality와 false sharing 측면에서 유리하다.

## False Sharing

False sharing은 서로 다른 thread가 논리적으로 다른 data를 쓰지만, 같은 cache line에 위치해 cache coherence traffic이 증가하는 현상이다. Column-wise로 C를 나누면 여러 thread가 같은 row 근처의 다른 column을 쓰면서 같은 cache line을 건드릴 가능성이 높다.

## Cache 관점의 문제

L1 cache는 보통 32KB~64KB 정도라 큰 행렬 전체를 담기 어렵다. `A[i][k]`는 row-wise 접근이라 cache line을 잘 활용하지만, `B[k][j]`는 column-wise 접근이라 매 접근이 멀리 떨어진 address가 된다.

결과:

- A는 spatial locality가 좋다.
- B는 cache line을 읽어도 실제로 필요한 값은 하나뿐인 경우가 많다.
- N이 2의 거듭제곱 배수이면 set associativity 때문에 특정 cache set에 충돌이 집중될 수 있다.

## Padding

N이 cache set과 나쁘게 맞물릴 때 padding을 추가해 stride를 바꾸면 set collision peak를 줄일 수 있다. 단, padding은 capacity와 bandwidth를 조금 더 쓰는 비용이 있다.

## Transpose

B를 미리 transpose하면 원래 column 접근이 row 접근으로 바뀐다.

```cpp
C[i][j] += A[i][k] * B_T[j][k];
```

Transpose 자체는 `O(N^2)`의 read/write 비용이 들지만, 행렬곱의 `O(N^3)` 비용에 비하면 큰 N에서는 충분히 감수할 수 있다.

## Blocked Matrix Multiplication

Blocked matrix multiplication은 행렬을 cache에 들어갈 크기의 tile로 나누어 계산한다.

장점:

- A/B tile이 cache에 머무는 동안 여러 번 재사용된다.
- memory traffic이 줄어든다.
- single-thread 최적화와 multi-thread 최적화 모두에 중요하다.

Block size `b`를 키우면 각 tile element 재사용이 늘지만, 너무 크면 L1 cache를 넘어서 eviction이 증가한다.

## 성능 분석 관점

| 대상 | Naive 접근 | 개선 방향 |
|---|---|---|
| A | row-wise라 상대적으로 유리 | block으로 temporal locality 추가 |
| B | column-wise라 miss 많음 | transpose 또는 blocking |
| C | read/write 필요 | thread별 row 분할로 false sharing 완화 |

## 정리

CPU 행렬곱 최적화는 “thread를 몇 개 쓰는가”보다 “각 thread가 cache line을 어떻게 읽고 쓰는가”가 중요하다. row-wise 분할, B transpose, padding, blocking은 모두 memory hierarchy를 의식한 최적화이며, 이후 CUDA shared memory tiling의 CPU 버전 배경이 된다.
