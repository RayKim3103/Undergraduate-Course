---
title: "10 Prefix Sum - GPU Scan 알고리즘"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - prefix-sum
  - scan
  - cuda
---

# 10 Prefix Sum - GPU Scan 알고리즘

이전: [[09 CUDA Others - TensorCore와 CUDA Libraries]]  
다음: [[10 Prefix Sum NVIDIA Supplement - Work-Efficient Scan]]

## 핵심 요약

Prefix sum 또는 scan은 배열의 각 위치에 그 앞 원소들의 누적값을 저장하는 병렬 primitive다. Quicksort partition, histogram, polynomial evaluation, radix sort, regular expression search 등 다양한 알고리즘의 building block이다.

## Reduce와 Prefix Sum

| 연산 | 결과 |
|---|---|
| Reduce | 전체 배열을 하나의 값으로 결합 |
| Prefix sum | 각 index별 partial accumulation을 모두 계산 |

예를 들어 `[1,2,3,4]`의 inclusive scan은 `[1,3,6,10]`이고, exclusive scan은 `[0,1,3,6]`이다.

## Naive Parallel Prefix Sum

각 단계마다 거리 `1, 2, 4, ...`의 neighbor를 더한다. Hillis-Steele 또는 Kogge-Stone 형태다.

장점:

- `O(log N)` 단계
- 병렬성이 매우 큼

단점:

- 전체 work가 `O(N log N)`으로 work-efficient하지 않다.
- 각 단계마다 barrier가 필요하다.

## Kogge-Stone Algorithm

Kogge-Stone은 가까운 neighbor부터 점점 먼 neighbor까지 더한다.

```text
step 1: i >= 1이면 a[i] += a[i-1]
step 2: i >= 2이면 a[i] += a[i-2]
step 4: i >= 4이면 a[i] += a[i-4]
...
```

특징:

- 빠른 depth: `O(log N)`
- 많은 연산량
- 많은 thread가 계속 active

## Brent-Kung Algorithm

Brent-Kung은 balanced binary tree pattern을 사용한다.

두 phase:

1. Up-sweep 또는 reduce phase: tree 위로 partial sum 생성
2. Down-sweep phase: prefix 값을 아래로 전파

특징:

- 시간: 대략 `2 log N`
- work: `O(N)`
- Kogge-Stone보다 단계는 많지만 계산량이 적다.

## Kogge-Stone vs Brent-Kung

| 항목 | Kogge-Stone | Brent-Kung |
|---|---|---|
| Time complexity | `log N` | `2 log N` |
| Work | `O(N log N)` | `O(N)` |
| 장점 | 짧은 latency | work-efficient |
| 단점 | 연산량 많음 | 단계 수 많음 |

## GPU 구현 포인트

- Shared memory에 block 내부 데이터를 올린다.
- 단계마다 `__syncthreads()`를 사용한다.
- Exclusive scan이면 입력을 한 칸 shift하고 첫 값을 identity로 둔다.
- 큰 배열은 block별 scan 후 block sum을 다시 scan하고 각 block에 더한다.
- Shared memory tree access에서 bank conflict가 생길 수 있어 padding이 필요하다.

## Lesson

강의의 마지막 메시지는 lock 기반 누적보다 병렬 알고리즘 구조를 찾는 것이 더 좋다는 것이다. Prefix sum은 단순해 보이지만 많은 병렬 알고리즘의 기반이며, scan을 잘 구현하면 여러 상위 문제를 효율적으로 풀 수 있다.

## 정리

Prefix sum은 reduce보다 더 많은 정보를 유지하는 누적 연산이다. GPU에서는 Kogge-Stone과 Brent-Kung의 depth/work tradeoff, shared memory bank conflict, 큰 배열 decomposition이 핵심이다.
