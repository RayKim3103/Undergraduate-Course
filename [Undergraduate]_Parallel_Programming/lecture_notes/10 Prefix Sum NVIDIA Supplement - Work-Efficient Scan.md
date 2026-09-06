---
title: "10 Prefix Sum NVIDIA Supplement - Work-Efficient Scan"
course: "Parallel Programming"
type: "supplement"
tags:
  - parallel-programming
  - prefix-sum
  - nvidia
  - scan
---

# 10 Prefix Sum NVIDIA Supplement - Work-Efficient Scan

이전: [[10 Prefix Sum - GPU Scan 알고리즘]]  
다음: [[11 Triton Introduction - Triton DSL과 Kernel Fusion]]

## 핵심 요약

이 보충자료는 NVIDIA의 Parallel Prefix Sum 문서로, CUDA에서 scan을 효율적으로 구현하는 과정을 설명한다. Naive scan의 `O(N log N)` work를 피하기 위해 Blelloch scan의 up-sweep/down-sweep tree 구조를 사용하고, shared memory bank conflict를 padding으로 줄이며, 큰 배열은 block 단위 scan과 block sum scan으로 확장한다.

## Prefix Sum 정의

Associative operator `⊕`와 identity `I`가 있을 때:

- Inclusive scan: `[a0, a0⊕a1, ..., a0⊕...⊕an]`
- Exclusive scan: `[I, a0, a0⊕a1, ..., a0⊕...⊕a(n-1)]`

Scan은 sorting, stream compaction, data structure construction 등에서 핵심 primitive로 사용된다.

## Naive Scan의 한계

Hillis-Steele 형태의 naive scan은 `log N` 단계만 필요하지만, 각 단계마다 거의 N개 연산을 수행하므로 work가 `O(N log N)`이다. Sequential scan이 `O(N)`임을 생각하면 work-efficient하지 않다.

## Work-Efficient Scan

Blelloch scan은 balanced tree pattern을 사용한다.

### Up-Sweep

Tree의 leaf에서 root 방향으로 partial sum을 만든다. 마지막 root에는 전체 합이 저장된다.

### Down-Sweep

Root를 identity로 바꾼 뒤, tree를 내려오면서 prefix 값을 전파한다. 이 과정을 통해 exclusive scan 결과를 얻는다.

## CUDA Shared Memory 구현

Block 내부에서 각 thread가 보통 두 원소를 shared memory에 load한다. 이후 up-sweep과 down-sweep loop를 수행한다.

중요한 점:

- Shared memory access index가 tree pattern으로 변한다.
- 각 phase 사이에 `__syncthreads()`가 필요하다.
- Power-of-two 크기에서 설명이 단순하지만 임의 크기는 padding 또는 boundary 처리가 필요하다.

## Bank Conflict 제거

Tree access pattern은 같은 bank에 여러 thread가 몰리기 쉽다. NVIDIA 문서는 conflict-free offset macro를 사용해 shared memory index에 padding을 더한다.

아이디어:

```text
physical_index = logical_index + conflict_free_offset(logical_index)
```

이렇게 하면 stride가 bank 수와 정렬되어 생기는 conflict를 분산할 수 있다.

## Arbitrary Size Array

한 block이 처리할 수 있는 크기를 넘는 배열은 다음 순서로 처리한다.

1. 각 block이 자기 구간을 scan한다.
2. 각 block의 total sum을 별도 배열에 저장한다.
3. block sum 배열을 다시 scan한다.
4. 각 block 결과에 앞 block들의 prefix sum을 더한다.

이 decomposition은 block 간 global sync가 없는 CUDA에서 큰 scan을 구현하는 표준 방식이다.

## 성능 관점

보충자료는 GPU scan이 CPU scan보다 큰 speedup을 낼 수 있음을 보인다. 다만 단순히 병렬화만 해서는 충분하지 않고, work efficiency와 bank conflict 제거가 함께 필요하다.

## 정리

이 보충자료의 핵심은 efficient scan이 알고리즘과 하드웨어 최적화를 동시에 요구한다는 점이다. `O(N)` work의 Blelloch scan, shared memory padding, block decomposition을 합쳐야 큰 배열에서도 빠른 CUDA scan이 된다.
