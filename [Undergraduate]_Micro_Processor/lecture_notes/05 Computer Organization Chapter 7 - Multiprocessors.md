# Computer Organization Chapter 7 - Multiprocessors

tags: #micro-processor #multiprocessor #parallelism #cache-coherence #mesi #synchronization #interconnection-network

관련 노트: [[04A Computer Organization Chapter 6 - Storage and IO Topics Annotated Copy]], [[06 SSD Overview 1 - Solid State Disk Basics]]

## 핵심 요약

이 자료는 multiprocessor와 parallel computer를 다룬다. Parallelism의 종류, SISD/SIMD/MIMD 분류, speedup과 Amdahl's law, shared-memory multiprocessor, cache coherence, snooping protocol, MESI, synchronization, interconnection network를 설명한다.

## Parallel Machine의 아이디어

여러 작은 processor를 연결해 더 큰 성능을 얻는 것이 parallel machine의 기본 아이디어이다. 필요할 때 processor 수를 늘려 성능을 확장할 수 있지만, 실제 성능은 communication, synchronization, serial portion 때문에 제한된다.

## Parallelism의 종류

| 종류 | 의미 |
|---|---|
| DLP | Data-level parallelism, 같은 연산을 많은 data에 적용 |
| TLP | Task-level parallelism, 서로 다른 task를 병렬 수행 |
| ILP | Instruction-level parallelism, 한 thread 내부 instruction 병렬성 |
| Thread-level parallelism | 여러 thread 동시 실행 |
| Request-level parallelism | 독립적인 request들을 병렬 처리 |

## Flynn 분류

| 분류 | 의미 | 예 |
|---|---|---|
| SISD | Single instruction stream, single data stream | 전통적 single processor |
| SIMD | Single instruction stream, multiple data streams | vector processor, GPU 일부 |
| MIMD | Multiple instruction streams, multiple data streams | multicore, cluster |

MIMD는 tightly-coupled system과 loosely-coupled system으로 나눌 수 있다.

## Speedup

Speedup은 processor를 늘렸을 때 execution time이 얼마나 줄었는지 나타낸다.

```text
S(n) = T(1) / T(n)
```

Ideal linear speedup은 다음과 같다.

```text
S(n) = n
```

하지만 실제로는 병렬화할 수 없는 부분, communication overhead, load imbalance 때문에 linear speedup이 어렵다.

## Amdahl's Law

Program의 일부만 병렬화된다면 전체 speedup은 serial fraction에 의해 제한된다.

```text
Speedup = 1 / (s + (1-s)/n)
```

| 기호 | 의미 |
|---|---|
| `s` | 병렬화할 수 없는 serial fraction |
| `n` | processor 수 |

Processor 수를 무한히 늘려도 최대 speedup은 `1/s`를 넘지 못한다.

## Shared Memory와 Cache Coherence

Shared-memory multiprocessor에서는 여러 processor가 같은 memory address space를 공유한다. 각 processor가 cache를 가지면 같은 memory block의 복사본이 여러 cache에 존재할 수 있다.

Cache coherence problem:

```text
한 processor가 값을 write했는데 다른 processor cache에는 old value가 남아 있음
```

## Coherence Protocol

대표 방식:

- Write invalidate: write할 때 다른 cache copy를 invalid로 만듦
- Write update: write한 값을 다른 cache copy에 broadcast함

Write invalidate는 같은 word에 여러 번 write할 때 traffic이 적다. Write update는 reader가 최신 값을 빨리 보지만 broadcast traffic이 커질 수 있다.

## Snooping Protocol

Bus 기반 shared-memory system에서는 각 cache controller가 bus transaction을 감시한다. 다른 processor의 read/write miss를 보고 자신의 cache line state를 바꾼다.

기본 state 예:

- Invalid
- Shared
- Modified

## MESI Protocol

MESI는 네 가지 state를 사용한다.

| State | 의미 |
|---|---|
| Modified | 이 cache만 최신 dirty copy 보유 |
| Exclusive | 이 cache만 clean copy 보유 |
| Shared | 여러 cache가 clean copy 공유 |
| Invalid | 유효하지 않음 |

Exclusive state는 shared되지 않은 clean block을 write할 때 bus transaction 없이 Modified로 바꿀 수 있게 해 traffic을 줄인다.

## Synchronization

공유 data를 여러 process/thread가 동시에 접근하면 race condition이 발생할 수 있다. Critical section은 mutual exclusion을 보장해야 한다.

Correctness criteria:

- mutual exclusion
- progress
- bounded waiting

## Test and Set

`test_and_set`은 lock 구현에 쓰이는 atomic operation이다.

```c
test_and_set(lock) {
    old = lock;
    lock = true;
    return old;
}
```

Naive spin lock은 bus traffic을 많이 만들 수 있으므로, local cache에서 spinning하다가 필요한 순간에 atomic operation을 수행하는 optimized synchronization이 필요하다.

## Interconnection Networks

Multiprocessor 성능은 processor 간 연결망에도 크게 의존한다.

| Network | 특징 |
|---|---|
| Bus | 단순하지만 확장성 낮음 |
| Crossbar | 모든 node 간 연결 가능, 비용 큼 |
| Omega network | multistage network, 비용과 성능 절충 |
| Ring | 저비용, latency가 node 수에 의존 |
| Mesh | 2D layout에 자연스러움 |
| 2D Torus | mesh의 edge를 연결해 균형 개선 |
| Hypercube | n차원 구조, 2^n node |

## 시험ㆍ복습 체크포인트

- SISD, SIMD, MIMD를 구분할 수 있어야 한다.
- Amdahl's law로 serial fraction이 speedup을 제한하는 이유를 설명할 수 있어야 한다.
- Cache coherence problem을 예로 설명할 수 있어야 한다.
- Write invalidate와 write update의 차이를 말할 수 있어야 한다.
- MESI 네 state의 의미를 구분할 수 있어야 한다.
- Test-and-set이 atomic해야 하는 이유를 이해해야 한다.

