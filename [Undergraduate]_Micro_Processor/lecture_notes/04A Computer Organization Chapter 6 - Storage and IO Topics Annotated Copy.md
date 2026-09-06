# Computer Organization Chapter 6 - Storage and I/O Topics Annotated Copy

tags: #micro-processor #computer-organization #io #storage #bus #interrupt #dma #annotated-copy

관련 노트: [[04 Computer Organization Chapter 6 - Storage and IO Topics]], [[05 Computer Organization Chapter 7 - Multiprocessors]]

## 핵심 요약

이 자료는 [[04 Computer Organization Chapter 6 - Storage and IO Topics]]와 같은 84쪽 Storage and Other I/O Topics 강의의 별도 사본이다. 텍스트 추출 기준으로 본문 내용은 동일하며, I/O 성능 지표, disk access, bus protocol, arbitration, OS I/O, polling, interrupt, DMA를 같은 순서로 다룬다.

## 이 사본을 볼 때의 관점

이 파일은 같은 강의의 duplicate 또는 annotation용 copy로 보인다. 따라서 개념 정리는 원 강의 노트와 같게 보되, 복습할 때는 다음 항목을 중점적으로 확인하면 좋다.

- I/O가 processor 성능 향상 후 system bottleneck이 되는 이유
- Throughput과 latency의 차이
- Disk service time 계산
- Bus protocol과 arbitration
- Polling, interrupt, DMA의 역할 분담

## I/O 성능 핵심

I/O 성능은 하나의 숫자로만 볼 수 없다.

| 지표 | 의미 | 중요한 상황 |
|---|---|---|
| Latency | 한 요청의 응답 시간 | 사용자 상호작용, random access |
| Throughput | 단위 시간당 처리량 | 대용량 전송, batch I/O |
| Bandwidth | link가 전달할 수 있는 data rate | bus/network/storage |

Block size를 키우면 throughput은 좋아질 수 있지만, 작은 요청의 latency는 나빠질 수 있다.

## Disk Access 복습

Hard disk의 접근 시간은 다음 합으로 본다.

```text
service time = seek time + average rotational latency + transfer time + controller overhead
```

평균 rotational latency는 디스크가 원하는 sector까지 평균 반 바퀴 돌아야 한다는 가정에서 나온다.

## Bus 복습

Bus는 address, data, control line으로 구성된다. Transaction은 master가 bus를 얻고, address/control을 내고, slave가 data를 주거나 받는 과정이다.

중요한 구분:

- synchronous bus: 공통 clock 사용
- asynchronous bus: request/ack handshake 사용
- multiplexed bus: address와 data line을 시간적으로 공유
- split transaction: 요청과 응답 사이 bus를 다른 transaction에 넘김

## Arbitration 복습

Multiple master system에서는 누가 bus를 사용할지 결정해야 한다.

| 방식 | 장점 | 약점 |
|---|---|---|
| Centralized | 구현과 priority 관리가 명확 | arbiter 병목과 single point |
| Daisy chain | 단순 | 위치 기반 priority 불공정 |
| Distributed | 중앙 장치 불필요 | 구현 복잡 |
| Collision detection | shared medium에 적합 | 충돌 시 재시도 overhead |

## Polling, Interrupt, DMA

| 방식 | CPU 역할 | 특징 |
|---|---|---|
| Polling | 계속 status 확인 | 단순하지만 CPU 낭비 |
| Interrupt | event 발생 시 handler 실행 | 효율적이지만 overhead와 priority 관리 필요 |
| DMA | setup 후 controller가 직접 전송 | 대량 전송에 적합, cache coherence 주의 |

## 시험ㆍ복습 체크포인트

- 이 사본은 04번 노트와 같은 개념을 담는 자료로 이해한다.
- Disk service time 식을 암기하고 각 항목을 설명할 수 있어야 한다.
- Synchronous/asynchronous bus protocol 차이를 설명할 수 있어야 한다.
- DMA가 bus master로 동작할 수 있다는 점을 이해해야 한다.
- DMA와 cache coherence 문제가 왜 연결되는지 말할 수 있어야 한다.

