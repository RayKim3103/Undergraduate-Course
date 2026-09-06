# ARM Chapter 1 - Processor Design Introduction

tags: #micro-processor #arm #embedded-system #processor-design #risc #low-power

관련 노트: [[02 ARM Chapter 2 - ARM Architecture]]

## 핵심 요약

이 자료는 ARM과 embedded system을 소개하고, processor architecture와 organization, abstraction, MU0라는 단순 processor, instruction set design, processor design tradeoff, RISC, low-power design의 큰 흐름을 설명한다. 핵심은 processor를 instruction set, datapath, control logic, memory interface의 계층으로 나누어 이해하는 것이다.

## ARM 개요

ARM은 Advanced RISC Machine을 의미하며, low power와 small area를 강점으로 embedded system과 SoC에서 널리 쓰인다. ARM은 자체 chip을 대량 생산하기보다 processor core와 architecture를 license하는 방식으로 발전했다.

ARM이 많이 쓰이는 이유:

- RISC 기반의 단순하고 효율적인 instruction set
- 전력 소모가 낮아 mobile/embedded에 적합
- SoC 통합에 유리한 IP core 생태계
- Thumb 같은 compressed instruction format 지원
- 다양한 성능/전력 target에 맞춘 core family 제공

## Embedded System

Embedded system은 특정 목적을 수행하기 위해 제품 안에 내장된 computing system이다.

공통 특징:

- 특정 기능에 최적화된 single-function system
- 비용, 전력, 크기 제약이 큼
- 외부 sensor, actuator, I/O와 밀접하게 연결
- real-time constraint가 존재할 수 있음

Hard real-time system은 deadline을 놓치면 system failure로 이어진다. Soft real-time system은 deadline miss가 품질 저하를 만들지만 즉시 치명적 failure가 되지는 않는다.

## Processor Architecture와 Organization

| 구분 | 의미 |
|---|---|
| Architecture | programmer에게 보이는 instruction set, register, memory model |
| Organization | architecture를 실제 hardware로 구현하는 datapath, control, pipeline 구조 |

같은 architecture도 서로 다른 organization으로 구현할 수 있다. 예를 들어 같은 ARM instruction set을 쓰더라도 core마다 pipeline depth, cache, branch predictor, execution unit 구성이 다를 수 있다.

## Abstraction

Hardware design에서는 낮은 계층의 복잡도를 추상화해 높은 계층에서 설계한다.

```text
transistor -> gate -> register/ALU -> datapath -> processor -> system
```

NAND gate 하나도 transistor와 layout 관점에서는 복잡하지만, logic design에서는 하나의 Boolean operator로 다룬다. 이런 추상화가 큰 processor 설계를 가능하게 한다.

## MU0 Processor

MU0는 processor design 개념을 설명하기 위한 매우 단순한 processor이다.

구성 요소:

- Program counter
- Instruction register
- Accumulator
- ALU
- Memory
- Control logic

MU0 instruction set은 단순하지만, instruction fetch, decode, execute, memory access, branch라는 processor 기본 동작을 모두 보여준다.

## Datapath와 Control

Datapath는 data가 이동하고 연산되는 hardware 경로이다. Register, ALU, mux, bus, memory interface가 포함된다.

Control logic은 현재 instruction을 decode해 datapath의 mux select, register write enable, memory read/write 같은 제어 신호를 만든다.

```text
instruction -> control logic -> datapath control signals
```

## Instruction Set Design

Instruction set은 processor가 실행할 수 있는 명령의 집합이다.

고려 요소:

- operand 개수: 0-address, 1-address, 2-address, 3-address
- instruction type: data processing, data transfer, control flow
- addressing mode: immediate, register, base+offset, indirect
- code density와 decode complexity

Instruction set은 software 편의성과 hardware 단순성 사이의 tradeoff를 만든다.

## CISC와 RISC

| 구분 | CISC | RISC |
|---|---|---|
| Instruction | 복잡하고 다양한 명령 | 단순하고 규칙적인 명령 |
| 실행 시간 | 명령마다 다를 수 있음 | pipeline에 유리 |
| Memory access | 다양한 명령에서 가능 | load-store 중심 |
| Hardware | 복잡 | 상대적으로 단순 |

ARM은 RISC 철학에 기반한다. 단순한 instruction과 load-store 구조는 pipeline과 low-power 구현에 유리하다.

## Pipeline과 Hazard

Pipeline은 instruction 실행 단계를 겹쳐 throughput을 높이는 방법이다.

대표 hazard:

- Structural hazard: hardware resource 충돌
- Data hazard: 이전 instruction 결과가 아직 준비되지 않음
- Control hazard: branch로 다음 PC가 불확실함

Pipeline은 latency를 줄인다기보다 단위 시간당 instruction 처리량을 늘린다.

## Low-Power 설계

Embedded processor에서 low power는 핵심 요구사항이다.

CMOS power 구성:

- switching power
- short-circuit power
- leakage power

Low-power strategies:

- VDD 최소화
- clock frequency 조절
- switching activity 감소
- capacitance 감소
- sleep mode와 power management

## 시험ㆍ복습 체크포인트

- Architecture와 organization의 차이를 설명할 수 있어야 한다.
- Embedded system의 real-time constraint를 hard/soft로 구분할 수 있어야 한다.
- MU0를 datapath와 control logic 관점에서 설명할 수 있어야 한다.
- RISC의 핵심 특징과 ARM이 embedded에 적합한 이유를 말할 수 있어야 한다.
- Pipeline hazard 세 종류를 구분할 수 있어야 한다.

