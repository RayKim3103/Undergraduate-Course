# AMBA, AHB, APB, Memory Transfer

tags: #basic-digital-experiment #amba #ahb #apb #bus #memory #zynq

관련 노트: [[07 Polling Interrupt Timer - 폴링 인터럽트 타이머]], [[10 Audio IP Digital Filters Stream Delay - 오디오 IP 디지털필터]]

## 핵심 요약

이 자료는 ARM 계열 SoC에서 많이 쓰이는 AMBA bus architecture를 소개하고, ASB/AHB/APB의 특징과 timing을 비교한다. 실험에서는 Zynq PS에서 AHB-Lite bridge를 통해 custom memory/multiplier 회로에 write/read를 수행하며 bus transfer의 address phase, data phase, alignment, timing 문제를 관찰한다.

## AMBA 개요

AMBA(Advanced Microcontroller Bus Architecture)는 ARM 기반 SoC에서 processor, memory, peripheral, custom IP를 연결하기 위한 bus 표준이다.

발전 흐름은 다음과 같다.

| 세대 | 특징 |
|---|---|
| ASB/APB | 초기 embedded bus 구조 |
| AHB | 고성능 system bus, burst와 pipeline 지원 |
| AXI | 고성능 point-to-point channel 구조 |
| ACE | cache coherency까지 확장 |

## ASB

ASB(Advanced System Bus)는 embedded microcontroller용 high-performance bus이다.

### 특징

- 16-bit 또는 32-bit data bus
- address/data non-multiplexed
- multiple master 지원
- pipelined transfer
- central decoder와 arbiter 사용

### 기본 동작

1. Master가 bus 사용을 요청한다.
2. Arbiter가 master에게 grant를 준다.
3. Master가 address와 control을 낸다.
4. Decoder가 대상 slave를 선택한다.
5. Slave가 response를 내고 data transfer가 완료된다.

### Transfer type

| Type | 설명 |
|---|---|
| NONSEQUENTIAL | 단일 전송 또는 burst의 첫 전송 |
| SEQUENTIAL | burst 내 연속 주소 전송 |
| ADDRESS-ONLY | idle, handover, speculative decode 등 데이터 없는 주소 단계 |

## AHB

AHB(Advanced High-performance Bus)는 ASB보다 확장된 고성능 bus이다.

### 특징

- burst transfer 지원
- single clock edge 동작
- 64/128-bit bus 확장 가능
- address phase와 data phase pipeline
- 고성능이지만 APB보다 전력과 복잡도가 크다.

### 주요 신호

| 신호 | 의미 |
|---|---|
| `HADDR` | 주소 |
| `HWRITE` | write/read 구분 |
| `HTRANS` | transfer type |
| `HSIZE` | transfer 크기 |
| `HBURST` | burst 종류 |
| `HWDATA` | write data |
| `HRDATA` | read data |
| `HREADY` | slave ready |
| `HRESP` | response |

`HREADY`가 low이면 slave가 아직 준비되지 않았다는 뜻이며, transfer가 연장된다.

### Transfer type

| HTRANS | 의미 |
|---|---|
| IDLE | 전송 없음 |
| BUSY | burst 유지 중 일시 지연 |
| NONSEQUENTIAL | 새 전송 또는 burst 첫 전송 |
| SEQUENTIAL | burst의 후속 전송 |

## APB

APB(Advanced Peripheral Bus)는 저전력ㆍ저복잡도 peripheral에 적합한 bus이다.

### 특징

- pipeline이 없다.
- setup cycle과 enable cycle로 단순하게 동작한다.
- bandwidth 요구가 낮은 timer, UART, GPIO 등에 적합하다.

### 주요 신호

| 신호 | 의미 |
|---|---|
| `PADDR` | peripheral 주소 |
| `PSELx` | slave 선택 |
| `PENABLE` | enable phase 표시 |
| `PWRITE` | write/read 구분 |
| `PWDATA` | write data |
| `PRDATA` | read data |
| `PREADY` | transfer 완료 |

## Experiment 1: AHB Bus System

### 구성

- Zynq7 Processing System
- AXI SmartConnect
- AXI AHB-Lite Bridge
- 외부 AHB interface
- top module과 constraint

### 동작

Vitis C 코드에서 base address에 0, 1, 2를 1초 간격으로 write한다. AHB slave 쪽에서는 `HREADY`와 `HRESP` 조건에 따라 transfer 완료 여부가 달라진다.

`HREADY`가 on이고 `HRESP`가 OK일 때만 LED 값이 정상적으로 갱신된다. 이 실험은 bus response와 ready 신호가 실제 data transfer 완료를 결정한다는 점을 보여준다.

## Experiment 2: Memory Write

### 핵심 구현

custom AHB module에 memory register `mem_A`부터 `mem_H`를 두고, write transfer가 들어오면 주소에 따라 해당 register를 갱신한다.

```text
write address phase: HADDR[4:2] capture
write data phase: HWDATA write
```

32-bit word access이므로 주소 bit `[1:0]`은 byte offset이고, word index는 `[4:2]`로 볼 수 있다.

### 결과

C 코드가 base address에 값을 write하면 `mem_A`가 갱신되고, LED 하위 4-bit에 해당 값이 표시된다.

## Experiment 3: Memory Read와 Multiplier

### 확장 구조

- `mem_A`부터 `mem_H`까지 8개 register를 C 코드로 write한다.
- `mem_I`는 `mem_A * mem_B * ... * mem_H` 결과를 저장한다.
- read address는 register가 9개이므로 더 넓은 index가 필요하다.

주소 예:

```text
base + 0x00 -> mem_A
base + 0x04 -> mem_B
...
base + 0x1C -> mem_H
base + 0x20 -> mem_I
```

1부터 8까지 write하면 `mem_I = 40320`이 되어야 한다.

## Timing 이슈

초기 read 실험에서 read 결과가 한 단계씩 밀려 보이는 문제가 발생했다. 원인은 AHB의 address phase와 data phase가 clock edge 기준으로 pipeline되어 있고, `Xil_In32`가 기대하는 시점과 Verilog read data 준비 시점이 어긋났기 때문이다.

관찰된 해결 방법은 다음과 같다.

- 같은 주소를 한 번 더 read해 안정된 값을 얻는다.
- read data assignment 순서를 조정한다.
- `negedge HCLK`에서 read data를 준비하도록 바꿔 timing을 맞춘다.

마지막 방법은 실험에서는 동작했지만, AHB protocol 관점에서는 정석적인 해결이라기보다 timing을 맞춘 우회에 가깝다.

## Address Alignment

`Xil_Out32`는 32-bit write이므로 주소는 4 byte 단위로 정렬되어야 한다.

```text
base + 0  -> 가능
base + 1  -> 부적절
base + 2  -> 부적절
base + 3  -> 부적절
base + 4  -> 가능
```

8-bit access를 사용하면 byte 주소 접근이 가능하고, 16-bit access는 2 byte 정렬이 필요하다.

## 시험ㆍ복습 체크포인트

- AHB와 APB의 목적과 복잡도 차이를 설명할 수 있어야 한다.
- AHB의 address phase/data phase pipeline을 이해해야 한다.
- `HREADY`, `HRESP`, `HTRANS`, `HWRITE`의 의미를 말할 수 있어야 한다.
- 32-bit MMIO 접근에서 주소가 4 byte 단위로 증가하는 이유를 설명할 수 있어야 한다.

