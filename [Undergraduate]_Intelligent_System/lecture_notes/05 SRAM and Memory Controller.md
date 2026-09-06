---
title: "05. SRAM and Memory Controller"
pages: 30
tags: [intelligent-system, lecture-note, SRAM, memory-controller, Verilog]
---

# 05. SRAM and Memory Controller

> 이전: [[04 Assignment 1 Vending Machine and Board Practice]]
> 다음: [[06 FPGA BRAM and DPRAM Wrapper]]

## 학습 목표

Week5-1 자료는 RAM의 기본 개념, SRAM cell 구조, read/write timing, decoder, 그리고 Verilog SRAM controller practice를 다룬다.

## RAM과 Volatility

RAM(Random Access Memory)은 임의 주소를 읽고 쓸 수 있는 대표적 memory device이다.

| 종류 | 특징 |
|---|---|
| SRAM | 전원이 유지되는 동안 같은 주소에 다시 쓰기 전까지 데이터 유지 |
| DRAM | leakage 때문에 수 ms 단위로 refresh 필요 |
| NVRAM/EEPROM | 전원이 꺼져도 데이터 유지 가능 |

이 수업에서는 controller 설계 관점에서 SRAM/BRAM interface를 주로 다룬다.

## SRAM 내부 구조

SRAM array는 apartment처럼 행/열 구조로 생각할 수 있다.

- address: 읽거나 쓸 위치 선택
- wordline(WL): row selection
- bitline(BL/BLB): column data path
- decoder: address를 wordline 선택 신호로 변환

read/write는 모두 bitline을 통해 수행된다.

## 12T SRAM과 6T SRAM

### 12T SRAM

자료에서는 latch와 bitline을 이용한 단순 구조를 소개한다. 교육용으로 이해하기 쉽지만 면적이 크다.

### 6T SRAM

상용 SRAM에서 널리 쓰이는 구조이다.

- cross-coupled inverter 2개로 1 bit 저장
- access transistor 2개로 BL/BLB와 연결
- cell size가 array 면적 대부분을 차지하므로 layout scaling이 중요

## 6T SRAM Read Operation

절차:

1. BL과 BLB를 VDD로 precharge한다.
2. bitline을 floating시킨다.
3. WL을 올려 cell을 bitline에 연결한다.
4. 저장값에 따라 한쪽 bitline이 내려간다.
5. sense amplifier 또는 read logic이 차이를 감지한다.

주의:

- read 중 내부 node가 뒤집히면 안 된다.
- read stability를 위해 transistor sizing이 중요하다.

## 6T SRAM Write Operation

절차:

1. BL과 BLB에 서로 보수인 data를 강하게 drive한다.
2. WL을 올려 cell을 bitline에 연결한다.
3. bitline driver가 기존 latch 값을 이기고 새 값을 저장한다.

주의:

- writeability를 위해 write driver/access transistor가 feedback inverter를 이길 수 있어야 한다.
- read stability와 writeability는 sizing trade-off를 가진다.

## Decoder for Random Access

N-bit address를 $2^N$개 wordline 중 하나로 변환한다.

기본적으로 $N:2^N$ decoder는 $2^N$개의 N-input gate가 필요하다. N이 커지면 큰 NAND/AND gate가 느려지므로 여러 단계의 작은 gate로 나누어 구현한다.

## Verilog SRAM Modeling

수업 practice의 Verilog SRAM 모델은 RTL simulation용이다.

예시 구조:

```verilog
reg [BW-1:0] mem [0:AMAX-1];
```

- `BW`: 한 address에 저장되는 data bitwidth
- `AMAX`: memory depth, address 개수
- `mem`: 2D register array

주의:

- 이런 register array 방식은 simulation에는 편하지만 FPGA 구현에서는 resource 낭비가 크다.
- 실제 FPGA 구현에는 BRAM IP 또는 memory inference를 사용하는 것이 일반적이다.

## SRAM I/O 신호

| 신호 | 역할 |
|---|---|
| `clk` | memory operation 동기화 |
| `en` | memory read/write 활성화 |
| `we` | write enable, high일 때 write |
| `addr` | read/write할 주소 |
| `din` | write할 data |
| `dout` | read된 data |

## Write Timing

write 조건:

- `en = 1`
- `we = 1`
- `addr` 유효
- `din` 유효

timing 개념:

1. T0: input signal 세팅
2. T1: memory가 clock edge에서 신호 감지
3. T2: `tWRITE` 이후 data 저장 완료

## Read Timing

read 조건:

- `en = 1`
- `we = 0`
- `addr` 유효
- `din`은 don't care

timing 개념:

1. T0: `en`, `we`, `addr` 세팅
2. T1: memory가 clock edge에서 address 감지
3. T2: `tREAD` 이후 `dout`에 data 제공

read latency를 모르면 controller FSM이 한 cycle 빠르거나 늦어지는 버그가 생기기 쉽다.

## Practice: SRAM Controller

목표는 서로 다른 bitwidth/depth를 가진 두 SRAM 사이에서 data를 옮기는 controller를 설계하는 것이다.

- SRAM1: 16-bit x 256
- SRAM2: 32-bit x 192

### Mapping 1: SRAM1 0x00-0x7F

두 개의 16-bit data를 하나의 32-bit word로 concatenate한다.

- lower address data를 MSB에 저장
- 결과는 SRAM2의 0x60-0xBF에 reversed order로 저장

### Mapping 2: SRAM1 0x80-0xFF

16-bit data 하나와 16-bit zero를 concatenate한다.

- even address data는 LSB
- odd address data는 MSB
- 결과는 SRAM2의 0x00-0x5F에 저장

## Controller 설계 포인트

- top module instance 이름을 바꾸면 testbench error가 날 수 있으므로 유지한다.
- `start`는 single clock signal로 주어진다.
- 모든 data transfer가 끝나면 `done`을 high로 올린다.
- SRAM1에서 read한 data를 SRAM2 write timing에 맞게 잡아두는 register가 필요할 수 있다.
- address mapping과 data packing을 FSM state별로 명확히 나눈다.

## Testbench와 Hex File

- `$readmemh`로 SRAM1 초기화
- 경로를 환경에 맞게 수정해야 한다.
- transfer 완료 후 SRAM2를 solution hex와 비교
- error 발생 시 simulation stop 및 `$writememh`로 dump 가능

주의:

- `$readmemh`는 simulation용 기능이다.
- 실제 FPGA BRAM 초기화에는 COE/MEM 파일 설정이 필요하다.

## 체크포인트

- SRAM read/write는 enable, write enable, address, data timing을 함께 봐야 한다.
- read latency를 controller FSM에 반영해야 한다.
- 16-bit에서 32-bit로 data packing할 때 MSB/LSB 위치를 헷갈리지 않는다.
- simulation용 memory model과 실제 FPGA BRAM 구현은 다르다.
