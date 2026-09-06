---
title: "06. FPGA BRAM and DPRAM Wrapper"
pages: 24
tags: [intelligent-system, lecture-note, BRAM, DPRAM, Vivado, memory]
---

# 06. FPGA BRAM and DPRAM Wrapper

> 이전: [[05 SRAM and Memory Controller]]
> 다음: [[07 FIFO and Line Buffer]]

## 학습 목표

Week5-2 자료는 FPGA 내부 Block RAM(BRAM)을 Vivado에서 생성하고, simple dual-port RAM wrapper와 memory controller를 설계하는 방법을 다룬다.

## Block RAM이란

BRAM은 FPGA fabric 안에 물리적으로 구현된 on-chip RAM이다.

특징:

- programmable logic과 직접 연결 가능
- bus system 없이 native port로 접근 가능
- FPGA 종류에 따라 용량이 다름
- 보통 18 Kb 또는 36 Kb unit
- dual-port 구성 가능

큰 register file을 LUT/FF로 만들면 resource가 낭비되므로, 큰 memory는 BRAM을 사용한다.

## Vivado BRAM Generator 선택

주요 선택지:

| 항목 | 선택 예 |
|---|---|
| Interface | Native 또는 AXI |
| Memory Type | Single Port RAM, Simple Dual Port RAM, True Dual Port RAM, ROM |
| ECC | 과제 범위 밖 |
| Byte Write Enable | byte 단위 write mask |
| Operating Mode | Write First, Read First, No Change |
| Output Register | timing 개선용 register |

## Single/Simple/True Dual Port

| 타입 | Port 특성 |
|---|---|
| Single Port RAM | 하나의 port에서 read/write |
| Simple Dual Port RAM | 한 port write, 다른 port read |
| True Dual Port RAM | 두 port 모두 read/write 가능 |
| ROM | read only |

FIFO나 동시에 read/write가 필요한 구조에서는 single port RAM으로는 collision 문제가 생길 수 있다.

## Byte Write Enable

data width가 32-bit일 때 byte write enable을 사용하면 특정 byte만 갱신할 수 있다.

예:

- 기존: `0x00_00_00_00`
- write data: `0x11_11_11_11`
- `we = 4'b1100`
- 결과: `0x11_11_00_00`

byte ordering과 endian-like mapping을 정확히 확인해야 한다.

## BRAM Operating Mode

### Write First

같은 address에 read/write가 동시에 발생하면 새로 쓴 data가 output에 즉시 반영된다. pipeline register처럼 사용할 때 적합하다.

### Read First

같은 address에 read/write가 동시에 발생하면 이전 data가 output된다. read-modify-write가 필요할 때 적합하다.

### No Change

동시 read/write 때 output이 이전 값을 유지한다. output glitch를 줄이고 안정적 상태를 유지하고 싶을 때 적합하다.

## Output Register와 Timing

BRAM read delay `tREAD` 이후 combinational logic delay `tCOMB`가 이어지면 setup margin이 부족해 timing violation이 날 수 있다.

해결:

- primitive output register 사용
- core output register 사용
- target CLB와 BRAM 사이 long routing path에 fetch FF 추가

대신 read latency가 늘어나므로 controller FSM에서 latency를 반영해야 한다.

## Memory Initialization

simulation의 `$readmemh`와 달리 board implementation에서는 Vivado BRAM IP의 initialization file을 설정해야 한다.

COE 파일 예:

```text
memory_initialization_radix=16;
memory_initialization_vector=
0001,
0001,
...
;
```

radix는 2, 10, 16 등으로 지정할 수 있다.

## Global vs Out-of-Context Synthesis

| 방식 | 설명 |
|---|---|
| Global | 전체 design synthesis flow에 포함, 일반적 기본값 |
| OOC per IP | IP를 별도로 synthesize하고 top에서는 black box처럼 연결 |

OOC는 큰 프로젝트나 재사용 IP에서 synthesis 시간을 줄일 수 있다.

## Practice: Memory Wrapper for DPRAM

요구 configuration:

- Native interface
- Simple Dual Port RAM
- No ECC
- No Byte Write Enable
- Port A: width 16, depth 256, Read First, Enable Port
- Port B: width 16, Primitive Output Register
- init file: `initialize_memory.coe`

## Practice 동작 요구

구조:

```text
Dual Port RAM
Port B: read consecutive addresses
Accumulation register
Port A: write accumulated values back
```

초기 RAM은 모든 address에 `0x0001`이 저장되어 있다. controller는 port B로 순차 read하면서 누적값을 만들고, port A로 연속 address에 다시 write한다.

예상 결과:

- address 0: `0x0001`
- address 1: `0x0002`
- address 2: `0x0003`
- ...
- address 255: `0x0100`

## Testbench와 Done

- testbench가 reset 후 `start` signal을 준다.
- 모든 address가 rewrite되면 `done`을 high로 만든다.
- `done` 이후 RAM control은 testbench가 가져가서 값을 읽고 비교한다.
- BRAM은 `$writememh`로 쉽게 debug하기 어려우므로 testbench가 address별로 read해 검증한다.
- read latency 2 clock cycles를 고려해야 한다.

## 체크포인트

- BRAM 설정은 interface, port type, width/depth, operating mode, latency가 핵심이다.
- output register를 쓰면 timing은 좋아지지만 latency가 증가한다.
- `$readmemh`와 COE initialization을 구분한다.
- controller는 BRAM latency에 맞춰 state를 설계해야 한다.
- `done` 이후 testbench가 memory를 읽을 수 있도록 control ownership을 고려한다.
