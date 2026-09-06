---
title: "09. PS/PL, AXI, PYNQ, and ILA"
pages: 94
tags: [intelligent-system, lecture-note, AXI, PYNQ, ZYNQ, ILA, FPGA]
---

# 09. PS/PL, AXI, PYNQ, and ILA

> 이전: [[08 UART and Memory Loopback]]
> 다음: [[10 Computer Vision Neural Networks and CNN]]

## 학습 목표

Week9 자료는 FPGA 단독 RTL 실습에서 ZYNQ 기반 PS/PL system으로 확장한다.

- Processing System(PS)과 Programmable Logic(PL)
- AXI protocol
- control/status register
- memory-mapped control
- Integrated Logic Analyzer(ILA)
- PYNQ setup
- custom IP packaging
- block design
- PYNQ Python driver

## PS와 PL

ZYNQ 계열 FPGA는 processor와 FPGA fabric을 함께 가진다.

| 블록 | 역할 |
|---|---|
| PS(Processing System) | ARM processor, software-controlled |
| PL(Programmable Logic) | Verilog/VHDL로 설계한 custom hardware |
| Interconnect | PS와 PL 사이 data/control 전달, 예: AXI |

PS는 control register를 write/read하고 start signal을 보내며, PL은 실제 하드웨어 연산을 수행한다.

## AXI Protocol

AXI(Advanced eXtensible Interface)는 ARM AMBA 계열의 고속 bus protocol이다.

특징:

- memory-mapped transaction
- address와 data phase 분리
- read/write channel 분리
- ready/valid handshake
- synchronized clock operation

AXI 주요 channel:

| Channel | 역할 |
|---|---|
| AR | Read Address |
| R | Read Data |
| AW | Write Address |
| W | Write Data |
| B | Write Response |

## Ready/Valid Handshake

AXI transfer는 sender의 `valid`와 receiver의 `ready`가 동시에 1일 때 발생한다.

```text
transfer = valid && ready
```

이 방식은 producer/consumer 속도 차이를 흡수하고 channel별 독립 동작을 가능하게 한다.

## Control/Status Register

custom accelerator는 PS가 직접 내부 신호를 wire로 건드릴 수 없으므로 register interface를 둔다.

| register 종류 | 방향 | 예 |
|---|---|---|
| control register | PS -> PL | mode, start, stride, channel 수 |
| status register | PL -> PS | done, busy, error |

software는 특정 memory address에 값을 write해서 PL을 제어하고, 특정 address를 read해서 상태를 확인한다.

## Memory-Mapped Operation 예시

calculator 예:

1. operand0을 `0x0000`에 write
2. operand1을 `0x0004`에 write
3. operator를 `0x1000`에 write
4. start를 `0x1004`에 1로 write 후 0으로 복귀
5. done을 `0x1008`에서 polling
6. output을 `0x0008`에서 read

AXI address는 byte address이므로 32-bit word는 주소가 4씩 증가한다.

## Accelerator Register Map 예시

자료에는 accelerator용 control/status register map이 제시된다.

예:

- `core_start`
- `core_status`
- `input_ch`
- `output_ch`
- `tile_width`
- `tile_height`
- `stride`
- `shamt`

설계자는 register 효율과 software 편의 사이에서 bit packing 방식을 선택할 수 있다.

## ILA

ILA(Integrated Logic Analyzer)는 Vivado에서 제공하는 on-chip debugging IP이다.

필요한 이유:

- simulation testbench는 실제 board behavior를 완전히 반영하지 못한다.
- FPGA 내부 신호는 외부 pin으로 직접 볼 수 없다.
- ILA를 연결하면 내부 FSM, memory interface, control signal waveform을 실시간으로 볼 수 있다.

ILA 동작:

1. 내부 signal을 probe로 연결한다.
2. trigger condition을 설정한다.
3. FPGA 내부 BRAM ring buffer에 waveform을 저장한다.
4. JTAG을 통해 host PC Vivado Hardware Manager에서 확인한다.

## Trigger와 Ring Buffer

ILA는 trigger event 전후의 data를 저장할 수 있다. ring buffer는 계속 최신 waveform을 저장하다가 trigger가 발생하면 trigger index 기준 앞뒤 cycle을 보존한다.

사용 예:

- `start`가 single-cycle pulse인지 확인
- `done`이 정상 timing에 올라오는지 확인
- SRAM/BRAM enable, write enable, address, data가 맞게 움직이는지 확인

## PYNQ

PYNQ는 AMD Xilinx platform을 위한 Jupyter 기반 framework이다.

구성:

- PS에서 PYNQ-Linux 실행
- Python API로 PL overlay 제어
- bitstream과 hardware handoff file을 이용해 FPGA logic 접근

PYNQ는 FPGA overlay를 Python library처럼 사용할 수 있게 해준다.

## PYNQ-Z2 주요 자원

자료 기준:

- ARM Cortex-A9 dual-core processor
- ARMv7 32-bit CPU
- programmable logic slices
- block RAM
- DSP slices
- DDR3 memory
- MicroSD, USB, Ethernet

## Practice: PYNQ Driver for SRAM Controller

이전 SRAM controller를 PS에서 제어할 수 있도록 PYNQ driver를 작성한다.

구조:

```text
PYNQ Python -> AXI -> Control/Status Register -> SRAM Controller
PYNQ Python -> AXI -> SRAM1/SRAM2 memory
```

Python driver 요구:

1. `__init__`에서 SRAM1, SRAM2, control/status register 객체 선언
2. input data를 SRAM1에 저장
3. control register로 start
4. status register에서 done polling
5. SRAM2에서 output result load
6. answer array와 비교

## IP Packaging

custom top module을 PS block design에서 쓰려면 IP로 package해야 한다.

흐름:

1. PL synthesis로 RTL 검증
2. top module을 IP로 package
3. control/status register를 AXI peripheral로 package
4. IP repository path를 Vivado project에 등록
5. block design에 custom IP 추가

## Control/Status Register Practice

SRAM controller용 CSR:

- control: `start`
- status: `done`

중요한 설계 요구:

- Python에서 register write는 수백 cycle 동안 유지될 수 있다.
- PL의 `start`는 single-cycle pulse가 필요하다.
- start control register는 activation 직후 0으로 restore해야 malfunction을 막는다.
- done signal은 status register에 반영해야 한다.

## Block Design 흐름

1. PYNQ-Z2 project 생성
2. IP repository 등록
3. ZYNQ processing system 추가
4. custom top module 추가
5. CSR IP 추가
6. AXI BRAM controller 2개 추가
7. AXI SmartConnect 추가
8. PS, CSR, BRAM, custom top 연결
9. address editor에서 range 설정
10. ILA IP 추가 및 probe 연결
11. Validate Design
12. HDL wrapper 생성
13. bitstream 생성

## Address와 Data Width 조정

AXI BRAM controller는 byte address를 사용하지만 BRAM native address는 word index를 사용할 수 있다.

예:

- 32-bit AXI data는 주소가 4씩 증가
- BRAM address는 1씩 증가
- 따라서 LSB 제거 또는 slice IP로 address bit 조정이 필요할 수 있다.

SRAM1/2의 data width가 다르면 slice/concat IP를 사용해 width를 맞춘다.

## Bitstream Upload

PYNQ에서 overlay를 쓰려면 다음 파일이 필요하다.

- `.bit`: generated bitstream
- `.hwh`: hardware handoff file

두 파일 이름은 같아야 하며, PYNQ Jupyter Notebook에 upload한다.

## ILA Practice Report

보고서에 포함할 내용:

- Control/Status Register code
- PYNQ verification result
- ILA waveform for `start` signal
- discussion

## 체크포인트

- PS는 software control, PL은 custom hardware execution을 담당한다.
- AXI는 ready/valid handshake와 memory-mapped address가 핵심이다.
- CSR은 software와 hardware 사이의 가장 기본적인 제어 interface이다.
- `start`는 single-cycle pulse로 변환해야 한다.
- ILA는 board에서만 드러나는 timing/control 문제를 잡는 필수 도구이다.
- PYNQ driver는 overlay, MMIO, memory access, polling을 한 class로 정리하면 좋다.
