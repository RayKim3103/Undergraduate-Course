---
title: "02주차 예비 - Zynq PS PL 기초"
course: "Embedded System"
week: 2
type: "pre"
tags:
  - embedded-system
  - zynq
  - fpga
  - vivado
---

# 02주차 예비 - Zynq PS PL 기초

이전: 없음  
다음: [[02주차 결과 - Vivado PS PL LED 실습]]

## 핵심 요약

이번 예비보고서는 Zynq SoC의 두 축인 PS와 PL을 구분하고, Vivado에서 두 영역을 이용해 설계를 구현한 뒤 FPGA 보드에서 확인하는 실습을 준비한다. 핵심은 **고정된 ARM 기반 처리 시스템(PS)** 과 **사용자가 회로를 직접 구성하는 프로그래머블 로직(PL)** 이 하나의 칩 안에서 AXI 인터페이스로 연결된다는 점이다.

## 목표

- Zynq가 단순 FPGA가 아니라 ARM 프로세서와 FPGA fabric이 결합된 SoC임을 이해한다.
- PS와 PL의 역할, 장점, 제약을 비교한다.
- Vivado에서 PS/PL 설계를 만들고 bitstream을 통해 보드에 다운로드하는 흐름을 이해한다.
- JTAG, USB Platform Cable, Xilinx SDK의 역할을 구분한다.

## Zynq 구조

| 구분 | 의미 | 특징 | 대표 구성 |
|---|---|---|---|
| PS | Processing System | 고정된 ARM 기반 시스템 | Dual Cortex-A9, DDR 컨트롤러, I/O peripheral |
| PL | Programmable Logic | 사용자가 HDL로 구성하는 하드웨어 | CLB, LUT, FF, BRAM, DSP, I/O block |
| 연결 | PS-PL interface | 고속 데이터 교환 | AXI bus |

Zynq의 중요한 특징은 소프트웨어와 하드웨어를 동시에 활용할 수 있다는 것이다. 일반적인 제어 흐름이나 OS, 통신 처리는 PS에서 수행하고, 병렬성이 큰 연산이나 정해진 신호 처리 회로는 PL에 구현하여 성능을 끌어올릴 수 있다.

## PS의 역할

PS에는 ARM dual Cortex-A9, I/O peripheral, memory interface가 포함된다. 실습 관점에서는 다음 역할이 중요하다.

- DDR 메모리와 주변장치를 초기화하고 제어한다.
- UART, SDIO, Ethernet, I2C 같은 인터페이스를 사용한다.
- SDK에서 작성한 펌웨어를 실행한다.
- 필요하면 bitstream을 읽어 PL을 프로그래밍할 수 있다.

Zynq에서는 PS가 SDIO 또는 Flash Memory에서 bitstream을 읽어 PL에 다운로드할 수 있다. Artix-7, Kintex-7 같은 순수 FPGA가 전용 PROM에서 bitstream을 읽는 방식과 비교하면, Zynq는 PS를 통해 PL 관리가 가능하다는 점에서 유연하다.

## PL의 역할

PL은 Verilog/VHDL 같은 HDL로 논리 회로를 직접 설계하는 영역이다. 주요 자원은 다음과 같다.

- CLB: 기본 조합/순차 논리 구성
- LUT: 논리 함수를 테이블 형태로 구현
- Flip-flop: 상태 저장
- Cascadable adder: 다중 비트 산술 연산
- 36Kb Block RAM: 온칩 메모리
- DSP block: signed multiply, accumulator 등 고속 연산
- Programmable I/O block: 외부 핀과 신호 연결
- ADC 및 센서: 전압, 온도, 외부 analog 입력 측정

PL의 가장 큰 장점은 병렬 처리이다. CPU처럼 명령어를 순차 실행하는 구조가 아니라, 회로 자체가 동시에 동작하므로 영상 처리, 신호 처리, AI 연산처럼 반복적이고 병렬적인 작업에 유리하다.

## JTAG와 Xilinx 도구

| 요소 | 역할 |
|---|---|
| Xilinx USB Platform Cable | 보드와 PC를 연결하여 프로그래밍/디버깅 지원 |
| JTAG connector | Zynq 보드와 USB Platform Cable 연결 |
| Vivado | RTL 설계, block design, synthesis, implementation, bitstream 생성 |
| Xilinx SDK | ARM용 펌웨어 작성, 컴파일, GDB 디버깅 |

Xilinx USB Platform은 ARM debugger처럼 동작하고, SDK는 ARM compiler 역할을 수행한다. 따라서 별도 외부 디버거 없이도 ARM 코드 컴파일과 보드 디버깅이 가능하다.

## Constraint 정보

PL을 사용할 때는 `.xdc` constraint 파일이 필요하다. 이 파일은 RTL 포트와 실제 보드 핀을 연결하고, clock, timing, I/O standard 같은 물리적 조건을 정의한다.

중요한 constraint 종류:

- Physical constraint: 보드 핀과 RTL 포트 매핑
- Timing constraint: clock 주기, 입출력 지연 등 시간 조건
- Configuration constraint: FPGA 설정 방식, I/O 전압 표준 등

## 정리

2주차 예비의 핵심은 Zynq를 **소프트웨어를 실행하는 PS와 하드웨어를 구성하는 PL이 AXI로 연결된 SoC** 로 이해하는 것이다. 이후 실습에서는 PS 단독 Hello World, PL 기반 LED 제어, `.xdc` 핀 매핑, bitstream 생성과 다운로드가 이어진다.
