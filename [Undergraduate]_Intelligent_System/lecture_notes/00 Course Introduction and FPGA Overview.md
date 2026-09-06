---
title: "00. Course Introduction and FPGA Overview"
pages: 34
tags: [intelligent-system, lecture-note, course-intro, FPGA, VLSI, AI-SoC]
---

# 00. Course Introduction and FPGA Overview

> 다음: [[01 Vivado Installation and Basic Flow]]

## 강의의 위치

이 자료는 EEE3551 지능형시스템설계및응용의 운영 방식과 큰 기술 배경을 소개한다. 과목의 중심은 “AI 알고리즘을 실제 하드웨어로 설계하고 FPGA에서 검증하는 것”이다.

## 담당 및 선수 지식

- 담당: Prof. Kyuho Lee, Intelligent Systems Laboratory, Yonsei EE
- 연구 키워드: AI SoC, neuromorphic processor, processing-in-memory/computing-in-memory, embedded AI systems
- 선수 과목:
  - Basic Circuit Theory
  - Digital Logic
  - Introductory Digital Labs

## 평가 구조

| 항목 | 비중 | 내용 |
|---|---:|---|
| 출석 | 10% | 3회 결석부터 감점 |
| 개인 프로젝트 | 30% | HDL 기반 디지털 회로 설계 및 FPGA 구현 |
| 팀 프로젝트 | 40% | convolution core, end-to-end CNN accelerator |
| 발표 | 20% | design review, final presentation |

프로젝트 중심 강의이므로 “코드가 simulation에서 맞는가”뿐 아니라 “FPGA board에서 실제로 동작하는가”가 중요하다.

## 프로젝트 구성

### 개인 프로젝트

1. FSM 기반 vending machine
2. UART TRx + memory loopback

개인 프로젝트는 Verilog, FSM, memory, UART, board implementation을 익히는 단계이다.

### 팀 프로젝트

1. Convolution accelerating core
2. End-to-end CNN accelerator

convolution core는 CNN accelerator의 핵심 구성 요소이다. 첫 팀 과제에서 convolution core를 제대로 이해하지 못하면 다음 CNN accelerator 구현이 어려워진다.

## Honor Code와 협업 기준

허용되는 협업은 개념, 요구사항, 일반적인 디버깅 방법, 도구 사용법에 대한 논의이다. 금지되는 것은 타인의 solution, Verilog code, 과거 본인 코드, 생성형 AI가 만든 코드를 그대로 제출하는 것이다.

과제 제출물에는 도움받은 내용을 reference로 명시해야 한다.

## 강의 일정 큰 흐름

| 주차 | 주제 |
|---|---|
| Week 1 | Course introduction |
| Week 2 | Verilog basics, combinational/sequential logic |
| Week 3 | 7-segment RTL/SYN, FSM RTL/SYN |
| Week 4 | Board implementation for assignment 1 |
| Week 5 | Memory, SRAM/BRAM control |
| Week 6 | FIFO, interface |
| Week 7 | UART TRx, assignment 2 board implementation |
| Week 9 | PS/PL, AXI, PYNQ, board implementation |
| Week 10-11 | Neural network basics, CNN, AI hardware |
| Week 12-15 | Team assignment, RTL/SYN/SIM, design review, final presentation |

## Industry Trend: AI와 Big Data

자료는 hyper-connected society, big data, AI를 주요 산업 흐름으로 제시한다. AI workload는 막대한 연산량과 메모리 bandwidth를 요구하므로, GPU/FPGA/ASIC 같은 하드웨어 가속이 중요해진다.

예시로 NVIDIA Blackwell GPU 같은 AI superchip이 언급된다.

## VLSI와 SoC

VLSI(Very-Large-Scale Integration)는 수천 개 이상의 transistor를 하나의 chip에 집적하는 기술이다. SoC(System-on-Chip)는 processor, memory, accelerator, interface 등을 하나의 chip 또는 시스템으로 통합한다.

역사적 흐름:

- ENIAC: 진공관 기반 초기 전자식 컴퓨터
- transistor: Bell Labs, 1948
- integrated circuit: Jack Kilby, 1958
- Intel 4004: 초기 microprocessor
- 현대 microprocessor와 AI accelerator

## Design Abstraction Levels

칩 설계는 여러 추상화 수준을 거친다.

| 수준 | 의미 |
|---|---|
| System level | 전체 기능과 사용 시나리오 |
| Architecture/algorithm level | 연산 구조, 데이터 흐름 |
| Digital system level | datapath, controller, memory |
| Logic level | gate, register, FSM |
| Electrical level | transistor 회로 |
| Layout level | 물리 배치와 배선 |
| Semiconductor level | 공정/소자 |

이 과목은 주로 architecture, digital system, RTL/logic level을 다룬다.

## FPGA란 무엇인가

FPGA(Field-Programmable Gate Array)는 사용자가 원하는 회로로 재구성할 수 있는 집적회로이다.

기본 구성:

- CLB(Configurable Logic Block)
- LUT(Look-Up Table)
- FF(Flip-Flop)
- IOB(I/O Block)
- routing fabric

FPGA는 hard-wired ASIC과 달리 재프로그램 가능하다. 병렬 하드웨어를 직접 구성할 수 있어 CPU/GPU보다 특정 workload에서 더 빠르고 효율적일 수 있다.

## 왜 FPGA를 쓰는가

- 설계 변경이 쉽다.
- ASIC보다 초기 비용이 낮다.
- dedicated hardware 구조로 병렬 연산 가능
- 검증과 교육에 적합하다.
- AI accelerator prototype을 빠르게 구현할 수 있다.

## 강의에서 사용하는 도구

- HDL: Verilog/VHDL
- FPGA board: Xilinx Arty A7/S7, PYNQ-Z2 등
- Tool: Xilinx Vivado

## 핵심 정리

- 이 강의는 Verilog로 하드웨어를 설계하고 FPGA에서 실제 동작을 검증하는 프로젝트형 수업이다.
- 초반에는 digital logic과 FPGA flow를 익히고, 중반에는 memory/FIFO/UART/interface를 구현한다.
- 후반에는 CNN과 AI accelerator 구조를 이해하고, convolution core와 end-to-end accelerator로 확장한다.
- AI 시스템 설계에서 중요한 것은 알고리즘뿐 아니라 data movement, memory bandwidth, hardware parallelism이다.
