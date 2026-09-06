---
title: "02주차 결과 - Vivado PS PL LED 실습"
course: "Embedded System"
week: 2
type: "result"
tags:
  - embedded-system
  - zynq
  - vivado
  - led
---

# 02주차 결과 - Vivado PS PL LED 실습

이전: [[02주차 예비 - Zynq PS PL 기초]]  
다음: [[03주차 예비 - LED와 7-Segment 디지털 시계]]

## 핵심 요약

이번 실습은 Vivado에서 Zynq의 PS와 PL을 각각 사용하는 흐름을 익히는 것이 목적이다. PS 실습에서는 Zynq Processing System IP를 불러와 SDK에서 `Hello World` 펌웨어를 실행하고, PL 실습에서는 RTL과 constraint를 추가해 PushButton, DIPSwitch, LED 동작을 보드에서 검증했다.

## PS 실습 흐름

| 단계 | 내용 | 의미 |
|---|---|---|
| 1 | Vivado project 생성, Zynq 보드 선택 | 대상 하드웨어 설정 |
| 2 | Block Design 생성 | IP 기반 시스템 설계 공간 생성 |
| 3 | Zynq PS IP 추가 | ARM, DDR, I/O peripheral 포함 |
| 4 | DDR, UART, delay 설정 | PS가 실행될 기본 하드웨어 설정 |
| 5 | HDL Wrapper 생성 | Block Design을 상위 RTL에서 인스턴스화 가능하게 변환 |
| 6 | Bitstream 생성 | FPGA 구성 파일 생성 |
| 7 | Hardware export, SDK 실행 | PS 펌웨어 개발 환경 연결 |
| 8 | Hello World 프로젝트 생성 | ARM에서 실행될 소프트웨어 작성 |
| 9 | Program Device, GDB 실행 | 보드 프로그래밍 및 디버깅 |

IP는 이미 설계된 재사용 가능한 하드웨어 블록이다. 이번에는 Xilinx의 Zynq Processor System IP를 사용했다. HDL Wrapper는 Block Design을 Verilog 모듈처럼 상위 설계에 연결하기 위한 껍데기 역할을 한다.

## PL 실습 흐름

PL 실습에서는 `top.v`와 `top.xdc`를 추가하고 bitstream을 만들어 보드에 다운로드했다.

- `top.v`: LED 동작을 정의하는 RTL
- `top.xdc`: PushButton, DIPSwitch, LED, clock, reset 등의 물리 핀 매핑
- `Program Device`: 생성한 bitstream을 PL에 다운로드

## 기존 RTL 동작

기존 `top.v`는 다음 입출력을 사용한다.

| 신호 | 역할 |
|---|---|
| `PushButton[2:0]` | 버튼 입력 |
| `DIPSwitch[7:0]` | 8비트 스위치 입력 |
| `LED[7:0]` | LED 출력 |
| `CLK` | 동기식 동작 기준 clock |
| `RESETn` | Active-low reset |
| `RegPushButton[2:0]` | 이전 button 상태 저장 |

동작:

- reset 시 `RegPushButton`과 `LED`를 0으로 초기화한다.
- 버튼 상태를 `RegPushButton`에 저장한다.
- `(!PushButton[i]) && RegPushButton[i]` 조건으로 버튼을 **누른 순간이 아니라 뗀 순간** 을 검출한다.
- Button 0: `LED <= DIPSwitch`
- Button 1: 오른쪽 끝 LED만 켠다.

## Quiz RTL 동작

Quiz에서는 버튼별 기능을 다음처럼 바꾸었다.

| 버튼 | 동작 |
|---|---|
| PushButton 0 | LED 전체 off |
| PushButton 1 | 가장 오른쪽 LED on |
| PushButton 2 | LED 점등 위치를 한 칸씩 왼쪽으로 이동, 끝에 도달하면 다시 오른쪽으로 순환 |

보드에서 보이는 LED 방향과 Verilog bit order가 반대이므로, 왼쪽 이동처럼 보이게 하려면 `LED <= LED >> 1`을 사용한다. 이 부분은 물리 배치와 논리 bit numbering이 다를 때 반드시 확인해야 하는 포인트다.

## RegPushButton의 의미

`RegPushButton`은 버튼의 이전 상태를 저장하는 register다. 버튼을 누르고 있을 때 clock마다 동작이 반복되지 않도록, 버튼을 뗀 순간의 edge만 잡기 위해 사용한다.

만약 `RegPushButton` 없이 현재 `PushButton`만 조건에 사용하면, 버튼을 누르고 있는 동안 모든 clock edge마다 LED가 shift된다. FPGA clock은 매우 빠르기 때문에 사람 눈에는 LED가 전부 켜진 것처럼 보일 수 있다.

## Shift와 Divider

`LED >> 1`은 수치적으로 2로 나누는 것과 같지만, 하드웨어 합성 관점에서는 다르다.

- `>> 1`: shifter로 합성되어 단순하고 빠르다.
- `/ 2`: divider로 해석될 수 있어 불필요하게 무거운 회로가 될 수 있다.

상수 2의 나눗셈처럼 shift로 표현 가능한 연산은 하드웨어 설계에서 shift를 사용하는 것이 효율적이다.

## DIPSwitch

DIPSwitch는 보드의 8개 스위치 입력이다. `[7:0] DIPSwitch`로 선언하여 스위치 상태를 8비트 데이터로 받고, 버튼 입력에 따라 그 값을 LED에 반영할 수 있다.

## 정리

2주차 결과의 핵심은 PS와 PL의 실습 흐름을 분리해서 익힌 점이다. PS는 SDK와 펌웨어, UART 출력 중심이고, PL은 RTL, `.xdc`, bitstream, 물리 입출력 검증 중심이다. 또한 `RegPushButton`을 이용한 edge 검출과 shift 기반 LED 제어는 이후 버튼 입력 처리의 기본 패턴이 된다.
