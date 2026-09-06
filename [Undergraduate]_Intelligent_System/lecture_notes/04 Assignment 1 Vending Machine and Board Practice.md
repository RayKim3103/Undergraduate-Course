---
title: "04. Assignment 1 Vending Machine and Board Practice"
pages: 16
tags: [intelligent-system, lecture-note, assignment, vending-machine, SSD, debouncer, clock-divider]
---

# 04. Assignment 1 Vending Machine and Board Practice

> 이전: [[03 FSM and Sequential System Design]]
> 다음: [[05 SRAM and Memory Controller]]

## 학습 목표

Week3-1 자료는 첫 번째 개인 과제인 vending machine 설계 지침과 board practice를 다룬다.

- vending machine FSM 요구사항
- FPGA board 입출력 매핑
- seven-segment display
- button debouncer
- clock divider
- SSD/GPIO counter practice
- FSM controller practice

## Assignment 1: Vending Machine

판매 항목 가격:

| 항목 | 가격 |
|---|---:|
| Item 1 | $3 |
| Item 2 | $5 |
| Item 3 | $7 |

사용 가능한 coin:

| coin | value |
|---|---:|
| coin 1 | $1 |
| coin 2 | $5 |
| coin 3 | $10 |

## Vending Machine 사용 절차

1. 기계를 turn on 또는 reset한다.
2. 판매할 item stock을 채운다. 최대 stock은 5.
3. coin을 넣어 balance를 증가시킨다. 최대 balance는 50.
4. item을 선택하면 stock과 balance가 감소한다.

이 흐름은 자연스럽게 FSM으로 구현된다.

가능한 state 예:

- OFF
- IDLE
- ITEM_FILLING
- COIN_INSERTING
- SELLING
- ERROR 또는 WAIT

## Board I/O 매핑

자료의 board setup:

| 입력/출력 | 역할 |
|---|---|
| SW3 | vending machine on/off, active high |
| SW1-SW2 | mode control |
| SW0 | reset, active high |
| BTN1-BTN3 | mode별 task 수행: filling, coin inserting, selling |
| LED5 | 기계 동작 중 표시 |
| LED2-LED4 | item out-of-stock 또는 recently filled 표시 |
| SSD | 최근 채운 stock 또는 balance 표시 |

Active High는 입력이 1일 때 활성화되는 신호이다. Active Low는 0일 때 활성화된다.

## Seven Segment Display

SSD는 10진 숫자를 표시하는 LED segment 장치이다.

- segment: A-G
- 선택적으로 decimal point 포함
- 권장 동작 clock: 50 Hz

숫자 0의 예:

- A, B, C, D, E, F = 1
- G = 0

실제 board에서는 common anode/cathode 여부에 따라 active level이 달라질 수 있으므로 constraint와 board manual을 확인해야 한다.

## Button Debouncing

기계식 버튼은 누르는 순간 jitter/bounce가 발생한다. 이를 그대로 쓰면 버튼 한 번이 여러 번 눌린 것처럼 보일 수 있다.

해결:

- D flip-flop 여러 개를 직렬로 연결해 signal을 안정화한다.
- 이 수업에서는 2 DFF 사용이 권장된다.

기본 아이디어:

```text
raw button -> DFF -> DFF -> debounced signal
```

필요하면 edge detector를 추가해 버튼 입력을 1 clock pulse로 만든다.

## Clock Divider

FPGA board는 보통 100 MHz clock을 제공하지만, SSD 표시나 사람이 누르는 버튼 처리는 훨씬 낮은 clock이 필요하다.

예:

- 100 MHz -> 50 Hz for SSD
- 100 MHz -> 1 Hz for visible counter update

구현 방식:

- counter가 목표 divide count에 도달하면 output clock 또는 tick을 toggle/assert
- 가능하면 내부 clock을 직접 새로 만들기보다 enable tick을 쓰는 방식이 timing 관리에 더 안전하다.

## Practice 1: SSD and GPIO Counter

요구사항:

- 두 버튼으로 counter 증가/감소
- SSD에 counter 표시
- counter 범위: 00-09 saturation
- 50 Hz 동작
- resetn active low

I/O 예:

| 신호 | board mapping |
|---|---|
| CLK100MHZ | clk |
| SW0 | resetn |
| BTN[0] | in_up |
| BTN[1] | in_down |
| jc[0:3] | aa-ad |
| jd[0:2] | ae-ag |
| jd[3] | cat |

설계 포인트:

- debouncer 필수
- clock divider 필수
- 0보다 작아지거나 9보다 커지지 않도록 saturation 처리

## Practice 2: FSM Controller

요구사항:

- 2 switches로 state 변경: IDLE, UP, DOWN, READY
- SSD는 counting number 표시
- LED[0]은 UP, LED[1]은 DOWN 상태 표시
- UP 상태에서는 1초마다 +1
- DOWN 상태에서는 1초마다 -1
- READY 상태에서는 값 유지
- counter 범위: 0-15 saturation

구조:

```text
clk_divider -> 50 Hz, 1 Hz tick
fsm_ctrl -> state/control
ssd_ctrl -> display
top module -> board I/O 연결
```

## 제출 및 Week4 안내

- Week3의 두 practice에 대한 report 1개 제출
- Week4에는 Assignment 1 board implementation 진행
- written code와 report 모두 제출
- board implementation 수업 출석이 필수

## 체크포인트

- vending machine은 상태와 mode가 분명한 FSM 문제이다.
- 버튼 입력은 반드시 debouncing/edge detection을 고려한다.
- SSD는 사람이 보는 장치라 느린 refresh clock 또는 enable이 필요하다.
- board implementation에서는 `.xdc` pin mapping이 코드만큼 중요하다.
- counter/FSM practice는 vending machine 과제의 작은 구성요소로 이해하면 좋다.
