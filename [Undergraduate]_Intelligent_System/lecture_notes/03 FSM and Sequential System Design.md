---
title: "03. FSM and Sequential System Design"
pages: 22
tags: [intelligent-system, lecture-note, FSM, sequential-logic, Verilog]
---

# 03. FSM and Sequential System Design

> 이전: [[02 Verilog Basics and Logic Design]]
> 다음: [[04 Assignment 1 Vending Machine and Board Practice]]

## 학습 목표

Week2-2 자료는 조합논리와 순차논리의 차이를 FSM 관점에서 정리하고, Moore/Mealy machine과 실제 FSM 설계 예제를 다룬다.

## Logic Circuit의 두 명세

logic circuit은 입력, 출력, 그리고 두 종류의 specification으로 설명된다.

| 명세 | 의미 |
|---|---|
| functional specification | 입력과 출력 사이의 논리 관계 |
| timing specification | 입력 변화 후 출력이 응답하기까지의 delay |

하드웨어 설계에서는 기능이 맞는 것만으로 충분하지 않다. clock period, setup/hold, propagation delay까지 고려해야 한다.

## Combinational Logic과 Sequential Logic

| 구분 | Combinational | Sequential |
|---|---|---|
| 메모리 | 없음 | 있음 |
| 출력 의존성 | 현재 입력 | 현재 입력 + 과거 상태 |
| 예 | decoder, mux, adder | counter, FSM, processor controller |

순차논리는 “상태(state)”를 저장하기 때문에 입력 history가 출력에 영향을 준다.

## State의 의미

state는 시스템의 현재 상황을 요약한 snapshot이다.

예: 금고 lock sequence가 `R3 -> L20 -> R8`이면 상태는 다음처럼 나뉜다.

1. 아무 유효 동작도 수행하지 않은 locked 상태
2. R3까지 완료한 상태
3. R3-L20까지 완료한 상태
4. R3-L20-R8 완료로 unlocked 상태

같은 현재 입력이라도 과거 상태에 따라 다음 동작이 달라지므로 state가 필요하다.

## Clock과 State Transition

상태는 언제 바뀌는가?

- clock edge에서 state register가 next state를 latch한다.
- combinational next-state logic은 한 clock cycle 동안 다음 상태를 계산한다.
- clock period는 combinational path의 최대 delay를 감당할 수 있어야 한다.

FSM 구조:

```text
inputs -> next state logic -> next state -> state register -> current state
                               current state -> output logic -> outputs
```

## Finite State Machine

FSM은 stateful system의 discrete-time model이다.

구성 요소:

- finite number of states
- external inputs
- external outputs
- state transition rules
- output generation rules

응용:

- traffic light
- elevator
- vending machine
- microprocessor controller
- UART controller
- memory controller

## Moore Machine과 Mealy Machine

### Moore Machine

출력이 현재 state에만 의존한다.

$$
output = f(state)
$$

장점:

- 출력이 안정적이다.
- timing 분석이 상대적으로 단순하다.

단점:

- 입력 변화가 출력에 반영되려면 state transition을 기다려야 할 수 있다.

### Mealy Machine

출력이 현재 state와 현재 input에 모두 의존한다.

$$
output = f(state,input)
$$

장점:

- 입력 변화에 빠르게 반응할 수 있다.
- state 수가 줄어들 수 있다.

단점:

- input glitch가 출력에 바로 나타날 수 있어 timing 관리가 더 중요하다.

## Verilog FSM 기본 패턴

권장 구조는 state register, next-state logic, output logic을 분리하는 방식이다.

```verilog
always @(posedge clk or negedge resetn) begin
    if (!resetn) state <= IDLE;
    else state <= next_state;
end

always @(*) begin
    case (state)
        IDLE: next_state = ...;
        ...
        default: next_state = IDLE;
    endcase
end

always @(*) begin
    // output logic
end
```

## T-Bird Tail Lights 예제

자료는 자동차 방향지시등을 FSM 예제로 제시한다.

설계 절차:

1. state register 선언
2. state encoding 정의
3. reset 시 known state로 초기화
4. 현재 state와 input에 따라 next state 결정
5. state에 따라 LED output 결정
6. simulation에서 방향 전환과 reset 동작 확인

## FSM의 한계

state 수가 많아지면 가능한 transition 수가 급격히 늘어난다.

문제:

- maintainability: state 추가/삭제 시 관련 transition 수정 범위가 커진다.
- scalability: state diagram이 복잡해져 가독성이 떨어진다.
- reusability: behavior가 state 내부 조건에 강하게 묶여 재사용이 어렵다.

해결 방향:

- hierarchical FSM
- 작은 FSM 여러 개로 분리
- datapath와 controller 분리
- 공통 동작을 module화

## 체크포인트

- sequential logic에는 memory/state가 있다.
- state transition은 clock edge에서 일어난다.
- Moore는 출력이 state만의 함수, Mealy는 state와 input의 함수이다.
- FSM은 controller 설계의 기본이며 이후 memory, UART, accelerator 제어에 반복적으로 등장한다.
- 큰 FSM은 계층화하거나 여러 작은 FSM으로 나누는 것이 좋다.
