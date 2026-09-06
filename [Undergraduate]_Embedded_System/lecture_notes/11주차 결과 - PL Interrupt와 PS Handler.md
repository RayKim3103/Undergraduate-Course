---
title: "11주차 결과 - PL Interrupt와 PS Handler"
course: "Embedded System"
week: 11
type: "result"
tags:
  - embedded-system
  - interrupt
  - ahb
  - apb
  - gic
---

# 11주차 결과 - PL Interrupt와 PS Handler

이전: [[11주차 예비 - Push Button Interrupt System]]  
다음: [[12주차 결과 - Zynq Ubuntu Root File System 구성]]

## 핵심 요약

이번 실습에서는 PL의 Push Button 입력으로 interrupt를 만들고, PS의 GIC와 handler가 이를 처리한 뒤 다시 PL LED를 제어했다. Verilog는 button, interrupt register, LED register를 담당하고, C code는 GIC 초기화와 interrupt routine을 담당한다.

## HW/SW 역할 분리

| 영역 | 역할 |
|---|---|
| PL Verilog | 버튼 edge 감지, interrupt status 저장, LED 출력, APB slave 동작 |
| PS C code | GIC 초기화, handler 등록, interrupt enable, status read, LED value write |
| Bridge | AXI -> AHB -> APB 변환으로 PS register 접근 연결 |

데이터 흐름:

```text
PB input
-> pb_intr
-> INTR
-> Core0_nIRQ / GIC
-> DeviceDriverHandler
-> Xil_In32LE / Xil_Out32LE
-> ahb2apb_zynq
-> pb_intr LED register
```

## `ahb2apb_zynq.v`

AHB master와 APB slave 사이의 bridge다. AHB의 valid transaction을 감지하면 APB의 `PSEL`, `PENABLE`, `PWRITE`, `PADDR`, `PWDATA`로 변환한다.

주요 동작:

- `st_ctrl=00`: idle 상태
- `M_AHB_htrans[1]=1`: valid transfer 감지
- 다음 상태에서 `PENABLE=1`로 APB access 수행
- `PADDR=reg_addr`, `PWDATA=M_AHB_hwdata`
- read data는 `PRDATA`를 `M_AHB_hrdata`로 전달

APB는 AHB보다 단순하므로, bridge는 복잡한 bus transaction을 peripheral register 접근 형태로 낮춰주는 역할을 한다.

## `pb_intr.v`

`pb_intr`는 interrupt generator이자 LED controller이다.

| register/signal | 역할 |
|---|---|
| `curr_intr[3:0]` | button별 interrupt 상태 |
| `curr_led[7:0]` | LED 출력 값 |
| `pb_1d`, `pb_2d` | button 입력 2단 delay |
| `INTR` | interrupt output |
| `PRDATA` | APB read data |
| `PWDATA` | APB write data |

버튼을 눌렀다가 떼는 순간 `pb_2d[i] && !pb_1d[i]` 조건으로 해당 interrupt bit가 0이 된다. PS가 처리 후 `PWDATA[i]=1`을 write하면 interrupt 상태가 다시 1로 복원된다. LED register address에 write하면 `curr_led`가 갱신된다.

## C 코드 구조

| 함수 | 역할 |
|---|---|
| `main` | interrupt generator 초기화 후 GIC example 실행 |
| `ScuGicExample` | GIC config lookup, 초기화, self-test, handler 등록, interrupt enable |
| `SetUpInterruptSystem` | ARM exception handler 등록 및 interrupt enable |
| `DeviceDriverHandler` | interrupt 원인 read, LED value write, interrupt 처리 완료 |

## GIC 초기화 흐름

1. `XScuGic_LookupConfig(DeviceId)`로 GIC hardware config 조회
2. `XScuGic_CfgInitialize`로 GIC instance 초기화
3. `XScuGic_SelfTest`로 정상 동작 확인
4. `Xil_ExceptionRegisterHandler`로 ARM exception과 GIC handler 연결
5. `Xil_ExceptionEnable`로 processor interrupt enable
6. `XScuGic_Connect`로 interrupt ID와 `DeviceDriverHandler` 연결
7. `XScuGic_Enable`로 해당 interrupt ID enable

## 기본 Handler 동작

`DeviceDriverHandler`는 `Xil_In32LE(AXI2AHBLite)`로 interrupt status를 읽는다. 예를 들어 button 0이 눌려 bit 0이 0이면 다음 동작을 한다.

- `Xil_Out32LE(AXI2AHBLite, pb | 1)`로 interrupt bit 복원
- `Xil_Out32(AXI2AHBLite + 4, (1 << 7) | (1 << 6))`로 LED 두 개 on
- `InterruptProcessed = TRUE`

button별로 서로 다른 LED 2개가 켜지도록 mapping되어 있다.

## Quiz: LED 2개 번갈아 점등

목표는 특정 button이 눌리면 해당하는 LED 2개가 1초 간격으로 번갈아 켜지고, 다른 button이 눌리면 즉시 해당 동작으로 전환되는 것이다.

구현 방식:

- handler 내부에서 `while(1)`로 현재 button 동작 유지
- loop마다 interrupt status를 다시 read
- 다른 button bit가 0이면 `break`
- 두 개의 delay loop를 사용해 좌/우 LED를 번갈아 write
- `InterruptProcessed = TRUE`를 제거해 handler 흐름을 계속 유지

## 고찰

- `Xil_ExceptionEnable`은 ARM processor가 interrupt를 받을 수 있게 하는 전체 enable이다.
- `XScuGic_Enable`은 GIC에서 특정 interrupt ID를 enable하는 함수다.
- `sleep(1)`은 구현은 간단하지만 sleep 중 다른 button interrupt에 즉시 반응하기 어려워 quiz 요구사항에는 부적합하다.
- `ReadRTC` 기반 구현은 timing은 좋지만 `xil_printf()` 유무에 따라 동작이 바뀌는 문제가 있었다. system call delay가 register update timing에 영향을 준 것으로 추정된다.

## 정리

11주차 결과의 핵심은 interrupt 처리 전체가 HW와 SW의 협업이라는 점이다. PL은 event를 만들고 status를 제공하며, PS는 GIC와 handler로 event를 해석하고 다시 PL register를 써서 LED 동작을 만든다.
