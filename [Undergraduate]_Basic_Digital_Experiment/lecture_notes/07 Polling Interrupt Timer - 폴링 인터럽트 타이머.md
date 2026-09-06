# Polling, Interrupt, Timer

tags: #basic-digital-experiment #io #polling #interrupt #timer #zynq #vitis

관련 노트: [[06 SoC ARM PL PS MMIO - Zynq SoC와 MMIO]], [[09 AMBA AHB APB Memory Transfer - AMBA AHB APB 메모리전송]]

## 핵심 요약

이 자료는 processor가 외부 입력 장치와 상호작용하는 대표 방식인 polling과 interrupt를 비교하고, PYNQ/Zynq 환경에서 AXI GPIO button 입력을 C 코드로 처리하는 실험을 다룬다. polling은 단순하지만 CPU가 계속 확인해야 하고, interrupt는 이벤트가 생겼을 때만 handler가 실행되어 효율적이다.

## I/O와 Controller

I/O 장치는 저장 장치, 통신 장치, 사용자 interface 장치 등으로 나눌 수 있다. Processor는 장치와 직접 모든 신호를 주고받기보다 controller를 통해 통신한다.

Controller의 역할은 다음과 같다.

- 장치의 serial bitstream이나 물리 신호를 processor가 읽을 수 있는 register 값으로 바꾼다.
- processor가 write한 register 값을 장치 제어 신호로 변환한다.
- bus protocol과 device protocol 사이를 중재한다.

## Polling

Polling은 CPU가 일정 주기마다 장치 상태 register를 읽어 이벤트가 발생했는지 확인하는 방식이다.

```text
while (1) {
    read device status
    if event happened:
        handle event
}
```

### 장점

- 구조가 단순하다.
- 실행 흐름이 예측 가능하다.
- interrupt controller 설정이 필요 없다.

### 단점

- 이벤트가 없어도 CPU time을 계속 사용한다.
- polling 주기가 길면 입력 반응이 늦다.
- 버튼을 매우 짧게 누르면 polling 사이에 지나가서 놓칠 수 있다.

## Interrupt

Interrupt는 장치가 이벤트 발생을 CPU에 알려 현재 실행 중인 프로그램을 잠시 멈추고 handler를 실행하게 하는 방식이다.

### 동작 순서

1. 외부 장치가 interrupt request를 발생시킨다.
2. CPU는 현재 실행 상태와 return address를 저장한다.
3. interrupt vector를 통해 해당 handler로 이동한다.
4. handler가 원인을 처리하고 interrupt 상태를 clear한다.
5. 저장한 상태를 복원하고 원래 프로그램으로 돌아간다.

### 종류

| 종류 | 설명 |
|---|---|
| Hardware interrupt | 외부 장치에서 비동기적으로 발생 |
| Software interrupt / Exception | 명령 실행 중 내부 조건으로 발생 |

## PYNQ/Zynq Interrupt 구조

PYNQ에서는 MIO, GPIO, IP block 등이 interrupt를 만들 수 있다. PL 쪽 IP가 만든 interrupt는 PS의 GIC(Generic Interrupt Controller)로 전달되고, GIC가 우선순위와 활성화 상태를 관리한 뒤 CPU handler를 호출한다.

주요 라이브러리는 다음과 같다.

| 헤더 | 역할 |
|---|---|
| `xscugic.h` | Zynq interrupt controller 설정 |
| `xgpio.h` | AXI GPIO 초기화와 읽기 |
| `xil_exception.h` | exception/interrupt framework 연결 |
| `xparameters.h` | hardware platform의 device ID와 interrupt ID |

## Experiment 1: Button Polling

### Vivado 구성

- AXI GPIO IP를 button 입력 4-bit로 설정한다.
- PS가 AXI bus를 통해 GPIO register를 읽을 수 있도록 연결한다.

### C 코드 동작

- `while(1)` 루프에서 button register를 반복해서 읽는다.
- `sleep(1)`을 사용해 1초 간격으로 polling한다.
- 버튼 bit pattern에 따라 버튼 번호 값을 누적하거나 출력한다.
- 네 번째 버튼을 누르면 loop를 종료한다.

### 결과 해석

버튼 1, 2, 3을 누르면 각각의 값이 반영되고, 조합 입력에서는 합산된 값이 출력된다. 다만 1초 주기 polling이므로 버튼을 짧게 누르면 읽지 못할 수 있다.

## Experiment 2: Button Interrupt

### Vivado 구성

- AXI GPIO의 interrupt 기능을 활성화한다.
- GPIO의 `ip2intc_irpt`를 Zynq PS의 `IRQ_F2P` 입력에 연결한다.
- block automation과 address 설정을 완료한 뒤 bitstream과 hardware platform을 export한다.

### C 코드 구성 요소

| 요소 | 역할 |
|---|---|
| `XGpio BTNInst` | button GPIO instance |
| `XScuGic INTCInst` | interrupt controller instance |
| `btn_value` | 현재 읽은 button 값 |
| `value` | handler가 main loop에 전달할 값 |
| `flag` | 종료 또는 상태 변경 표시 |

Handler는 button 값을 읽고, interrupt pending 상태를 clear한 뒤, 다시 interrupt를 enable한다. Interrupt handler 안에서는 너무 오래 걸리는 작업을 피하고, main loop가 처리할 최소 정보만 저장하는 것이 좋다.

## Experiment 3: Timer 동작

Timer 실험은 interrupt 기반 button 처리에 countdown 동작을 추가한다.

### 버튼 기능

| 버튼 | 기능 |
|---|---|
| BTN1 | count 증가 |
| BTN2 | count 감소, 0 아래로 내려가지 않음 |
| BTN3 | count reset |
| BTN4 | countdown 시작 |

시작 버튼을 누르면 `Start`를 출력하고, 1초마다 count를 감소시킨 뒤 0이 되면 `End`를 출력한다.

## Polling과 Interrupt 비교

| 항목 | Polling | Interrupt |
|---|---|---|
| CPU 사용 | 계속 확인하므로 낭비 가능 | 이벤트 때만 처리 |
| 구현 난이도 | 낮음 | controller 설정 필요 |
| 반응성 | polling period에 의존 | 빠름 |
| 놓치는 입력 | 짧은 입력을 놓칠 수 있음 | edge/level 설정에 따라 안정적 |

## 추가 개념: DMA

I/O 방식에는 polling과 interrupt 외에도 DMA(Direct Memory Access)가 있다. DMA는 CPU가 매번 데이터를 옮기지 않고, DMA controller가 memory와 peripheral 사이의 대량 데이터를 직접 전송하게 한다. 오디오나 영상처럼 데이터량이 큰 응용에서 유리하다.

## 시험ㆍ복습 체크포인트

- Polling과 interrupt의 장단점을 비교할 수 있어야 한다.
- Interrupt handler에서 interrupt clear와 re-enable이 필요한 이유를 설명할 수 있어야 한다.
- GIC, AXI GPIO, IRQ_F2P 연결 관계를 이해해야 한다.
- 짧은 버튼 입력이 polling에서 누락될 수 있는 이유를 말할 수 있어야 한다.

