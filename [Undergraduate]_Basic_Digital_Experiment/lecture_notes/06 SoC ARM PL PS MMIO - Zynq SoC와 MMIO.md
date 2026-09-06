# Zynq SoC, ARM, PL/PS, MMIO

tags: #basic-digital-experiment #soc #zynq #arm #mmio #axi-gpio #vitis

관련 노트: [[05 FSM Traffic Light - 유한상태머신 신호등]], [[07 Polling Interrupt Timer - 폴링 인터럽트 타이머]]

## 핵심 요약

이 자료는 Verilog로 PL만 다루던 이전 실험에서 확장해, Zynq SoC의 PS와 PL을 함께 사용하는 흐름을 다룬다. Vivado에서 Zynq Processing System과 AXI GPIO IP를 구성하고, Vitis에서 C 코드로 MMIO 주소에 접근해 LED와 RGB LED를 제어한다.

## SoC

SoC(System on Chip)는 CPU, memory controller, peripheral, bus, custom logic 등을 하나의 칩 안에 통합한 구조이다.

### 장점

- 칩 간 연결보다 지연이 작다.
- 전력 소모를 줄이기 쉽다.
- 보드 면적과 비용을 줄일 수 있다.
- embedded system에 적합하다.

### 단점

- 설계와 검증이 복잡하다.
- 초기 개발 비용이 크다.
- 높은 집적도로 인해 발열 관리가 중요하다.

## ARM

ARM은 RISC 철학을 기반으로 한 processor architecture이다. 단순하고 효율적인 명령어 구조, 낮은 전력 소모, embedded 환경에서의 강점 때문에 모바일ㆍ임베디드ㆍSoC에 널리 사용된다. Zynq 보드에서는 ARM processor가 PS 영역의 hard core로 포함된다.

## PL과 PS

| 구분 | 의미 | 특징 |
|---|---|---|
| PL | Programmable Logic | FPGA fabric, 사용자가 Verilog/VHDL로 원하는 하드웨어 구성 |
| PS | Processing System | ARM core, memory controller, peripheral, interface 등 고정된 처리 시스템 |

PL은 자유롭게 회로를 만들 수 있는 영역이고, PS는 이미 만들어진 processor system을 설정해서 사용하는 영역이다. 두 영역은 AXI 같은 bus를 통해 통신한다.

## IP

IP(Intellectual Property)는 미리 설계된 하드웨어 블록이다.

| 종류 | 설명 |
|---|---|
| Soft IP | HDL 코드 형태, FPGA logic으로 합성 |
| Hard IP | 칩 내부에 물리적으로 고정 구현 |
| Firm IP | soft와 hard의 중간 형태 |

IP를 사용하면 UART, GPIO, FIFO, processor system 같은 기능을 직접 처음부터 구현하지 않고 재사용할 수 있다.

## MMIO

MMIO(Memory-Mapped I/O)는 peripheral register를 memory address 공간에 배치하는 방식이다. CPU는 일반 memory를 읽고 쓰듯 특정 주소를 읽고 써서 장치를 제어한다.

```c
volatile unsigned int *led = (volatile unsigned int *)0x41200000;
led[0] = 10;
```

`volatile`은 compiler가 이 메모리 접근을 최적화로 제거하거나 순서를 바꾸지 못하게 한다. hardware register는 일반 변수와 달리 외부 장치 상태와 연결되므로 `volatile`이 중요하다.

## Experiment 1: Hello World

### Vivado 흐름

1. Zynq Processing System을 block design에 추가한다.
2. board preset과 DDR 설정을 적용한다.
3. block automation으로 PS 구성을 자동 연결한다.
4. HDL wrapper를 만든다.
5. bitstream을 생성한다.
6. hardware를 `.xsa`로 export한다.

### Vitis 흐름

1. exported hardware platform을 기반으로 application project를 만든다.
2. `init_platform()`으로 board runtime을 초기화한다.
3. `print()` 또는 `xil_printf()`로 serial console에 출력한다.
4. `cleanup_platform()`으로 종료한다.

Hello World 실험은 PS가 정상적으로 동작하고 serial console과 연결되는지 확인하는 가장 기본적인 단계이다.

## Experiment 2: AXI GPIO로 LED 제어

### Vivado 구성

- AXI GPIO IP를 추가한다.
- GPIO width를 4-bit LED에 맞춘다.
- Address editor에서 base address를 확인한다.
- PS의 AXI master가 AXI GPIO slave register에 접근하도록 연결한다.

### C 코드 핵심

```c
volatile unsigned int *led = (volatile unsigned int *)0x41200000;
led[0] = 2 + 8;
```

`2 + 8 = 10 = 4'b1010`이므로 LED bit 1과 bit 3이 켜진다.

## Experiment 3: AXI GPIO로 RGB LED 제어

RGB LED는 여러 channel을 하나의 GPIO vector로 묶어 제어한다.

```c
volatile unsigned int *rgb_led = (volatile unsigned int *)0x41200000;
rgb_led[0] = 1 + 4 + 16;
```

값 `21`은 6-bit로 `010101`에 해당하며, 연결된 RGB channel mapping에 따라 두 RGB LED의 색이 결정된다. 다른 예로 `1 + 32 = 33`을 쓰면 특정 red/blue channel만 켜는 식으로 색 조합을 만들 수 있다.

## 이전 Verilog 실험과의 차이

이전 주차에서는 PL 내부 회로를 Verilog로 직접 만들고 스위치/LED에 연결했다. 이번 주차에서는 Vivado block design에서 IP를 배치하고, Vitis C 코드로 PS가 AXI bus를 통해 PL의 GPIO register를 제어한다.

즉, 하드웨어 동작 일부는 PL IP가 담당하고, 동작 순서나 값 결정은 C 프로그램이 담당한다.

## 문제 해결 포인트

- serial port가 다른 Vitis workspace나 terminal에 잡혀 있으면 console 출력이 보이지 않을 수 있다.
- address editor의 base address와 C 코드의 pointer 주소가 다르면 peripheral이 동작하지 않는다.
- AXI GPIO width와 실제 LED/RGB LED bit 수가 맞아야 한다.
- `volatile`을 빼면 compiler 최적화 때문에 hardware register write가 의도와 다르게 처리될 수 있다.

## 시험ㆍ복습 체크포인트

- SoC, ARM, PL, PS, IP의 의미를 구분할 수 있어야 한다.
- MMIO가 왜 pointer 접근으로 peripheral 제어를 가능하게 하는지 설명할 수 있어야 한다.
- Vivado block design과 Vitis software project의 역할 차이를 말할 수 있어야 한다.
- AXI GPIO base address와 LED bit mapping을 연결해 해석할 수 있어야 한다.

