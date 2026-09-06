# CMOS 집적회로 개요

tags: #cmos-integrated-circuit #cmos #mosfet #layout #standard-cell #verification

관련 노트: [[02 Devices - MOS 소자 모델과 비이상성]]

## 핵심 요약

이 장은 CMOS 집적회로 설계의 전체 지도를 제공한다. MOS transistor의 switch 동작, complementary CMOS gate, pass transistor와 transmission gate, latch/flip-flop, fabrication/layout, HDL 기반 design flow, verification과 packaging까지 이어지는 큰 흐름을 다룬다.

## MOS Transistor 기본

MOS 구조는 gate, oxide, body가 만드는 capacitor로 이해할 수 있다. Gate 전압이 body와 channel 상태를 제어하고, source와 drain 사이에 전류가 흐를 수 있는 path를 만든다.

### nMOS

| Gate 전압 | 동작 |
|---|---|
| 낮음 | OFF, source-drain 경로가 끊김 |
| 높음 | ON, electron channel 형성 |

nMOS는 강한 0을 잘 전달하므로 pull-down network에 적합하다.

### pMOS

| Gate 전압 | 동작 |
|---|---|
| 낮음 | ON |
| 높음 | OFF |

pMOS는 강한 1을 잘 전달하므로 pull-up network에 적합하다.

## 전원 전압과 Scaling

VDD는 technology scaling과 함께 낮아졌다. 이유는 reliability와 low power 때문이다. 산화막이 얇아지고 device 크기가 줄어들수록 높은 전압은 breakdown, leakage, hot carrier 문제를 키우며, 동적 전력도 `VDD^2`에 비례해 증가한다.

## Complementary CMOS Logic

Complementary CMOS gate는 pMOS pull-up network와 nMOS pull-down network를 상보적으로 구성한다.

```text
입력이 어떤 조합이든 출력은 VDD 또는 GND 중 하나로 강하게 연결된다.
```

### Inverter

- nMOS: 입력이 1일 때 ON, 출력을 GND로 당김
- pMOS: 입력이 0일 때 ON, 출력을 VDD로 당김

### NAND

NAND에서 출력이 0이 되려면 모든 입력이 1이어야 한다. 따라서 nMOS는 series로 두고, pMOS는 parallel로 둔다.

### NOR

NOR에서 출력이 0이 되려면 입력 중 하나라도 1이면 된다. 따라서 nMOS는 parallel, pMOS는 series가 된다. pMOS series는 저항이 커서 NOR는 NAND보다 느린 경향이 있다.

## Compound Gate

Compound gate는 AOI, OAI처럼 여러 AND/OR 조합을 한 번에 inverted function으로 구현한다.

예:

```text
Y = NOT(AB + CD)
```

복잡한 논리를 여러 단순 gate로 나누는 대신 하나의 CMOS compound gate로 만들면 stage 수와 parasitic capacitance를 줄일 수 있다. 다만 transistor stacking과 input별 logical effort가 달라진다.

## Pass Transistor와 Transmission Gate

### Pass Transistor

- nMOS pass transistor는 strong 0을 잘 전달하지만 high level은 `VDD - Vt`로 약해진다.
- pMOS pass transistor는 strong 1을 잘 전달하지만 low level 전달이 약해진다.

### Transmission Gate

Transmission gate는 nMOS와 pMOS를 병렬로 묶고 보수 제어 신호를 사용한다. 0과 1을 모두 잘 전달하므로 mux, latch, tristate 구조에 많이 쓰인다.

## Tristate와 Mux

Transmission gate는 enable에 따라 출력 연결을 켜고 끄는 tristate buffer처럼 사용할 수 있다. 여러 tristate inverter 또는 transmission gate를 조합하면 mux를 만들 수 있다.

주의할 점은 입력 noise가 transmission gate를 통해 출력으로 그대로 전달될 수 있다는 것이다. 필요하면 restoring inverter를 뒤에 둔다.

## Latch와 Flip-Flop

D latch는 clock level에 따라 transparent 또는 opaque가 된다. D flip-flop은 edge-triggered 동작을 하며, 일반적으로 master-slave latch 두 개로 구성한다.

이 과목에서는 latch/flip-flop을 단순 논리 기호가 아니라 transistor-level storage element로 본다. 그래서 clocking, pass device, feedback path, setup/hold time이 이후 timing 분석과 연결된다.

## CMOS Fabrication

CMOS transistor는 silicon wafer 위에 mask 공정을 반복해 만든다. 보통 p-type substrate에 nMOS를 만들고, pMOS body를 위해 n-well을 형성한다.

주요 layer:

- diffusion 또는 active
- polysilicon gate
- contact
- metal interconnect
- via

Transistor와 wire는 mask set으로 정의된다. Layout은 회로 기능뿐 아니라 process design rule을 만족해야 한다.

## Layout과 Design Rules

Design rule은 fabrication 가능성을 보장하기 위한 최소 폭, 간격, 겹침 규칙이다. Feature size로 normalize해 공정이 바뀌어도 비슷한 layout 원리를 적용할 수 있다.

Standard cell layout 관례:

- pMOS는 위쪽, nMOS는 아래쪽에 배치
- vertical polysilicon gate 사용
- cell 높이를 맞춰 서로 붙여 배치 가능하게 설계
- 전원 rail은 보통 위쪽 VDD, 아래쪽 GND로 정렬

## Design Partitioning과 Design Flow

큰 SoC는 계층적으로 나누어 설계한다. 예를 들어 MIPS datapath는 control, register file, ALU, memory interface 같은 block으로 분해된다.

일반적인 흐름:

1. HDL로 기능 기술
2. Synthesis로 gate-level netlist 생성
3. Standard cell library를 사용해 place and route
4. Layout, parasitic extraction, timing/power 검증
5. Fabrication, packaging, test

## Verification의 중요성

Fabrication은 느리고 비싸다. 한 번 tape-out한 뒤 오류를 발견하면 시간과 비용 손실이 크다. 그래서 DRC, LVS, simulation, timing analysis, functional verification이 전체 chip 설계 effort의 큰 비중을 차지한다.

## 시험ㆍ복습 체크포인트

- nMOS와 pMOS가 각각 strong 0/strong 1을 전달하는 이유를 설명할 수 있어야 한다.
- NAND와 NOR의 pMOS/nMOS network 구성을 그릴 수 있어야 한다.
- Pass transistor와 transmission gate의 차이를 말할 수 있어야 한다.
- Standard cell layout의 pMOS/nMOS 배치 관례를 이해해야 한다.
- HDL부터 fabrication까지 CMOS design flow를 순서대로 설명할 수 있어야 한다.

