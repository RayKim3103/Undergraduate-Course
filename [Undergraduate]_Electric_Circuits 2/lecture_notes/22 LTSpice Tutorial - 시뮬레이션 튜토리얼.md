---
과목: Electric Circuits 2
유형: Tutorial
주제: LTspice installation, schematic, NCSU 45nm CMOS, DC/AC/transient simulation
tags:
  - electric-circuits-2
  - ltspice
  - simulation
  - cmos
---

# LTSpice Tutorial - 시뮬레이션 튜토리얼

## 핵심 요약

이 자료는 LTspice 설치부터 NCSU 45nm CMOS model 설정, inverter schematic 작성, DC operating point, DC sweep, parametric sweep, AC analysis, transient simulation까지 실습 절차를 설명한다. 마지막에는 CS amplifier homework를 통해 `VTH`, `gm`, `ro`, gain, 3-dB bandwidth를 simulation으로 측정한다.

## LTspice란

LTspice는 Analog Devices가 제공하는 무료 circuit simulation program이다. 전력회로, analog 회로, power system 설계에 널리 사용되며, schematic 기반으로 다양한 SPICE simulation을 수행할 수 있다.

## 주요 단축키

| 기능 | 단축키 |
|---|---|
| Configure analysis | A |
| Run/Pause | Alt+R |
| Stop | Alt+S |
| Zoom to fit | Space |
| Component | P |
| Wire | W |
| Ground | G |
| Voltage source | V |
| Resistor | R |
| Capacitor | C |
| Inductor | L |
| Net name | N |
| SPICE directive | . |

## LTspice 단위 표기

| 표기 | 값 |
|---|---:|
| k | `1e3` |
| MEG | `1e6` |
| G | `1e9` |
| m | `1e-3` |
| u | `1e-6` |
| n | `1e-9` |
| p | `1e-12` |
| f | `1e-15` |

주의: LTspice에서 `M`은 mega가 아니라 milli로 해석될 수 있으므로 `MEG`를 쓴다.

## NCSU 45nm CMOS Model 설정

이 과목에서는 NCSU 45nm CMOS model을 사용한다.

절차:

1. `models_nom` 폴더 준비
2. LTspice setting의 user libraries directory에 추가
3. schematic에 `.inc` directive로 model file 포함
4. MOSFET instance model name을 `NMOS_VTL`, `PMOS_VTL` 등으로 설정
5. length/width 지정

## Inverter Design 예

구성:

- `nmos4`
- `pmos4`
- VDD voltage source
- input voltage source
- ground
- IN/OUT net label
- PMOS body는 VDD에 연결
- NMOS body는 ground에 연결

PMOS symbol에서 drain/source 표시가 기대와 다를 수 있으므로 simulation 결과의 terminal naming에 주의한다.

## DC Operating Point Simulation

목적:

- 각 node voltage 확인
- device current 확인
- MOSFET operating region 확인

특징:

- capacitor는 open으로 처리
- inductor는 short으로 처리

사용 예:

- bias voltage 확인
- saturation 조건 확인
- current mirror 동작 확인

## DC Sweep Simulation

목적:

- DC source 값을 바꾸며 voltage/current curve 관찰

예:

- inverter VTC
- `Id` vs `Vgs`
- threshold voltage 추정

## Parametric Simulation

특정 parameter를 sweep한다.

예:

```spice
.step param width_nmos 0.5u 1u 0.1u
```

component 값에는 `{width_nmos}`처럼 parameter 이름을 넣는다.

활용:

- transistor width 변화에 따른 gain 비교
- resistor/capacitor tuning
- design trade-off 탐색

## AC Analysis

목적:

- frequency response 확인
- magnitude/phase Bode plot
- DC gain
- 3-dB bandwidth

주의:

- AC analysis는 time-domain simulation이 아니다.
- 입력 source의 AC amplitude를 보통 1로 설정한다.

결과 분석:

- magnitude dB plot
- phase plot 제거 가능
- cursor로 gain과 3-dB bandwidth 측정

## Transient Simulation

목적:

- time-domain response 확인
- sine/pulse/PWL 입력에 대한 출력 파형 확인
- eye diagram 생성

simulation time이 너무 길면 파형이 조밀해 보이므로 x-axis range를 조정한다.

## Homework 핵심

CS amplifier 조건:

- `VDD = 1.2 V`
- `VSS = 0 V`
- `RD = 1.4 kOhm`
- `M1 length = 180 nm`
- `M1 width = 4.5 um`
- `CL = 100 fF`
- `Vin DC offset = 0.6 V`
- amplitude `0.05 V`
- frequency `300 MHz`

해야 할 일:

1. `Id-Vgs` curve로 `VTH` 결정
2. DC sweep에서 `gm` plot
3. `VGS = 0.6 V`에서 `ro` 결정
4. transient simulation과 AC analysis 실행
5. DC gain과 3-dB bandwidth 측정

## 시험/실습 포인트

- `.op`, `.dc`, `.ac`, `.tran`, `.step`의 용도를 구분한다.
- AC amplitude를 1로 두면 transfer function을 바로 읽기 쉽다.
- `D(Id(M1))` 같은 derivative expression으로 `gm`을 얻을 수 있다.
- `ro`는 `Id-Vds` curve의 기울기 역수로 구한다.
- CMOS body connection을 반드시 확인한다.

## 같이 보면 좋은 노트

- [[03 MOS Small-Signal Characteristics - MOS 소신호 모델]]
- [[04 Common-Source Amplifier - CS 증폭기]]
- [[23 Project Design Guide - TIA CTLE 설계 가이드]]

