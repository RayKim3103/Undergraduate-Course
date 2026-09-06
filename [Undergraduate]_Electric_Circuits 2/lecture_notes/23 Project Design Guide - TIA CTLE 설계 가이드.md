---
과목: Electric Circuits 2
유형: Design Guide
주제: Optical receiver, TIA, CTLE, eye diagram, project specification
tags:
  - electric-circuits-2
  - project-guide
  - tia
  - ctle
  - eye-diagram
---

# Project Design Guide - TIA CTLE 설계 가이드

## 핵심 요약

이 자료는 설계 프로젝트의 전체 구조와 목표 스펙을 설명한다. 대상 회로는 optical receiver이며, photodiode가 optical signal을 current signal로 변환하고, TIA가 current를 voltage로 바꾸며, CTLE가 PD/TIA의 제한된 bandwidth를 보상한다. 성능은 frequency response, eye diagram, power consumption, FoM으로 평가한다.

## 전체 시스템

```text
Optical signal
-> Photodiode
-> Current signal
-> TIA
-> Voltage signal
-> CTLE
-> Output signal
```

역할:

- PD: optical-to-current conversion
- TIA: current-to-voltage conversion and amplification
- CTLE: bandwidth extension and ISI reduction

## Eye Diagram

eye diagram은 unit interval 단위로 time waveform을 겹쳐 표시한다.

주요 지표:

- eye height: vertical noise margin
- eye width: timing margin

자료 기준 예:

```text
eye height ≈ 0.7 Vo
0.15Vo, 0.85Vo 기준 cursor 사용
```

10Gbps data의 unit interval:

```text
UI = 1 / 10Gbps = 100 ps
```

## OTA 설계 목표

OTA는 TIA의 amplifier core로 사용된다.

이론적 gain 형태:

```text
Av ≈ gm1 gm6 (ro2 || ro4)(ro6 || ro7)
```

target:

- OTA open-loop gain > 15 dB
- OTA open-loop bandwidth > 6.5 GHz

중요:

- `IREF`와 input common-mode voltage를 조절해 모든 transistor가 saturation에 있도록 한다.
- gain과 bandwidth는 `gm`, `ro`, parasitic capacitance의 trade-off이다.

## TIA 설계

TIA는 shunt negative feedback 구조의 transimpedance amplifier이다.

입력/출력:

```text
input: current
output: voltage
```

target:

```text
Transimpedance gain > 60 dBOhm
```

피드백 저항 `RF`가 gain과 bandwidth에 모두 영향을 준다.

## CTLE 설계

CTLE는 PD+TIA의 high-frequency loss를 보상하기 위해 zero를 배치한다.

### 입력 High-Pass Filter

입력 bias를 만들고 DC를 차단한다.

```text
Vx/Vin = s(R1 || R2)Ci / [1 + s(R1 || R2)Ci]
```

DC bias:

```text
Vx(DC) = R2/(R1+R2) * VDD
```

target:

```text
cut-off frequency < 1 MHz
```

### CTLE Core

source degeneration capacitor를 이용한 high-frequency boosting:

```text
Vout/Vx = -gm RD (1 + s RS CS) /
          (1 + gm RS + s RS CS)
```

zero:

```text
wz = 1 / (RS CS)
```

pole:

```text
wp = (1 + gm RS) / (RS CS)
```

zero를 PD+TIA bandwidth roll-off 근처에 배치해 loss를 보상한다.

## Overall Peaking

overall peaking은 system gain의 maximum과 low-frequency gain 사이 차이이다.

target:

```text
overall peaking < 1.5 dB
```

peaking이 너무 크면 high-frequency noise가 과도하게 증폭되고 eye가 왜곡될 수 있다.

## LTspice Eye Diagram 설정

transient setting 예:

- stop time: 300 ns
- time to start saving: 150 ns
- maximum timestep: 1 ps

초기 0-150ns는 high-pass filter settling 때문에 버리고, 150-300ns 구간을 eye diagram에 사용한다.

SPICE directive:

```spice
.option baudrate={1/100p}
```

## Photodiode Simulation

PD symbol 파일을 user library directory에 넣고 사용한다.

simulation 모드:

- transient: `PWL FILE`에 `project_input.txt` 지정
- AC analysis: current input의 AC amplitude를 1로 설정

## Power Consumption 측정

transient simulation에서 VDD source로 들어가는 전류 평균을 사용한다.

예:

```spice
.meas TRAN Power_consumption 1*AVG I(V1)
```

SPICE output log에서 값을 확인한다.

## Target Specification

| 항목 | 목표 |
|---|---:|
| Data rate | 10 Gbps |
| OTA open-loop gain | > 15 dB |
| OTA open-loop bandwidth | > 6.5 GHz |
| Transimpedance gain | > 60 dBOhm |
| HPF cut-off | < 1 MHz |
| Overall bandwidth | > 4.5 GHz |
| Overall peaking | < 1.5 dB |
| Power consumption | < 5 mW |

FoM:

```text
FoM = eye height * eye width / power consumption
```

## 같이 보면 좋은 노트

- [[00 Design Project Summary - Optical Receiver PD TIA CTLE]]
- [[15 OTA and Op-Amp - OTA와 연산증폭기]]
- [[21 Feedback - 음귀환 기초]]
- [[22 LTSpice Tutorial - 시뮬레이션 튜토리얼]]

