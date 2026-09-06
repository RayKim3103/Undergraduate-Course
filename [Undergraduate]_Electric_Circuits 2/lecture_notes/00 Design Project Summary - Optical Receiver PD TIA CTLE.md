---
과목: Electric Circuits 2
유형: Design Project Summary
주제: Optical receiver, PD, TIA, CTLE, OTA, eye diagram
tags:
  - electric-circuits-2
  - design-project
  - optical-receiver
  - tia
  - ctle
  - ota
---

# Design Project Summary - Optical Receiver PD TIA CTLE

## 핵심 요약

이 자료는 광수신기 `Photodiode(PD) + Transimpedance Amplifier(TIA) + Continuous-Time Linear Equalizer(CTLE)` 설계 결과를 정리한 프로젝트 보고서이다. 핵심 목표는 PD에서 나온 전류 신호를 TIA로 전압 신호로 변환/증폭하고, CTLE로 고주파 손실을 보상하여 10Gbps급 eye opening을 확보하는 것이다.

## 최종 설계 수치

| 항목 | 값 |
|---|---:|
| OTA gain at 10MHz | 22.99 dB |
| OTA 3-dB bandwidth | 6.99 GHz |
| High-pass filter cut-off | 788.6 kHz |
| TIA gain at 10MHz with CTLE load | 62 dB |
| System 3-dB bandwidth | 4.501 GHz |
| Overall peaking | 0.381 dB |
| Power consumption | 1.285 mW |
| Eye height | 226.32 mV |
| Eye width | 88.54 ps |
| FoM | `1.56e-8` |

## 회로 파라미터

| 파라미터 | 값 |
|---|---:|
| `RF` | 1.6 kOhm |
| `CIN` | 40 pF |
| `R1`, `R2` | 10 kOhm |
| `RS` | 1.2 kOhm |
| `CS` | 0.08 pF |
| `RD` | 2.29 kOhm |
| `IREF` | 300 uA |
| `VREF` | 0.6 V |

MOS 크기는 45nm/180nm 길이를 섞어 사용하며, headroom과 bandwidth를 동시에 맞추기 위해 width를 조정했다.

## OTA 검증

### 포화 영역 확인

모든 MOSFET에 대해 다음 조건을 확인했다.

- NMOS: `VGS - VTH > 0`, `VDS > VGS - VTH`
- PMOS: `VSG - |VTH| > 0`, `VSD > VSG - |VTH|`

보고서에서는 `VTH ≈ 0.3V` 가정을 사용했고, 실제 threshold 측정 결과 PMOS는 대략 `0.23V ~ 0.25V` 수준으로 확인되었다. 따라서 M1-M8이 모두 saturation region에서 동작한다고 판단했다.

### OTA 주파수 응답

OTA 단독 AC simulation 결과:

- DC gain: `22.99 dB`
- 3-dB bandwidth: `6.99 GHz`

설계 관점:

- `gm`이 커지면 gain과 bandwidth에 유리할 수 있다.
- `ro = 1/(lambda ID)`이므로 bias current가 커지면 `ro`가 작아져 gain이 감소할 수 있다.
- 따라서 `IREF`, MOS width, headroom의 균형이 중요하다.

## TIA 설계

PD는 고주파에서 내부 capacitance 때문에 전류 일부가 capacitor로 빠지며 bandwidth 제한을 만든다.

PD 단독 응답:

- `IOUT/IIN` at 10MHz: 0 dB
- `IOUT/IIN` at 8GHz: -3 dB

TIA의 기본 목적:

```text
input current -> output voltage
Z_T(s) = Vout(s) / Iin(s)
```

피드백 저항 `RF`는 transimpedance gain을 결정하지만 너무 크게 잡으면 출력단 MOSFET이 triode로 밀려 gain이 감소할 수 있다. 그래서 출력단 포화 조건을 유지하면서 gain을 키우는 값이 필요하다.

## TIA 전달함수 핵심

open-loop gain을 `A0`, PD capacitance를 `CPD`, feedback resistor를 `RF`라고 하면 이상적인 근사에서:

```text
Vout/Iin = - A0 RF / (s CPD RF + 1 + A0)
```

open-loop bandwidth가 finite이면:

```text
A(s) = A0 / (1 + s/wp)
```

이를 대입하면 feedback에 의해 bandwidth가 증가하고 gain은 감소한다. 즉 gain-bandwidth product trade-off가 그대로 나타난다.

## CTLE 설계

CTLE는 PD+TIA의 고주파 gain 저하를 zero로 보상한다.

CTLE 구조:

- 입력 bias용 high-pass filter
- source degeneration에 capacitor를 붙인 고주파 boost stage

보고서에서 추정한 pole/zero:

| 요소 | 위치 |
|---|---:|
| HPF zero | 0 Hz |
| HPF pole | 788 kHz |
| CTLE zero | 약 1.23 GHz |
| CTLE pole 1 | 약 3 GHz |
| CTLE pole 2 | 약 18.7 GHz |

PD+TIA만의 3-dB bandwidth가 약 2.28GHz라서, CTLE zero를 이 근처에 배치해 고주파 손실을 보상했다.

## Eye Diagram 결과

최종 시스템 `PD + TIA + CTLE` 결과:

- eye height: `226.32 mV`
- eye width: `88.54 ps`
- overall bandwidth: `4.502 GHz`
- overall peaking: 약 `0.4 dB`

eye height는 voltage gain과 관련이 크고, eye width는 bandwidth와 ISI에 민감하다.

## 설계 Discussion

### OTA

gain을 높이려면 `gm`과 `ro`가 중요하다.

- width 증가 -> `gm` 증가
- current 증가 -> bandwidth 개선 가능
- current 증가 -> `ro` 감소로 gain 감소 가능

결국 시뮬레이션으로 `gain > 15 dB`, `bandwidth > 6.5 GHz` 조건을 동시에 만족시키도록 조정했다.

### TIA

조정 가능한 핵심 값은 `Vref`, `Rf`이다.

- `Rf` 증가 -> transimpedance gain 증가
- `Rf` 과대 -> output headroom 부족, MOS triode 진입 가능
- `Vref` 부적절 -> MOSFET saturation 실패

### CTLE

CTLE는 작은 capacitance와 GHz 대역 parasitic 영향 때문에 이론식과 simulation 차이가 크다. 따라서 pole/zero 공식으로 방향을 잡고 parameter sweep으로 조정하는 접근이 필요하다.

## 같이 보면 좋은 노트

- [[15 OTA and Op-Amp - OTA와 연산증폭기]]
- [[21 Feedback - 음귀환 기초]]
- [[23 Project Design Guide - TIA CTLE 설계 가이드]]
- [[24 IO Resistance Improvement with Feedback 1 - 전압 증폭기 피드백]]

