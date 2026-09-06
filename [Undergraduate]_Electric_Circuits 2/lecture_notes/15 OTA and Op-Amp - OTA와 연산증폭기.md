---
과목: Electric Circuits 2
유형: Lecture Note
주제: OTA, op-amp, ICMR, transconductance gain, negative feedback
tags:
  - electric-circuits-2
  - ota
  - op-amp
  - feedback
---

# OTA and Op-Amp - OTA와 연산증폭기

## 핵심 요약

OTA는 differential input을 output current로 바꾸는 operational transconductance amplifier이다. 전류미러와 single-ended differential amplifier, CS load를 조합해 만들 수 있다. voltage amplifier가 필요하면 source follower buffer를 붙여 op-amp 구조로 확장한다. op-amp는 보통 negative feedback과 함께 사용해 gain을 resistor ratio로 안정화한다.

## OTA의 구성

기본 요소:

- current mirrors
- single-ended differential amplifier
- CS stage with PMOS load

증폭기 유형:

```text
input: voltage
output: current
-> transconductance amplifier
```

## Input Common-Mode Range

OTA 입력단 MOSFET들이 saturation을 유지해야 한다.

하한 조건:

```text
VCM,min = VGS3 - |VTH,p|
```

상한 조건:

```text
VCM,max = VDD - VSG5 - VSG1 + |VTH,p|
```

큰 ICMR을 얻으려면 bias current를 작게 하여 필요한 overdrive voltage를 줄이는 것이 유리하다. 하지만 current를 너무 줄이면 `gm`과 bandwidth가 줄어든다.

## Transconductance Gain

OTA는 differential pair와 gain stage가 직렬로 연결된 구조로 볼 수 있다.

자료의 핵심 형태:

```text
Gm,total ≈ -gm1 gm6 (ro2 || ro4)
Rout ≈ ro6 || ro7
```

voltage gain:

```text
Av ≈ -gm1 gm6 (ro2 || ro4)(ro6 || ro7)
```

## Frequency Response

각 stage의 dominant pole은 주로 Miller capacitance에 의해 결정된다.

1단:

```text
C1 ≈ gm2(ro2 || ro4) Cgd2
```

2단:

```text
C2 ≈ gm6(ro6 || ro7) Cgd6
```

두 번째 stage output resistance가 source resistance보다 훨씬 크면 두 번째 pole이 전체 dominant pole이 될 수 있다.

## OTA에서 Op-Amp로

OTA는 output resistance가 크다. voltage amplifier로 쓰려면 output buffer가 필요하다.

```text
OTA + source follower -> op-amp
```

ideal op-amp:

- `Rin = infinite`
- `Rout = 0`
- open-loop gain very large

## Negative Feedback Op-Amp

op-amp는 거의 항상 negative feedback과 함께 사용한다.

non-inverting amplifier에서 ideal op-amp 가정:

```text
V+ = V-
Vo/Vs = 1 + R2/R1
```

inverting amplifier:

```text
Vo/Vs = - RF/Rin
```

장점:

- gain이 transistor parameter가 아니라 resistor ratio로 결정된다.
- gain을 쉽게 바꿀 수 있다.
- 안정성이 좋아진다.

## 시험 포인트

- OTA는 transconductance amplifier이다.
- ICMR는 input pair와 current source/load의 saturation 조건으로 정한다.
- OTA voltage gain은 `Gm * Rout` 형태이다.
- op-amp는 OTA에 voltage buffer를 붙인 구조로 이해한다.
- negative feedback에서 `V+ ≈ V-`가 성립하는 이유를 loop gain으로 설명한다.

## 같이 보면 좋은 노트

- [[08 Differential Amplifiers 1 - 차동 증폭기 기본]]
- [[09 Differential Amplifiers 2 - CMRR와 Active Load]]
- [[21 Feedback - 음귀환 기초]]

