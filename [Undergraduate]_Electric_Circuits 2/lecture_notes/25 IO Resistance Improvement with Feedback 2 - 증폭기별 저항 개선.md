---
과목: Electric Circuits 2
유형: Lecture Note
주제: Feedback topologies, voltage/current/transconductance/transimpedance amplifier resistance
tags:
  - electric-circuits-2
  - feedback
  - amplifier-types
  - resistance-improvement
---

# IO Resistance Improvement with Feedback 2 - 증폭기별 저항 개선

## 핵심 요약

feedback이 input/output resistance에 주는 효과는 amplifier type과 sampling/mixing 방식에 따라 달라진다. 전압 증폭기, 전류 증폭기, transconductance amplifier, transimpedance amplifier는 각각 이상적인 input/output resistance가 다르므로, negative feedback은 그 이상 조건에 가까워지도록 저항을 증가 또는 감소시킨다.

## 증폭기별 이상 저항

| 유형 | 입력 | 출력 | 이상적 Rin | 이상적 Rout |
|---|---|---|---:|---:|
| Voltage amplifier | V | V | infinite | 0 |
| Current amplifier | I | I | 0 | infinite |
| Transconductance amplifier | V | I | infinite | infinite |
| Transimpedance amplifier | I | V | 0 | 0 |

## Voltage Amplifier

Voltage-voltage feedback:

```text
Av,closed = Av,open / (1 + KAv)
Rin,closed = Rin,open (1 + KAv)
Rout,closed = Rout,open / (1 + KAv)
```

전압 증폭기에 적합한 방향:

- input resistance 증가
- output resistance 감소

## Transconductance Amplifier

Current-voltage feedback:

```text
Gm,closed = Gm,open / (1 + K Gm,open)
Rin,closed = Rin,open (1 + K Gm,open)
Rout,closed = Rout,open (1 + K Gm,open)
```

transconductance amplifier의 이상 조건은:

- input voltage를 잘 받기 위해 `Rin` 큼
- output current source처럼 보이기 위해 `Rout` 큼

## Current Amplifier

Current-current feedback:

```text
Ai,closed = Ai,open / (1 + K Ai,open)
Rin,closed = Rin,open / (1 + K Ai,open)
Rout,closed = Rout,open (1 + K Ai,open)
```

current amplifier의 이상 조건:

- input resistance 작음
- output resistance 큼

## Transimpedance Amplifier

Voltage-current feedback:

```text
Rm,closed = Rm,open / (1 + K Rm,open)
Rin,closed = Rin,open / (1 + K Rm,open)
Rout,closed = Rout,open / (1 + K Rm,open)
```

transimpedance amplifier의 이상 조건:

- input current를 받기 위해 `Rin` 작음
- output voltage source처럼 보이기 위해 `Rout` 작음

## 회로 예 - Transconductance Feedback

differential amplifier + CS 구조에서 output current를 feedback voltage로 변환해 입력에 되돌린다.

효과:

- transconductance gain 감소
- output resistance 증가
- voltage-to-current amplifier에 더 적합한 특성

## 회로 예 - Current Amplifier

CG + CS 구조에서 output current 일부를 current로 feedback한다.

효과:

- current gain 감소
- input resistance 감소
- output resistance 증가

## 회로 예 - TIA

CG + CS와 큰 feedback resistor `RF`를 사용한 transimpedance amplifier에서는:

```text
Zt,open ≈ -gm RD RF
Zt,closed ≈ -RF   (large loop gain)
```

즉 feedback이 충분히 크면 transimpedance gain이 `RF`에 의해 안정적으로 결정된다.

## 시험 포인트

- feedback topology별로 `Rin`, `Rout`이 증가/감소하는 방향을 외우기보다 이상적인 증폭기 조건과 연결해 이해한다.
- voltage amplifier: `Rin up`, `Rout down`
- current amplifier: `Rin down`, `Rout up`
- transconductance amplifier: `Rin up`, `Rout up`
- transimpedance amplifier: `Rin down`, `Rout down`
- gain은 항상 loop gain만큼 감소한다.

## 같이 보면 좋은 노트

- [[21 Feedback - 음귀환 기초]]
- [[24 IO Resistance Improvement with Feedback 1 - 전압 증폭기 피드백]]
- [[23 Project Design Guide - TIA CTLE 설계 가이드]]

