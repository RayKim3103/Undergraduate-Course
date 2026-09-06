---
과목: Electric Circuits 2
유형: Lecture Note
주제: Inductor simulator, active filters, synthetic inductance
tags:
  - electric-circuits-2
  - filters
  - inductor-simulator
  - active-filter
---

# Filters with Inductor Simulator - 인덕터 시뮬레이터 필터

## 핵심 요약

IC에서 실제 inductor는 면적이 크고 특히 저주파용으로 구현하기 어렵다. 따라서 op-amp, resistor, capacitor를 이용해 inductor처럼 보이는 synthetic inductance를 만들 수 있다. 이 강의는 inductor-simulation circuit을 이용해 HP/BP/AP 등 2차 필터를 구현하는 방법을 설명한다.

## IC에서 Inductor가 어려운 이유

inductor 관계:

```text
v(t) = L di(t)/dt
V(s) = sL I(s)
```

저주파에서 큰 inductance가 필요하면 on-chip 면적이 매우 커진다. 그래서 active circuit으로 effective inductance를 만든다.

## Inductance-Simulation Circuit

자료의 핵심 결과:

```text
Leq = R1 R3 C4 R5 / R2
```

입력에서 본 impedance가:

```text
Zin ≈ s Leq
```

처럼 보이도록 op-amp 네트워크를 구성한다.

## Inductor를 대체한 2차 필터

수동 RLC 필터에서 `L`을 synthetic inductance로 대체하면 IC 친화적인 active filter가 된다.

HP filter 예:

```text
H_HP(s) = s^2 / [s^2 + (1/RC)s + 1/(LC)]
```

여기서 `L`에 `Leq`를 대입해 pole frequency와 Q를 조절한다.

## Band-Pass Filter

RLC band-pass도 synthetic inductor로 구현할 수 있다.

```text
H_BP(s) = (s/RC) / [s^2 + (1/RC)s + 1/(LC)]
```

center frequency:

```text
w0 = 1/sqrt(Leq C)
```

## All-Pass Filter 구현

2차 all-pass는 band-pass 출력 `T(s)`를 이용한 linear combination으로 만들 수 있다.

핵심 아이디어:

```text
Vout = 2 Vi T(s) - Vi
```

또는 op-amp summing 구조로 pole은 유지하고 zero를 mirror 위치에 배치한다.

## Grounded vs Floating Inductor

자료는 기본 inductor simulator가 grounded inductor임을 지적한다.

문제:

- 어떤 필터에서는 floating inductor가 필요하다.
- grounded inductor simulator만으로는 직접 대체가 안 될 수 있다.

이를 위해 generalized impedance converter(GIC)를 사용한다.

## Generalized Inductance-Simulation Circuit

GIC는 여러 impedance를 조합해 원하는 equivalent impedance를 만든다.

개념:

```text
Zin = product/ratio of several Z elements
```

적절히 resistor와 capacitor를 선택하면 floating 또는 grounded inductance-like impedance를 얻을 수 있다.

## 시험 포인트

- IC에서 physical inductor가 어려운 이유를 설명한다.
- `Leq = R1 R3 C4 R5 / R2` 형태의 synthetic inductance 개념을 이해한다.
- RLC filter에서 `L`을 synthetic inductor로 대체하면 active 2차 필터가 된다.
- grounded inductor와 floating inductor의 차이를 안다.
- GIC는 impedance 변환기로 이해한다.

## 같이 보면 좋은 노트

- [[17 Passive Second-Order Filters - 수동 2차 필터]]
- [[19 Filters with Integrators - KHN Tow-Thomas Biquad]]

