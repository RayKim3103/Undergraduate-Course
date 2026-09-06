---
과목: Electric Circuits 2
유형: Lecture Note
주제: Ring oscillator, Barkhausen condition, CMOS inverter oscillator, VCO
tags:
  - electric-circuits-2
  - oscillator
  - ring-oscillator
  - barkhausen
  - vco
---

# Ring Oscillators - 링 발진기

## 핵심 요약

발진기는 입력 없이 주기적인 출력을 만드는 회로이다. feedback loop가 특정 주파수에서 한 바퀴 돌아온 신호의 magnitude를 1 이상, phase를 0도 또는 360도로 만들면 oscillation이 가능하다. Ring oscillator는 odd number inverter 또는 amplifier stage를 loop로 연결해 지연과 phase shift로 발진을 만든다.

## Oscillator 정의

oscillator:

```text
input 없이 periodic output을 만드는 회로
```

MOS 회로에서는 amplifier와 feedback network를 이용한다.

## Barkhausen 조건

loop transfer가 `H(jw)`일 때 발진 조건:

```text
|H(jw_osc)| = 1
angle H(jw_osc) = 0 deg 또는 360 deg
```

강의에서는 한 바퀴 돌아온 신호가 in-phase이고 같은 magnitude가 되어야 한다고 설명한다.

실제로는 startup을 위해 처음에는 loop gain이 1보다 약간 커야 하고, large-signal nonlinearity가 amplitude를 제한한다.

## 단일 CS Stage

CS stage 하나:

```text
H(s) = -gm (RD || 1/(sCD))
```

하나의 CS만으로는 필요한 phase/magnitude 조건을 동시에 만족시키기 어렵다.

## 두 개의 CS Stage

두 개의 CS는 DC에서 phase가 360도에 가깝지만 pole phase shift와 magnitude 조건 때문에 안정적인 oscillation 조건을 만들기 어렵다.

## 세 개의 CS Stage

세 개의 CS stage는 각 stage가 60도 정도의 추가 phase shift를 제공해 전체 조건을 만족할 수 있다.

각 stage:

```text
H_stage(jw) = -gm RD / (1 + jw RD CD)
```

3-stage ring에서:

```text
3 tan^-1(w RD CD) = pi
```

oscillation frequency:

```text
w_osc ≈ sqrt(3) / (RD CD)
```

loop gain 조건은 이 주파수에서 magnitude가 1 이상이어야 한다.

## CMOS Inverter Ring Oscillator

실제 CMOS inverter ring oscillator는 출력이 rail-to-rail로 swing하므로 small-signal linear analysis가 정확하지 않다.

특징:

- oscillation은 noise가 시작한다.
- MOSFET은 항상 saturation에 있지 않다.
- large-signal simulation이 필요하다.
- clock signal 생성에 충분한 non-sinusoidal waveform을 만든다.

## N-Stage Ring Oscillator

odd number inverter ring oscillator의 주파수:

```text
fosc = 1 / (2 N TD)
```

여기서:

- `N`: stage 수
- `TD`: 한 stage의 propagation delay

예: 3-stage이면:

```text
fosc = 1 / (6 TD)
```

## Differential Ring Oscillator

differential amplifier stage를 이용하면 even-number stage도 가능하다.

특징:

- CMOS inverter chain보다 빠를 수 있다.
- 하지만 static current 때문에 power consumption이 증가한다.

## VCO

Voltage-Controlled Oscillator는 control voltage로 oscillation frequency를 조절하는 oscillator이다.

ring oscillator에서 frequency control 방법:

- current control
- load capacitance control
- delay cell bias control

## 시험 포인트

- Barkhausen 조건을 magnitude/phase로 설명한다.
- ring oscillator는 odd number inversion과 delay로 발진한다.
- CMOS inverter ring은 large-signal 회로라 small-signal 해석만으로 부족하다.
- `fosc = 1/(2NTD)`를 기억한다.
- differential ring oscillator는 faster but more power라는 trade-off가 있다.

## 같이 보면 좋은 노트

- [[21 Feedback - 음귀환 기초]]
- [[27 LC Oscillators - LC 발진기]]

