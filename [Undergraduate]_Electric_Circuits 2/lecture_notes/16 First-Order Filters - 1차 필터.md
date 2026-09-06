---
과목: Electric Circuits 2
유형: Lecture Note
주제: First-order filters, LP, HP, AP, active filters
tags:
  - electric-circuits-2
  - filters
  - first-order-filter
  - active-filter
---

# First-Order Filters - 1차 필터

## 핵심 요약

필터는 입력 신호의 frequency spectrum을 바꾸는 회로이다. 1차 필터는 하나의 real pole을 가지며, zero 위치에 따라 low-pass, high-pass, all-pass 특성을 만든다. passive RC filter는 단순하지만 gain이 고정되고 loading effect가 있으며, active filter는 op-amp를 이용해 gain 제어와 buffering을 제공한다.

## Filter 정의

```text
Vo(s) = H(s) Vi(s)
```

`H(s)`가 transfer function이다.

일반 형태:

```text
H(s) = a * product(s - z_i) / product(s - p_i)
```

filter order는 denominator 차수로 결정된다.

## First-Order Filter

일반 형태:

```text
H(s) = (a1 s + a0) / (s + w0)
```

pole/zero 위치가 magnitude와 phase 변화를 결정한다.

## Low-Pass Filter

RC low-pass:

```text
H(s) = 1 / (1 + sRC)
```

pole:

```text
wp = 1 / RC
```

특징:

- low frequency 통과
- high frequency 감쇠
- high frequency slope: `-20 dB/dec`

## High-Pass Filter

RC high-pass:

```text
H(s) = sRC / (1 + sRC)
```

zero:

```text
s = 0
```

pole:

```text
wp = 1 / RC
```

특징:

- DC 차단
- high frequency 통과
- low frequency slope: `+20 dB/dec`

## All-Pass Filter

all-pass filter는 magnitude는 일정하고 phase만 변화시킨다.

대표 형태:

```text
H(s) = (s - w0) / (s + w0)
```

또는 gain sign을 포함한 형태로 구현된다.

특징:

- magnitude는 1
- phase shift 제공
- time domain에서 delay line처럼 활용 가능

## Passive RC Filter의 한계

한계:

- fixed gain
- loading effect
- source/load resistance에 따라 pole이 변함

## Active Filter

op-amp를 사용하면:

- loading effect 감소
- gain control 가능
- LP/HP/AP 구현이 쉬움

예:

```text
inverting active LP gain = -Rf/Rin * 1/(1+sRC)
```

## 시험 포인트

- LP, HP, AP의 transfer function 형태를 구분한다.
- RC pole `1/RC`를 바로 계산한다.
- passive filter의 loading effect를 설명한다.
- active filter는 op-amp feedback으로 gain과 impedance를 제어한다.

## 같이 보면 좋은 노트

- [[10 Pole Zero Bode Plot - 극점 영점 보드선도]]
- [[17 Passive Second-Order Filters - 수동 2차 필터]]
- [[19 Filters with Integrators - KHN Tow-Thomas Biquad]]

