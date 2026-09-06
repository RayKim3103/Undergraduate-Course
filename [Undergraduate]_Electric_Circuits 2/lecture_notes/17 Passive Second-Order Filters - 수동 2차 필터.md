---
과목: Electric Circuits 2
유형: Lecture Note
주제: Second-order passive filters, complex poles, Q, LP HP BP AP
tags:
  - electric-circuits-2
  - filters
  - second-order-filter
  - quality-factor
---

# Passive Second-Order Filters - 수동 2차 필터

## 핵심 요약

2차 필터는 두 개의 pole을 가지며, complex conjugate pole을 사용하면 더 sharp한 frequency response와 resonance를 얻을 수 있다. 핵심 파라미터는 natural frequency `w0`와 quality factor `Q`이다. LP, HP, BP, AP 필터는 같은 denominator를 공유하고 numerator의 차이에 의해 기능이 달라진다.

## 왜 고차 필터가 필요한가

고차 필터의 장점:

- cutoff가 더 sharp하다.
- 더 다양한 magnitude/phase 응답을 만들 수 있다.
- Butterworth, Chebyshev, elliptic 같은 표준 응답을 구현할 수 있다.

## 2차 필터의 표준형

complex pole을 가진 2차 denominator:

```text
D(s) = s^2 + (w0/Q)s + w0^2
```

pole:

```text
p = -w0/(2Q) ± j w0 sqrt(1 - 1/(4Q^2))
```

`Q > 1/2`이면 complex conjugate pole이고 resonance가 나타날 수 있다.

## Damping Factor

감쇠비 `zeta`와 Q 관계:

```text
zeta = 1/(2Q)
```

분류:

- `zeta < 1`: under-damped
- `zeta = 1`: critically damped
- `zeta > 1`: over-damped

## Passive Second-Order LP Filter

RLC 기반 LP filter:

```text
H_LP(s) = w0^2 / [s^2 + (w0/Q)s + w0^2]
```

파라미터:

```text
w0 = 1/sqrt(LC)
Q = w0 R C
```

물리적 의미:

- `w0`: capacitor와 inductor 사이 energy transfer frequency
- `Q`: damping이 얼마나 작은지, resonance가 얼마나 sharp한지

## HP Filter

HP numerator는 `s^2`이다.

```text
H_HP(s) = s^2 / [s^2 + (w0/Q)s + w0^2]
```

특징:

- low frequency 차단
- high frequency 통과
- low frequency에서 `+40 dB/dec` slope

## BP Filter

BP numerator는 `(w0/Q)s` 형태이다.

```text
H_BP(s) = (w0/Q)s / [s^2 + (w0/Q)s + w0^2]
```

특징:

- peak at `w0`
- 3-dB bandwidth:

```text
BW = w0 / Q
```

Q가 클수록 passband가 좁고 sharp하다.

## All-Pass Filter

2차 all-pass는 pole과 mirror-image zero를 배치해 magnitude를 일정하게 유지하고 phase만 변화시킨다.

```text
H_AP(s) = [s^2 - (w0/Q)s + w0^2] /
          [s^2 + (w0/Q)s + w0^2]
```

특징:

- magnitude constant
- phase shift는 frequency에 따라 변함
- 고차 all-pass일수록 더 큰 phase shift 가능

## 시험 포인트

- 2차 denominator `s^2 + (w0/Q)s + w0^2`를 기억한다.
- `Q > 1/2`이면 complex pole이다.
- LP/HP/BP/AP는 numerator로 구분한다.
- BP bandwidth는 `w0/Q`이다.
- Q가 클수록 peaking과 sharpness가 커진다.

## 같이 보면 좋은 노트

- [[16 First-Order Filters - 1차 필터]]
- [[18 Filters with Inductor Simulator - 인덕터 시뮬레이터 필터]]
- [[20 Higher-Order Filters - Butterworth 고차 필터]]

