---
과목: Electric Circuits 2
유형: Lecture Note
주제: Higher-order filters, Butterworth, pole placement, cascade implementation
tags:
  - electric-circuits-2
  - filters
  - butterworth
  - higher-order-filter
---

# Higher-Order Filters - Butterworth 고차 필터

## 핵심 요약

고차 필터는 1차/2차 필터보다 더 가파른 cutoff와 다양한 응답 특성을 제공한다. 이 강의는 Butterworth low-pass filter를 중심으로, ripple 없는 maximally flat passband와 pole 위치를 구하는 방법을 설명한다. 고차 Butterworth 필터는 1차 섹션과 2차 섹션의 cascade로 구현한다.

## 필터 종류 비교

| 필터 | passband ripple | stopband ripple | cutoff |
|---|---|---|---|
| Butterworth | 없음 | 없음 | 느림 |
| Chebyshev | 있음 또는 없음 | 유형에 따라 | 중간/빠름 |
| Elliptic | 있음 | 있음 | 빠름 |

Butterworth는 ripple이 없는 대신 cutoff가 가장 완만한 편이다.

## Butterworth Magnitude

N차 Butterworth LPF:

```text
|H(jw)|^2 = 1 / [1 + (w/wp)^(2N)]
```

`wp`는 cutoff frequency이며, 이 주파수에서 3-dB drop이 발생한다.

```text
|H(jwp)| = 1/sqrt(2)
```

## Pole 위치

Butterworth pole은 반지름 `wp`인 원 위에 균일하게 배치되고, 안정성을 위해 left-half plane pole만 선택한다.

개념:

```text
sk = wp exp[j theta_k]
```

left-half plane pole만 사용하여 stable transfer function을 만든다.

## N = 1

1차 Butterworth:

```text
H(s) = wp / (s + wp)
```

이는 일반적인 1차 low-pass filter이다.

## N = 2

2차 Butterworth는 complex conjugate pole 한 쌍을 갖는다.

표준형:

```text
H(s) = w0^2 / [s^2 + (w0/Q)s + w0^2]
```

Butterworth 조건:

```text
w0 = wp
Q = 1/sqrt(2) ≈ 0.707
```

## N = 3

3차 Butterworth는 다음 cascade로 구현할 수 있다.

```text
1차 Butterworth LPF
+
2차 LPF with Q = 1
```

즉 real pole 하나와 complex pole pair 하나를 cascade한다.

## N차 구현 전략

N차 Butterworth 필터는 다음 섹션들의 곱으로 구현한다.

- N이 홀수: 1차 section 하나 + 2차 sections
- N이 짝수: 2차 sections만

각 2차 section은 서로 다른 Q를 갖고 같은 cutoff scale을 공유한다.

## 설계 절차

1. passband/stopband specification에서 필요한 order `N`을 구한다.
2. cutoff `wp`를 정한다.
3. Butterworth pole 위치를 계산한다.
4. 1차/2차 section으로 factorization한다.
5. op-amp active filter 또는 passive filter로 각 section을 구현한다.

## 시험 포인트

- Butterworth는 passband가 maximally flat이고 ripple이 없다.
- `|H(jw)|^2 = 1/[1+(w/wp)^(2N)]` 형태를 기억한다.
- 2차 Butterworth의 `Q = 1/sqrt(2)`가 중요하다.
- 고차 필터는 1차/2차 section cascade로 구현한다.

## 같이 보면 좋은 노트

- [[17 Passive Second-Order Filters - 수동 2차 필터]]
- [[19 Filters with Integrators - KHN Tow-Thomas Biquad]]

