---
과목: Electric Circuits 2
유형: Lecture Note
주제: Pole, zero, Bode plot, s-domain transfer function
tags:
  - electric-circuits-2
  - pole-zero
  - bode-plot
  - frequency-response
---

# Pole Zero Bode Plot - 극점 영점 보드선도

## 핵심 요약

회로의 frequency response는 s-domain transfer function의 pole과 zero로 결정된다. Bode plot은 `20log10|H(jw)|`와 phase를 log frequency 축에 그린 것이다. pole은 magnitude slope를 `-20 dB/dec`씩 낮추고 phase를 `-90 deg` 변화시키며, zero는 반대로 `+20 dB/dec`, `+90 deg` 효과를 만든다.

## s-Domain 해석

시간 영역 미분방정식은 Laplace transform으로 s-domain algebra 문제로 바뀐다.

소자 impedance:

```text
ZC = 1/(sC)
ZL = sL
```

sinusoidal steady state에서는:

```text
s = jw
```

## Transfer Function

회로의 frequency-domain 특성은:

```text
H(s) = Vout(s) / Vin(s)
```

형태:

```text
H(s) = A * product(s - z_i) / product(s - p_i)
```

여기서:

- `z_i`: zeros
- `p_i`: poles

## dB Scale

power ratio:

```text
dB = 10 log10(Pout/Pin)
```

voltage ratio는 power가 voltage square에 비례하므로:

```text
dB = 20 log10(Vout/Vin)
```

## Zero at Origin

```text
H(s) = s
H(jw) = jw
```

효과:

- magnitude: `+20 dB/dec`
- phase: `+90 deg`

## Pole at Origin

```text
H(s) = 1/s
H(jw) = 1/(jw)
```

효과:

- magnitude: `-20 dB/dec`
- phase: `-90 deg`

## Real Zero

```text
H(s) = 1 + s/wz
```

효과:

- `w << wz`: 거의 변화 없음
- `w = wz`: +3 dB
- `w >> wz`: `+20 dB/dec`
- phase: `wz/10`부터 `10wz` 사이에서 0도에서 90도로 변함

## Real Pole

```text
H(s) = 1 / (1 + s/wp)
```

효과:

- `w << wp`: 거의 변화 없음
- `w = wp`: -3 dB
- `w >> wp`: `-20 dB/dec`
- phase: 0도에서 -90도로 변함

## CS Amplifier 예

MOS 자체 고주파 응답을 무시하고 output load capacitance `CL`만 고려하면:

```text
H(s) = -gm RD / (1 + s RD CL)
```

pole:

```text
wp = 1 / (RD CL)
```

DC gain:

```text
Av0 = -gm RD
```

## Bode Plot 작성 절차

1. transfer function을 pole-zero 형태로 정리한다.
2. DC gain 또는 기준 gain을 dB로 표시한다.
3. 각 zero에서 slope를 `+20 dB/dec` 추가한다.
4. 각 pole에서 slope를 `-20 dB/dec` 추가한다.
5. phase는 각 pole/zero의 decade 전후에서 부드럽게 변화시킨다.

## 시험 포인트

- capacitor/inductor의 s-domain impedance를 기억한다.
- voltage ratio에는 `20log`, power ratio에는 `10log`를 쓴다.
- pole/zero가 magnitude slope와 phase에 주는 영향을 설명할 수 있어야 한다.
- single-pole CS amplifier의 pole `1/(RDCL)`를 바로 도출한다.

## 같이 보면 좋은 노트

- [[11 MOSFET High-Frequency Model - 고주파 모델]]
- [[12 Frequency Response of CS - CS 주파수 응답]]
- [[16 First-Order Filters - 1차 필터]]

