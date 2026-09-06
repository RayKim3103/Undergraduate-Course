---
과목: Electric Circuits 2
유형: Lecture Note
주제: Common-source frequency response, Miller effect, input/output poles, GBW
tags:
  - electric-circuits-2
  - common-source
  - frequency-response
  - miller-effect
---

# Frequency Response of CS - CS 주파수 응답

## 핵심 요약

CS amplifier의 주파수 응답은 MOS capacitance, 특히 input과 output을 연결하는 `Cgd` 때문에 복잡해진다. Miller theorem을 사용하면 `Cgd`를 input/output node의 등가 capacitance로 나누어 근사할 수 있다. CS는 negative gain이 크므로 input에서 `Cgd`가 `1+|Av|`배 커진 것처럼 보이고, 이것이 dominant input pole을 만든다.

## CS 고주파 해석 문제

고려할 capacitance:

- `Cgs`
- `Cgd`
- `Cdb`

`Cgd`는 input과 output을 직접 연결하므로 node가 coupling되어 해석이 복잡하다.

## Miller's Theorem

두 node 사이 impedance `ZF`가 있고 voltage gain이 `Av = Vout/Vin`이면 이를 input/output ground capacitance로 등가 변환할 수 있다.

capacitor `CF`의 경우:

```text
Cin,Miller = CF (1 - Av)
Cout,Miller ≈ CF (1 - 1/Av)
```

CS amplifier는 `Av < 0`이므로:

```text
Cin,Miller ≈ Cgd (1 + |Av|)
```

이를 Miller effect라고 한다.

## CS Amplifier의 등가 Capacitance

load resistance를 `RL`이라 하고 gain을 대략:

```text
Av ≈ -gm RL
```

라고 하면 input capacitance:

```text
CX ≈ Cgs + Cgd(1 + gm RL)
```

output capacitance:

```text
CY ≈ Cdb + Cgd(1 + 1/(gm RL))
```

gain이 크면 input 쪽 Miller capacitance가 매우 커진다.

## 근사 전달함수

3-dB bandwidth 추정을 위한 근사:

```text
Vout/Vin ≈ -gm RL /
[(1 + s RS CX)(1 + s RL CY)]
```

input pole:

```text
wp,in ≈ 1 / [RS (Cgs + Cgd(1 + gm RL))]
```

output pole:

```text
wp,out ≈ 1 / [RL (Cdb + Cgd(1 + 1/(gm RL)))]
```

## Dominant Pole

CS에서는 보통 input Miller capacitance가 커서 input pole이 dominant가 되기 쉽다.

큰 gain 조건에서:

```text
wp,in ≈ 1 / (RS Cgd gm RL)
```

즉 gain이 커질수록 bandwidth가 줄어든다.

## Gain-Bandwidth Product

CS의 gain-bandwidth product는 대략:

```text
GBW ≈ gm / (Cgd) * 1/RS factor
```

슬라이드의 핵심 메시지는 gain을 키우면 Miller effect로 bandwidth가 줄어 gain-bandwidth trade-off가 생긴다는 것이다.

## 시험 포인트

- `Cgd`가 Miller effect를 만드는 이유를 설명한다.
- negative gain amplifier에서 input Miller capacitance가 `Cgd(1+|Av|)`가 된다.
- input/output pole을 resistance와 capacitance 곱으로 근사한다.
- CS의 dominant pole은 대개 input Miller pole이다.

## 같이 보면 좋은 노트

- [[10 Pole Zero Bode Plot - 극점 영점 보드선도]]
- [[11 MOSFET High-Frequency Model - 고주파 모델]]
- [[13 Frequency Response of CS Degeneration and CG - Degeneration CG 응답]]

