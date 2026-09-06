---
과목: Electric Circuits 2
유형: Lecture Note
주제: MOSFET capacitance, high-frequency small-signal model, ft
tags:
  - electric-circuits-2
  - mosfet
  - high-frequency
  - capacitance
  - ft
---

# MOSFET High-Frequency Model - 고주파 모델

## 핵심 요약

MOSFET은 gate oxide와 junction 구조 때문에 여러 capacitance를 가진다. 고주파 응답에서는 `Cgs`, `Cgd`, `Cdb`, `Csb` 같은 capacitance가 pole/zero를 만들고, amplifier bandwidth를 제한한다. transistor 자체의 속도는 unit-gain frequency `ft`로 비교한다.

## MOSFET Capacitance

이 과목에서는 실제 복잡한 capacitance model 중 핵심 capacitor만 고려한다.

주요 capacitance:

- `Cgs`: gate-source
- `Cgd`: gate-drain
- `Cdb`: drain-body
- `Csb`: source-body

PMOS도 NMOS와 동일한 방식으로 모델링한다.

## 왜 단순화하는가

실제 SPICE model은 매우 복잡하지만 손계산에서는 frequency response를 결정하는 dominant capacitor를 파악하는 것이 중요하다.

해석 흐름:

```text
MOS small-signal model
-> essential capacitances 추가
-> pole/zero 근사
-> bandwidth 추정
```

## Unit-Gain Frequency `ft`

`ft`는 common-source short-circuit current gain의 magnitude가 1이 되는 주파수이다.

정의:

```text
|Iout / Iin| = 1 at f = ft
```

대략:

```text
omega_T = gm / (Cgs + Cgd)
fT = gm / [2 pi (Cgs + Cgd)]
```

의미:

- transistor speed 비교 지표
- 어떤 transistor operation이 가능한 최대 주파수의 감각 제공

## 더 빠른 MOSFET을 만드는 법

`ft`를 키우려면:

- `gm`을 크게 한다.
- `Cgs`, `Cgd`를 작게 한다.

하지만 width를 키우면 `gm`도 커지고 capacitance도 커지므로 단순하지 않다.

## Capacitance 근사

saturation에서:

```text
Cgd ≈ W Lov Cox
Cgs ≈ W Lov Cox + (2/3) W L Cox
```

보통:

```text
Cgs > Cgd
```

overlap capacitance와 channel capacitance가 모두 고려된다.

## 예시 결과

자료의 예시 조건:

- `L = 0.25 um`
- `W = 10 um`
- `VDS = 2 V`
- `VGS = 1.5 V`

추정 `fT`는 약 `24 GHz` 수준이며, simulation/모델 값과 비슷한 범위를 보인다.

## 시험 포인트

- 고주파 MOS model에서 네 capacitor 위치를 그릴 수 있어야 한다.
- `ft = gm / [2pi(Cgs+Cgd)]`의 의미를 안다.
- `Cgd`는 Miller effect 때문에 amplifier bandwidth에 큰 영향을 준다.
- `gm`을 키우는 것과 capacitance 증가 사이 trade-off를 이해한다.

## 같이 보면 좋은 노트

- [[03 MOS Small-Signal Characteristics - MOS 소신호 모델]]
- [[12 Frequency Response of CS - CS 주파수 응답]]
- [[14 Frequency Response of Cascode SF Differential - 고주파 응답 비교]]

