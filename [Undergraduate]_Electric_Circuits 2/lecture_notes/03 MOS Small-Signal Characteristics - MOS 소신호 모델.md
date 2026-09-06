---
과목: Electric Circuits 2
유형: Lecture Note
주제: MOS small-signal model, gm, ro, body effect, PMOS model
tags:
  - electric-circuits-2
  - mosfet
  - small-signal
  - gm
  - ro
---

# MOS Small-Signal Characteristics - MOS 소신호 모델

## 핵심 요약

MOSFET의 saturation 영역 전류는 `VGS`에 비선형적으로 의존하지만, bias point 근처의 작은 변화는 선형화하여 voltage-controlled current source로 볼 수 있다. 이때 핵심 파라미터가 transconductance `gm`, output resistance `ro`, body transconductance `gmb`이다.

## 왜 Small-Signal Model을 쓰는가

MOSFET의 대신호 특성은 비선형이다. 하지만 특정 DC bias point 근처에서 작은 입력 변화만 고려하면 Taylor expansion의 1차항으로 근사할 수 있다.

```text
ID ≈ ID0 + gm * vgs
```

DC 성분과 small-signal 성분을 분리하면 회로 해석이 훨씬 쉬워진다.

## MOSFET as VCCS

saturation에서 MOSFET은 gate-source voltage로 drain current를 조절하는 voltage-controlled current source이다.

```text
id = gm vgs
```

이 전류가 load resistor를 흐르면 voltage amplification이 생긴다.

```text
vout = - gm vgs R
Av = vout / vgs = - gm R
```

## Transconductance `gm`

saturation current:

```text
ID = (1/2) mu_n Cox (W/L) (VGS - VTH)^2
```

정의:

```text
gm = dID / dVGS
```

대표 표현:

```text
gm = mu_n Cox (W/L) (VGS - VTH)
gm = 2ID / (VGS - VTH)
gm = sqrt(2 mu_n Cox (W/L) ID)
```

해석:

- `gm`은 `VGS - VTH`에 선형적으로 비례한다.
- `gm`은 `ID`의 제곱근에 비례한다.
- bias current와 transistor size가 small-signal gain을 결정한다.

## Small-Signal Circuit 작성 규칙

small-signal equivalent를 만들 때:

- DC voltage source는 AC ground
- DC current source는 open
- MOSFET은 `gm vgs` current source로 대체
- channel-length modulation을 고려하면 drain-source 사이에 `ro` 추가

## Channel-Length Modulation과 `ro`

channel-length modulation 포함:

```text
ID = (1/2) mu_n Cox (W/L) (VGS - VTH)^2 (1 + lambda VDS)
```

small-signal output resistance:

```text
ro = dVDS / dID ≈ 1 / (lambda ID)
```

`ro`가 finite이면 amplifier gain이 `RD`가 아니라 `RD || ro`에 의해 제한된다.

## Body Effect와 `gmb`

body voltage가 source와 다르면 drain current가 변한다.

small-signal에서는 body effect를 다음 current source로 표현한다.

```text
id_body = gmb vbs
gmb = chi gm
```

여기서 `chi`는 보통 `0.1 ~ 0.3` 수준으로 다룬다. 손계산에서는 자주 무시하지만 IC simulation에서는 중요하다.

## PMOS Small-Signal Model

PMOS도 NMOS와 같은 구조로 small-signal model을 만든다. 다만 전압 기준을 `VSG`, `VSD`로 잡고 current 방향을 주의한다.

PMOS transconductance:

```text
gm = mu_p Cox (W/L) (VSG - |VTH|)
gm = 2ID / (VSG - |VTH|)
```

small-signal model은 NMOS와 동일한 형태로 사용할 수 있으나, controlled current 방향과 node polarity를 일관되게 잡아야 한다.

## 시험 포인트

- `gm = dID/dVGS` 정의와 세 가지 표현을 모두 연결한다.
- DC source 처리 규칙을 기억한다.
- `ro ≈ 1/(lambda ID)`가 channel-length modulation에서 나온다.
- PMOS도 같은 model이지만 polarity가 반대임을 주의한다.
- body effect를 포함하면 `gmb vbs` current source가 추가된다.

## 같이 보면 좋은 노트

- [[02 MOS Large-Signal Characteristics - MOS 대신호 특성]]
- [[04 Common-Source Amplifier - CS 증폭기]]
- [[11 MOSFET High-Frequency Model - 고주파 모델]]

