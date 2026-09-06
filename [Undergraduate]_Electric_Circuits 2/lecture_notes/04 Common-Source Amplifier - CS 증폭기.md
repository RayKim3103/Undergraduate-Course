---
과목: Electric Circuits 2
유형: Lecture Note
주제: Common-source amplifier, amplifier types, gain, Rin, Rout, source degeneration
tags:
  - electric-circuits-2
  - amplifier
  - common-source
  - source-degeneration
---

# Common-Source Amplifier - CS 증폭기

## 핵심 요약

Common-source(CS) 증폭기는 MOSFET의 `gm vgs` 전류원 특성을 이용해 입력 전압을 출력 전압 또는 출력 전류로 변환한다. 기본 gain은 `-gm RD`이며, channel-length modulation을 고려하면 `-gm(RD || ro)`가 된다. source degeneration을 넣으면 gain은 줄지만 안정성과 선형성이 좋아진다.

## 증폭기의 네 가지 유형

| 입력 | 출력 | 유형 | 이상적 Rin | 이상적 Rout |
|---|---|---|---:|---:|
| V | V | Voltage amplifier | infinite | 0 |
| I | I | Current amplifier | 0 | infinite |
| V | I | Transconductance amplifier | infinite | infinite |
| I | V | Transimpedance amplifier | 0 | 0 |

CS는 특히 transconductance amplifier에 잘 맞는다.

## 기본 CS 증폭기

small-signal model에서:

```text
vout = - gm vin RD
Av = vout / vin = - gm RD
```

입출력 저항:

```text
Rin = infinite
Rout = RD
```

gate 전류가 없으므로 input resistance는 매우 크다.

## Channel-Length Modulation 포함

finite `ro`를 고려하면:

```text
Av = - gm (RD || ro)
Rout = RD || ro
Rin = infinite
```

`ro`가 작을수록 gain과 output resistance가 감소한다.

## CS as Transconductance Amplifier

출력 포트를 등가적으로 보면:

```text
iout = gm vin
```

따라서 transconductance gain은:

```text
Gm = gm
```

전압 gain은 이 current가 output resistance를 흐르면서 만들어진 결과이다.

## PMOS Load CS

저항 대신 PMOS load를 사용하면 IC에서 면적, 정확도, 구현성이 좋아진다.

gain:

```text
Av = - gm1 (ro1 || ro2)
```

입출력:

```text
Rin = infinite
Rout = ro1 || ro2
```

## CS with Source Degeneration

source에 `RS`를 넣으면 local negative feedback이 생긴다.

`ro` 무시 시:

```text
Av = - gm RD / (1 + gm RS)
Rin = infinite
Rout = RD
```

`gm RS >> 1`이면:

```text
Av ≈ - RD / RS
```

장점:

- gain이 transistor parameter `gm`에 덜 민감해진다.
- 선형성이 좋아진다.
- bias 안정성이 좋아진다.

단점:

- voltage gain이 감소한다.

## 시험 포인트

- 기본 CS gain `-gmRD`와 `ro` 포함 gain `-gm(RD||ro)`를 구분한다.
- CS의 `Rin`은 gate 때문에 이상적으로 infinite이다.
- source degeneration의 gain 식과 `-RD/RS` 근사를 기억한다.
- degeneration은 local negative feedback으로 이해한다.

## 같이 보면 좋은 노트

- [[03 MOS Small-Signal Characteristics - MOS 소신호 모델]]
- [[05 Source Follower and Common-Gate - SF CG 증폭기]]
- [[12 Frequency Response of CS - CS 주파수 응답]]
- [[21 Feedback - 음귀환 기초]]

