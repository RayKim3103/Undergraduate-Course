---
과목: Electric Circuits 2
유형: Lecture Note
주제: Single-ended differential output, mismatch, CMRR, active-loaded differential amplifier
tags:
  - electric-circuits-2
  - differential-amplifier
  - cmrr
  - active-load
  - mismatch
---

# Differential Amplifiers 2 - CMRR와 Active Load

## 핵심 요약

차동 증폭기는 symmetry에 의존한다. symmetry를 깨는 모든 요소, 예를 들어 resistor mismatch, current source finite resistance, active load mirror action은 성능에 영향을 준다. 이 강의는 single-ended output에서 common-mode gain이 생기는 이유, CMRR, component mismatch, active-loaded differential amplifier의 gain을 다룬다.

## Single-Ended Output

differential output이 아니라 한쪽 drain만 output으로 쓰면 common-mode 성분이 완전히 사라지지 않는다.

DM gain:

```text
Adm(single-ended) ≈ - (1/2) gm RD
```

CM gain은 tail resistance `RSS`가 finite일 때 생긴다.

```text
Acm ≈ - RD / (2RSS + 1/gm)
```

`gm RSS >> 1`이면 대략:

```text
Acm ≈ - RD / (2RSS)
```

## CMRR

Common-Mode Rejection Ratio:

```text
CMRR = |Adm / Acm|
```

single-ended output에서는 대략:

```text
CMRR ≈ gm RSS
```

tail current source의 output resistance가 클수록 common-mode rejection이 좋아진다.

## Component Mismatch 영향

차동 pair는 좌우 대칭이 깨지면 CM 입력이 DM 출력으로 변환된다.

저항 mismatch 예:

```text
RD1 = RD
RD2 = RD + Delta RD
```

common-mode 입력에서도 두 output이 정확히 같지 않아 differential output이 생긴다.

대략:

```text
Acm,mismatch ∝ Delta RD / RSS
```

반면 differential-mode gain은 작은 resistor mismatch에 1차적으로 크게 영향을 받지 않을 수 있다.

## Active-Loaded Differential Amplifier

저항 load 대신 PMOS current mirror active load를 사용한다.

장점:

- resistor 없이 구현 가능
- matching이 좋음
- 면적 작음
- single-ended output을 쉽게 얻음
- 큰 output resistance로 high gain 가능

## Half-Circuit 근사의 한계

active load에서는 current mirror action 때문에 단순 half-circuit으로는 정확한 gain이 나오지 않는다.

단순 추정:

```text
Av ≈ - gmN (roN || roP)
```

하지만 current mirror가 short-circuit transconductance를 두 배로 키우는 효과가 있어 더 정확한 해석에서는 factor of 2 차이가 나타난다.

강의의 핵심 해석:

```text
Gm ≈ -gmN
Rout ≈ roN || roP
Av ≈ -Gm Rout
```

## 시험 포인트

- differential pair의 성능은 symmetry에 크게 의존한다.
- single-ended output은 common-mode gain을 가질 수 있다.
- CMRR는 `Adm/Acm`이고 tail resistance가 클수록 좋아진다.
- mismatch는 common-mode를 differential output으로 변환한다.
- active load의 current mirror action 때문에 half-circuit 해석이 틀릴 수 있다.

## 같이 보면 좋은 노트

- [[08 Differential Amplifiers 1 - 차동 증폭기 기본]]
- [[15 OTA and Op-Amp - OTA와 연산증폭기]]
- [[07 Bias Circuits and Current Mirrors - 바이어스와 전류미러]]

