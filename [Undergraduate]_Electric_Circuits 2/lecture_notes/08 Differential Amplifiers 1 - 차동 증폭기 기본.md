---
과목: Electric Circuits 2
유형: Lecture Note
주제: MOS differential pair, DM/CM, ICMR, CMRR 기초
tags:
  - electric-circuits-2
  - differential-amplifier
  - common-mode
  - differential-mode
---

# Differential Amplifiers 1 - 차동 증폭기 기본

## 핵심 요약

MOS differential pair는 두 입력의 차이 `Vin1 - Vin2`에 반응하고, 두 입력에 공통으로 들어오는 noise는 억제한다. small-signal 해석은 differential mode와 common mode로 나누어 수행한다. differential mode에서는 대칭성 때문에 half-circuit을 사용하고, common mode에서는 tail current source의 finite resistance가 중요해진다.

## Differential Pair 구조

입력:

```text
VG1, VG2
```

출력:

```text
Vout = VD1 - VD2
```

대칭 조건에서:

```text
Vin1 = Vin2 -> Vout1 = Vout2 -> Vout = 0
```

만약 `Vin1 > Vin2`이면 M1 current가 증가하고 `Vout1`은 감소한다.

## 왜 Differential Pair를 쓰는가

장점:

- 두 입력에 동시에 들어오는 noise를 cancel
- op-amp 입력단에 적합
- analog IC에서 matching과 symmetry를 활용하기 좋음
- SNR 향상

## Large-Signal 관점

tail current `ISS`는 두 transistor 사이에 나뉜다.

- `Vin1 = Vin2`: `ID1 = ID2 = ISS/2`
- `Vin1`이 커짐: `ID1` 증가, `ID2` 감소
- 충분히 큰 differential input에서는 한쪽 transistor가 tail current 대부분을 가져간다.

linear region은 `Vin1 ≈ Vin2` 근처의 작은 differential input 영역이다.

## Differential/Common Mode 분해

두 입력은 평균과 차이로 표현한다.

```text
VCM = (Vin1 + Vin2) / 2
Vid = Vin1 - Vin2

Vin1 = VCM + Vid/2
Vin2 = VCM - Vid/2
```

linear system처럼 DM 응답과 CM 응답을 superposition한다.

## Differential Mode 해석

DM에서는 회로가 anti-symmetric이다. 가운데 tail node는 AC ground로 볼 수 있다.

half-circuit:

```text
각 절반은 CS amplifier
```

`ro` 무시:

```text
Vout / Vid = - gm RD
```

`ro` 포함:

```text
Adm = - gm (RD || ro)
```

## Common Mode 해석

CM 입력에서는 두 입력이 같이 움직인다.

ideal tail current source이면:

```text
Vout1 = Vout2
Vout = Vout1 - Vout2 = 0
```

즉 differential output에서는 common-mode gain이 0이다.

finite tail resistance `RSS`가 있으면 각 single-ended output은 변할 수 있다. half-circuit에서는 source degeneration `2RSS`가 있는 CS처럼 해석할 수 있다.

## Input Common-Mode Range

입력 common-mode voltage가 너무 크거나 작으면 transistor saturation이 깨진다.

상한은 input transistor의 drain-source saturation 조건에서 결정된다.

```text
VCM,max ≈ VDD - ID RD + VTH
```

하한은 tail current source가 정상 동작할 수 있는 voltage headroom으로 결정된다.

설계 시에는:

- input pair saturation
- tail current source saturation
- output swing

을 모두 확인해야 한다.

## 시험 포인트

- `Vin1`, `Vin2`를 `VCM`, `Vid`로 분해한다.
- DM half-circuit에서 tail node가 AC ground가 되는 이유를 설명한다.
- DM gain은 CS gain과 같아진다.
- ideal tail current source에서 differential output CM gain은 0이다.
- ICMR는 모든 MOSFET saturation 조건으로 결정된다.

## 같이 보면 좋은 노트

- [[09 Differential Amplifiers 2 - CMRR와 Active Load]]
- [[15 OTA and Op-Amp - OTA와 연산증폭기]]

