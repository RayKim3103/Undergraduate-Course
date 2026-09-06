---
과목: Electric Circuits 2
유형: Lecture Note
주제: Cascode amplifier, CS+CG, Rout boosting, headroom
tags:
  - electric-circuits-2
  - cascode
  - amplifier
  - output-resistance
---

# Cascode Amplifier - 캐스코드 증폭기

## 핵심 요약

Cascode amplifier는 CS stage 뒤에 CG stage를 붙인 구조이다. CS가 transconductance를 만들고, CG가 current buffer처럼 동작하여 output resistance를 크게 키운다. 결과적으로 높은 voltage gain을 얻을 수 있지만, transistor를 stack하기 때문에 headroom 문제가 생긴다.

## CG with Source Resistance

CG stage의 source에 저항 `RS`가 보이면 output resistance가 커진다.

```text
Rout = RD || [ro + RS(1 + gm ro)]
```

핵심은 source resistance가 `1 + gm ro`만큼 증폭되어 drain에서 보인다는 것이다.

## Cascode 구조

Cascode:

```text
CS + CG
```

하단 CS:

- input voltage를 drain current로 변환
- `Gm ≈ gm1`

상단 CG:

- current를 output node로 전달
- output resistance를 boost

## Output Resistance

두 MOS의 `ro1`, `ro2`를 고려하면:

```text
Rout ≈ RD || [ro2 + ro1(1 + gm2 ro2)]
```

`gm2 ro2 >> 1`이면:

```text
Rout ≈ RD || (gm2 ro2 ro1)
```

따라서 단순 CS보다 훨씬 큰 output resistance를 얻는다.

## Voltage Gain

CS의 transconductance와 cascode output resistance로 gain이 결정된다.

```text
Av ≈ - gm1 Rout
```

active load까지 cascode로 만들면:

```text
Rout,N ≈ gm2 ro2 ro1
Rout,P ≈ gm3 ro3 ro4
Av ≈ -gm1 (Rout,N || Rout,P)
```

## Cascode의 장점

- 높은 output resistance
- 높은 voltage gain
- Miller effect 감소
- bandwidth 개선 가능

## Cascode의 단점

가장 큰 단점은 headroom이다.

- 여러 MOSFET이 supply 사이에 stack된다.
- 각 transistor가 saturation을 유지할 최소 `VDS` 또는 `VSD`가 필요하다.
- low-voltage process에서 voltage swing이 제한된다.

## 시험 포인트

- cascode는 `CS + CG`로 이해한다.
- `RS`가 CG output에서 `RS(1+gmro)`로 boost됨을 기억한다.
- gain은 `-gm1 Rout`로 보는 것이 핵심이다.
- cascode의 장점은 큰 gain/bandwidth, 단점은 headroom이다.

## 같이 보면 좋은 노트

- [[05 Source Follower and Common-Gate - SF CG 증폭기]]
- [[14 Frequency Response of Cascode SF Differential - 고주파 응답 비교]]
- [[07 Bias Circuits and Current Mirrors - 바이어스와 전류미러]]

