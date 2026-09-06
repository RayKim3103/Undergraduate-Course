---
과목: Electric Circuits 2
유형: Lecture Note
주제: Source follower, common-gate amplifier, voltage buffer, current buffer
tags:
  - electric-circuits-2
  - source-follower
  - common-gate
  - buffer
---

# Source Follower and Common-Gate - SF CG 증폭기

## 핵심 요약

Source follower(SF)는 voltage gain이 1보다 조금 작은 voltage buffer이고, common-gate(CG)는 낮은 input resistance와 높은 output resistance를 가진 current buffer/current amplifier로 해석하기 좋다. CS의 큰 gain 뒤에 SF를 붙이면 load가 작은 경우에도 CS gain을 보존할 수 있다.

## Source Follower

`ro`를 무시하면 source follower의 voltage gain은:

```text
Av = gm RL / (1 + gm RL)
```

input/output resistance:

```text
Rin = infinite
Rout ≈ 1/gm || RL
```

특징:

- gain은 1보다 작고 1에 가깝다.
- input resistance가 크다.
- output resistance가 작다.
- voltage buffer로 적합하다.

## `ro` 포함 Source Follower

finite `ro`를 고려하면 source node에서 보는 저항은 대략:

```text
Rout ≈ 1/gm || ro || RL
```

body effect까지 고려하면 effective transconductance가 `gm + gmb`가 되어 output resistance가 더 작아질 수 있다.

## CS + SF

CS 출력에 작은 `RL`을 직접 연결하면:

```text
Av = -gm (RD || RL)
```

가 되어 gain이 크게 줄어든다. SF를 buffer로 붙이면 CS는 큰 input resistance를 보는 셈이 되어 gain을 유지하고, SF가 작은 output resistance로 load를 구동한다.

```text
CS: large gain
SF: voltage buffer
```

## Common-Gate Amplifier

CG는 gate가 AC ground이고 source로 입력을 넣고 drain에서 출력을 얻는다.

`ro` 무시 시:

```text
Rin = 1/gm
Rout = RD
Av = gm RD
```

CS와 달리 voltage gain이 non-inverting이다.

## CG as Current Buffer

CG는 input resistance가 작고 output resistance가 크므로 current amplifier/current buffer에 적합하다.

전류 gain은 이상적으로:

```text
Ai ≈ -1
```

즉 source로 들어온 전류가 drain 쪽으로 거의 전달된다.

## `ro` 포함 CG 결과

강의 homework에서 제시된 결과:

```text
Rin = (ro + RD) / (1 + gm ro)
Rout = ro || RD
Av = (1 + gm ro) RD / (RD + ro)
```

`gm ro >> 1`이면 `Rin`은 `1/gm`에 가까워진다.

## CS, SF, CG 비교

| 회로 | gain | Rin | Rout | 대표 용도 |
|---|---:|---:|---:|---|
| CS | 큼, negative | 큼 | 중간/큼 | transconductance, voltage gain |
| SF | 약 1 | 큼 | 작음 | voltage buffer |
| CG | positive, `gmRD` | 작음 | 큼 | current buffer, wideband input |

## 시험 포인트

- SF gain `gmRL/(1+gmRL)`과 buffer 역할을 기억한다.
- CG의 `Rin ≈ 1/gm`이 왜 낮은지 설명할 수 있어야 한다.
- CS+SF가 load effect를 줄이는 이유를 이해한다.
- CG는 current buffer로 적합하며 current gain이 대략 `-1`이다.

## 같이 보면 좋은 노트

- [[04 Common-Source Amplifier - CS 증폭기]]
- [[06 Cascode Amplifier - 캐스코드 증폭기]]
- [[13 Frequency Response of CS Degeneration and CG - Degeneration CG 응답]]

