---
과목: Electric Circuits 2
유형: Lecture Note
주제: Feedback effect on input/output resistance, voltage amplifier
tags:
  - electric-circuits-2
  - feedback
  - input-resistance
  - output-resistance
---

# IO Resistance Improvement with Feedback 1 - 전압 증폭기 피드백

## 핵심 요약

negative feedback은 gain desensitization과 bandwidth extension뿐 아니라 input/output resistance도 이상적인 증폭기 조건에 가깝게 만든다. 전압 증폭기에서는 input resistance를 키우고 output resistance를 줄이는 방향으로 개선된다.

## Feedback 기본식

closed-loop gain:

```text
A_cl = A / (1 + KA)
```

loop gain:

```text
T = KA
```

feedback이 강할수록 gain은 줄지만 회로 특성은 안정화된다.

## 전압 증폭기의 이상적 저항

voltage amplifier:

```text
Rin -> infinity
Rout -> 0
```

negative voltage-voltage feedback은 이 방향으로 저항을 개선한다.

## Input Resistance 개선

feedback 없는 경우:

```text
Rin,open = 1/gm
```

feedback 있는 경우:

```text
Rin,closed = Rin,open (1 + KA)
```

직관:

- 입력 test current가 들어오면 feedback이 input voltage 변화를 줄인다.
- 같은 current에 대해 더 큰 effective resistance처럼 보인다.

## Output Resistance 개선

feedback 없는 경우:

```text
Rout,open = RD
```

feedback 있는 경우:

```text
Rout,closed = Rout,open / (1 + KA)
```

직관:

- output voltage가 흔들리면 feedback이 amplifier를 통해 반대 방향 전류를 만든다.
- test voltage에 대해 더 큰 test current가 흐르므로 effective output resistance가 작아진다.

## 전압 증폭기 요약

| 항목 | feedback 효과 |
|---|---|
| gain | `1/(1+KA)`만큼 감소 |
| input resistance | `(1+KA)`만큼 증가 |
| output resistance | `(1+KA)`만큼 감소 |
| bandwidth | `(1+KA)`만큼 증가 |

## 시험 포인트

- 전압 증폭기 negative feedback은 `Rin` 증가, `Rout` 감소를 만든다.
- 개선 비율은 loop gain `1+KA`이다.
- closed-loop gain, input resistance, output resistance를 같은 loop gain 관점에서 정리한다.

## 같이 보면 좋은 노트

- [[21 Feedback - 음귀환 기초]]
- [[25 IO Resistance Improvement with Feedback 2 - 증폭기별 저항 개선]]

