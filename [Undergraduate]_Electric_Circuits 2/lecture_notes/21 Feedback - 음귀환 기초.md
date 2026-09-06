---
과목: Electric Circuits 2
유형: Lecture Note
주제: Negative feedback, loop gain, gain desensitization, bandwidth extension
tags:
  - electric-circuits-2
  - feedback
  - negative-feedback
  - loop-gain
---

# Feedback - 음귀환 기초

## 핵심 요약

Feedback은 output의 일부를 input으로 되돌리는 구조이다. negative feedback에서는 closed-loop gain이 open-loop gain `A`보다 작아지지만, gain이 feedback factor에 의해 안정화되고 bandwidth가 증가한다. 핵심 식은 `A/(1+KA)`이며, `KA`가 loop gain이다.

## Feedback Block Diagram

open-loop amplifier:

```text
Y = A F
```

feedback:

```text
F = X - K Y
```

closed-loop:

```text
Y/X = A / (1 + KA)
```

여기서:

- `A`: open-loop gain
- `K`: feedback factor
- `KA`: loop gain

## Negative Feedback 조건

negative feedback이면 output이 input error를 줄이는 방향으로 되돌아온다.

`KA > 0`이고 크면:

```text
Y/X ≈ 1/K
F = X - KY ≈ 0
```

즉 amplifier 입력 error가 거의 0이 된다.

## Gain Desensitization

open-loop gain `A`가 transistor parameter에 민감해도 closed-loop gain은:

```text
A_cl = A / (1+KA)
```

`KA >> 1`이면:

```text
A_cl ≈ 1/K
```

따라서 gain이 resistor ratio 같은 passive component에 의해 결정된다.

## Op-Amp Feedback 예

op-amp가 large gain을 가지면:

```text
V+ ≈ V-
```

non-inverting amplifier:

```text
Vo/Vs = 1 + R2/R1
```

inverting amplifier:

```text
Vo/Vs = -Rf/Rin
```

## CS with Degeneration as Feedback

source degeneration도 local negative feedback이다.

CS open-loop:

```text
A ≈ -gm RD
```

feedback factor는 source resistor `RS`에 의해 생긴다.

결과:

```text
Av = -gm RD / (1 + gm RS)
```

`gmRS >> 1`이면:

```text
Av ≈ -RD/RS
```

gain이 `gm` 변화에 덜 민감해진다.

## Bandwidth Extension

single-pole open-loop gain:

```text
A(s) = A0 / (1 + s/wp)
```

feedback 적용:

```text
A_cl(s) = A0 / (1 + KA0 + s/wp)
```

closed-loop bandwidth:

```text
wp,cl = (1 + KA0) wp
```

즉 gain은 줄지만 bandwidth는 loop gain만큼 증가한다. gain-bandwidth product는 대략 일정하게 유지된다.

## Feedback Polarity 판단

feedback 회로는 output 변화가 amplifier input error를 줄이는지 확인해 negative/positive를 판단한다.

절차:

1. input을 조금 증가시킨다.
2. feedforward output 변화를 본다.
3. feedback path가 input error를 줄이면 negative feedback이다.
4. error를 더 키우면 positive feedback이며 불안정해질 수 있다.

## 시험 포인트

- closed-loop gain `A/(1+KA)`를 기억한다.
- loop gain `KA`가 클수록 gain은 `1/K`에 가까워진다.
- feedback은 gain desensitization과 bandwidth extension을 제공한다.
- source degeneration은 local negative feedback이다.
- polarity는 output이 input error를 줄이는지로 판단한다.

## 같이 보면 좋은 노트

- [[15 OTA and Op-Amp - OTA와 연산증폭기]]
- [[24 IO Resistance Improvement with Feedback 1 - 전압 증폭기 피드백]]
- [[25 IO Resistance Improvement with Feedback 2 - 증폭기별 저항 개선]]

