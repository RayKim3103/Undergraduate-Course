---
과목: Electric Circuits 2
유형: Lecture Note
주제: LC tank, negative resistance, cross-coupled LC oscillator, ring vs LC
tags:
  - electric-circuits-2
  - oscillator
  - lc-oscillator
  - lc-tank
---

# LC Oscillators - LC 발진기

## 핵심 요약

LC oscillator는 capacitor와 inductor 사이의 energy exchange를 이용해 비교적 깨끗한 sinusoidal oscillation을 만든다. 이상적인 LC tank는 `1/sqrt(LC)`에서 무손실로 진동하지만, 실제 tank는 parasitic resistance 때문에 에너지를 잃는다. amplifier 또는 cross-coupled pair가 이 손실을 보상하면 지속 발진이 가능하다.

## Ring Oscillator의 한계

ring oscillator는 CMOS에서 구현이 쉽고 tuning range가 넓지만 frequency가 sharp하지 않고 phase noise가 클 수 있다.

따라서 더 깨끗한 oscillation이 필요하면 LC oscillator를 사용한다.

## LC Tank의 물리적 동작

capacitor energy:

```text
EC = (1/2) C V^2
```

inductor energy:

```text
EL = (1/2) L I^2
```

capacitor가 충전되어 있으면 capacitor 전압이 inductor current를 만들고, inductor current가 다시 capacitor를 반대 방향으로 충전한다.

진동 주파수:

```text
w0 = 1 / sqrt(LC)
```

## s-Domain LC Tank

parallel 또는 series tank impedance는 `s = ±j/sqrt(LC)`에 pole을 갖는다.

이상적인 경우:

```text
Q = infinite
```

즉 energy loss가 없으면 resonance가 무한히 sharp하다.

## 실제 LC Tank

실제 tank에는 parasitic resistance `R`이 존재한다.

결과:

- energy loss 발생
- pole이 left-half plane으로 이동
- resonance는 있지만 self-sustained oscillation은 되지 않음

Q:

```text
Q = w0 R C
```

## 손실 보상

oscillation을 유지하려면 active circuit이 tank loss를 보상해야 한다.

조건:

```text
gm R >= 1
```

강의 표현:

- `gm R = 1`이면 loss를 정확히 보상
- startup을 위해 실제로는 `gm R > 1` 필요
- amplitude가 커지면 nonlinear effect로 제한됨

## Cross-Coupled LC Oscillator

cross-coupled pair는 LC tank에 negative resistance를 제공한다.

개념:

```text
active pair supplies energy lost in R
LC tank determines oscillation frequency
```

처음에는 noise가 작은 differential perturbation을 만들고, `gm > 1/R`이면 `VX`, `VY`의 차이가 성장한다. 이후 tail current `ISS`와 transistor nonlinearity가 amplitude를 제한한다.

## LC vs Ring Oscillator

| 항목 | LC Oscillator | Ring Oscillator |
|---|---|---|
| phase noise | 작음 | 큼 |
| 주파수 sharpness | 좋음 | 낮음 |
| CMOS 구현 | inductor 때문에 어려움 | 쉬움 |
| 최대 주파수 | 높게 가능 | delay 제한 |
| tuning range | 좁음 | 넓음 |
| 면적 | 큼 | 작음 |

## 시험 포인트

- LC tank의 resonance frequency `1/sqrt(LC)`를 기억한다.
- 실제 tank는 parasitic resistance 때문에 loss가 있다.
- active circuit은 loss를 보상하는 negative resistance 역할을 한다.
- cross-coupled LC oscillator의 startup 조건은 `gmR > 1`로 이해한다.
- LC는 clean oscillation, ring은 구현 용이성과 tuning range가 장점이다.

## 같이 보면 좋은 노트

- [[26 Ring Oscillators - 링 발진기]]
- [[21 Feedback - 음귀환 기초]]

