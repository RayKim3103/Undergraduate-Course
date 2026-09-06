---
과목: Electric Circuits 2
유형: Lecture Note
주제: Bias circuits, coupling capacitor, current mirror, current-source load
tags:
  - electric-circuits-2
  - bias
  - current-mirror
  - current-source-load
---

# Bias Circuits and Current Mirrors - 바이어스와 전류미러

## 핵심 요약

MOS amplifier의 small-signal parameter는 bias current와 gate-source voltage에 의해 결정된다. 따라서 원하는 동작점을 만들고 유지하는 bias circuit이 필수이다. IC에서는 resistor보다 current mirror를 이용해 기준전류 `IREF`를 복사하고 scaling하는 방식이 자주 쓰인다.

## 왜 Bias가 필요한가

MOSFET 증폭기가 원하는 gain을 내려면 MOSFET이 saturation에 있어야 하고, `ID`, `VGS`, `gm`, `ro`가 적절해야 한다.

```text
gm = 2ID / VOV
ro ≈ 1 / (lambda ID)
```

따라서 DC operating point를 먼저 정한 뒤 small-signal을 얹는다.

## Gate Biasing

저항 divider로 gate voltage를 고정할 수 있다.

```text
VGS = VDD * R2 / (R1 + R2)
```

입력 신호는 coupling capacitor를 통해 넣는다.

## Coupling Capacitor

capacitor impedance:

```text
ZC = 1 / (j omega C)
```

동작:

- DC에서는 open
- 관심 주파수에서 충분히 큰 `omega C`이면 short처럼 동작

따라서 DC bias는 유지하면서 AC input만 증폭기에 전달할 수 있다. 단, coupling network가 input resistance를 낮추고 low-frequency pole을 만든다.

## Current Mirror

IC에서는 current mirror를 이용해 기준전류를 복사한다.

```text
Icopy = IREF * [(W/L)copy / (W/L)ref]
```

전제:

- 두 MOSFET이 saturation
- threshold와 process가 잘 matching
- channel-length modulation 무시 또는 작음

## Current Mirror의 장점

- 하나의 `IREF`로 여러 bias current 생성 가능
- transistor size ratio로 current scaling 가능
- IC에서 resistor보다 면적/정밀도 측면에서 유리

## Current Mirror의 제한

### Output Compliance

copy transistor가 saturation을 유지해야 하므로 output voltage 범위에 제한이 있다.

NMOS current sink:

```text
VO >= VGS - VTH = VOV
```

### Channel-Length Modulation

`VO`가 변하면 `VDS`가 변하고, finite `ro` 때문에 `IO`가 `IREF`와 달라진다.

```text
IO ≈ IREF + (VO - VGS) / ro
```

즉 ideal current source가 아니라 finite output resistance를 가진 current source이다.

## CS with Current Mirror Load

저항 load 대신 current mirror 또는 current source load를 사용하면:

- DC bias current를 안정적으로 설정
- load resistance를 크게 만들어 gain 증가
- IC 면적 감소

하지만 current mirror의 left side가 input resistance처럼 작용하거나, output voltage range가 제한되는 문제가 있을 수 있다.

## 시험 포인트

- MOS small-signal parameter가 bias에 의존한다는 점을 기억한다.
- coupling capacitor는 DC open, AC short 근사이다.
- current mirror current ratio는 `(W/L)` ratio로 결정된다.
- saturation compliance와 channel-length modulation이 current mirror 오차의 핵심이다.
- current source load는 큰 output resistance로 gain을 높인다.

## 같이 보면 좋은 노트

- [[02 MOS Large-Signal Characteristics - MOS 대신호 특성]]
- [[04 Common-Source Amplifier - CS 증폭기]]
- [[08 Differential Amplifiers 1 - 차동 증폭기 기본]]

