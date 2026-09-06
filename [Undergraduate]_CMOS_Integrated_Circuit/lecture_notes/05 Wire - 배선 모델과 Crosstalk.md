# Wire 모델과 Crosstalk

tags: #cmos-integrated-circuit #wire #interconnect #rc-delay #crosstalk #repeater

관련 노트: [[04 Power - CMOS 전력소모와 저전력기법]], [[06 Scaling Reliability Variability - 스케일링 신뢰성 변동성]]

## 핵심 요약

이 장은 chip interconnect의 resistance, capacitance, crosstalk, repeater insertion을 다룬다. 현대 CMOS에서는 transistor만큼 wire가 delay와 power를 지배한다. 긴 wire의 RC delay는 길이의 제곱에 비례하므로 repeater로 나누어 delay를 줄인다.

## Interconnect의 중요성

Chip은 많은 transistor로 구성되지만, 실제 layout에서는 여러 metal layer의 wire가 대부분의 면적과 parasitic을 차지한다. Technology scaling 이후 gate delay보다 wire delay가 더 큰 병목이 되는 경우가 많다.

## Wire Geometry

Wire는 폭, 두께, 길이, 주변 wire와의 간격으로 특성이 결정된다.

- 좁고 긴 wire는 resistance가 크다.
- 이웃 wire와 가까우면 coupling capacitance가 커진다.
- 상하 metal layer와 substrate도 capacitance를 만든다.

## Wire Resistance

Wire resistance는 다음 경향을 가진다.

```text
R = rho * L / A
```

| 요소 | 변화 | Resistance |
|---|---|---|
| 길이 `L` 증가 | 전류 경로 증가 | 증가 |
| 단면적 `A` 증가 | 전류 경로 넓어짐 | 감소 |
| resistivity 증가 | 재료가 덜 도전적 | 증가 |

과거에는 aluminum wire를 많이 사용했지만, scaling과 성능 요구로 copper interconnect가 사용되었다.

## Wire Capacitance

Wire capacitance는 단순 parallel plate만으로 설명되지 않는다. Fringe field와 neighbor coupling이 중요하다.

주요 성분:

- wire와 substrate 사이 capacitance
- 위아래 metal layer와의 capacitance
- 인접 wire와의 coupling capacitance

Diffusion runner는 capacitance가 매우 커서 긴 배선으로 사용하면 좋지 않다. Polysilicon도 gate에는 필요하지만 긴 wire로 쓰기에는 저항이 크다.

## Distributed RC와 Elmore Delay

Wire는 길이에 따라 R과 C가 분포된 distributed system이다. 간단한 분석에서는 pi model 또는 single segment model로 근사하고 Elmore delay를 사용한다.

긴 wire의 unrepeated RC delay는 대략 길이의 제곱에 비례한다.

```text
t_wire ∝ R_wire * C_wire ∝ L^2
```

## Crosstalk

인접 wire 사이 coupling capacitance 때문에 한 wire의 switching이 다른 wire에 영향을 준다.

### Crosstalk 효과

- nonswitching victim wire에 noise pulse 발생
- switching victim wire의 delay 증가 또는 감소
- false switching과 timing violation 가능

Aggressor와 victim이 반대 방향으로 switching하면 effective capacitance가 커져 delay가 증가한다. 같은 방향으로 switching하면 coupling 전압 변화가 작아져 delay가 줄 수 있다.

## Crosstalk Delay

Coupling capacitance는 worst-case timing에서 Miller factor로 더 크게 보일 수 있다.

```text
C_effective = C_ground + k * C_coupling
```

여기서 `k`는 aggressor switching 방향에 따라 0, 1, 2에 가까운 값이 될 수 있다.

## Repeater

긴 wire를 하나의 driver가 직접 구동하면 RC delay가 너무 크다. Repeater는 wire를 여러 구간으로 나누는 inverter buffer이다.

```text
long wire -> repeater -> shorter wire segments
```

길이 `L`의 wire를 `N`개 구간으로 나누면 각 구간 길이는 `L/N`이 되고, wire delay의 제곱 의존성이 완화된다.

## Repeater Design

Repeater 설계에서는 두 값을 정해야 한다.

| 값 | 의미 |
|---|---|
| `N` | repeater 개수 또는 segment 수 |
| `W` | repeater inverter width |

Repeater가 너무 적으면 wire RC delay가 크고, 너무 많으면 repeater 자체 parasitic과 power가 커진다. 최적점은 wire 저항/정전용량과 inverter drive 특성에 의해 결정된다.

## Repeated Wire의 의미

Properly repeated wire의 delay per unit length는 unrepeated long wire보다 훨씬 작다. 하지만 repeater는 area와 dynamic power를 추가하므로 모든 wire에 무조건 넣지는 않는다. Global interconnect나 timing-critical long net에 주로 사용한다.

## 시험ㆍ복습 체크포인트

- Wire resistance가 길이와 단면적에 어떻게 의존하는지 설명할 수 있어야 한다.
- Wire capacitance에서 coupling capacitance가 중요한 이유를 말할 수 있어야 한다.
- Unrepeated wire delay가 길이 제곱에 비례하는 이유를 이해해야 한다.
- Crosstalk이 noise와 delay variation을 만드는 과정을 설명할 수 있어야 한다.
- Repeater insertion의 delay/power/area tradeoff를 말할 수 있어야 한다.

