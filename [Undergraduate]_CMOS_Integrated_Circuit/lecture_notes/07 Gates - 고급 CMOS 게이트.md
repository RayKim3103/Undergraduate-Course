# 고급 CMOS Gates

tags: #cmos-integrated-circuit #gates #compound-gate #skewed-gate #pseudo-nmos #dynamic-circuit #domino

관련 노트: [[06_1 MTCMOS and Power Gating - 한글날 보강]], [[08 Datapaths - 가산기 시프터 곱셈기]]

## 핵심 요약

이 장은 기본 CMOS inverter, NAND, NOR를 넘어 compound gate, asymmetric gate, skewed gate, pseudo-nMOS, dynamic circuit, domino gate를 다룬다. 목표는 speed, area, power, noise margin 사이의 tradeoff를 이해하고 critical input 또는 critical transition에 맞춰 gate를 최적화하는 것이다.

## Compound Gates

Compound gate는 여러 논리 연산을 하나의 CMOS gate로 합친 구조이다.

예:

```text
Y = NOT(A B + C D)
```

이 함수는 AOI22 gate로 직접 구현할 수 있다. 별도의 AND, OR, inverter를 여러 stage로 연결하는 것보다 parasitic capacitance와 stage delay를 줄일 수 있다.

## AOI와 OAI

| Gate | 의미 | 형태 |
|---|---|---|
| AOI | AND-OR-Invert | `Y = NOT(AB + CD + ...)` |
| OAI | OR-AND-Invert | `Y = NOT((A+B)(C+D)...)` |

Static CMOS는 inverting gate 구현이 자연스럽다. 그래서 AOI/OAI는 CMOS compound logic에서 자주 사용된다.

## Compound Gate의 Logical Effort

Compound gate는 입력 위치에 따라 logical effort와 parasitic delay가 다르다. Series stack 내부에 있는 transistor가 늦게 switching하면 출력 node와 내부 node의 discharge/charge 경로가 달라져 delay 차이가 생긴다.

중요한 직관:

- Stage 수를 줄이면 parasitic을 줄일 수 있다.
- 너무 복잡한 gate는 transistor stack이 길어져 logical effort가 커진다.
- Critical input을 빠른 위치에 배치하는 것이 중요하다.

## Asymmetric Gates

Asymmetric gate는 특정 입력을 더 빠르게 만들기 위해 transistor 크기나 배치를 비대칭으로 조정한 gate이다.

예를 들어 NAND에서 input A가 critical이면 A가 보는 logical effort를 줄이도록 설계할 수 있다. 단, critical input은 빨라지지만 다른 input의 effort가 커지고 전체 평균 특성이 나빠질 수 있다.

## Symmetric Gates

Symmetric gate는 입력들이 같은 electrical behavior를 갖도록 균형 있게 설계한다. 어떤 입력이 critical인지 모를 때 또는 모든 input timing이 비슷할 때 유리하다.

## Skewed Gates

Skewed gate는 rising 또는 falling transition 중 하나를 더 빠르게 만들기 위해 pMOS/nMOS 비율을 의도적으로 바꾼 gate이다.

| 종류 | 유리한 transition | 방법 |
|---|---|---|
| HI-skew | output rising | pMOS 상대 강화 또는 nMOS 축소 |
| LO-skew | output falling | nMOS 상대 강화 또는 pMOS 축소 |

Skewed gate는 noncritical transition을 희생해 critical transition delay와 capacitance를 줄인다.

## P/N Ratio

일반 inverter에서 pMOS는 mobility가 낮아 nMOS보다 크게 잡는다. 평균 delay를 최소화하는 P/N ratio는 rise/fall equalization만을 목표로 한 비율과 다를 수 있다. 최적 비율은 delay뿐 아니라 area와 power까지 함께 고려해야 한다.

## Pseudo-nMOS

Pseudo-nMOS는 pMOS pull-up을 항상 ON으로 두고, nMOS pull-down network만으로 논리를 구현한다.

### 장점

- pMOS network가 단순해져 input capacitance가 작아질 수 있다.
- wide NOR 같은 일부 구조에서 빠르고 area가 작을 수 있다.

### 단점

- 출력이 0일 때 VDD에서 GND로 DC current가 흐른다.
- Static power가 크다.
- Noise margin이 ratioed design에 의존한다.

이 static power 문제 때문에 순수 nMOS logic이 사라지고 complementary CMOS가 주류가 되었다.

## Dynamic Circuits

Dynamic gate는 clocked pMOS로 precharge하고, evaluation phase에서 nMOS network가 조건에 따라 dynamic node를 discharge한다.

### 동작

| Phase | 동작 |
|---|---|
| Precharge | dynamic node를 high로 충전 |
| Evaluation | pull-down 조건이 참이면 node를 low로 방전 |

Dynamic gate는 pMOS pull-up network가 없어 input capacitance와 area를 줄이고 빠르게 만들 수 있다. 하지만 dynamic node가 떠 있기 때문에 leakage, noise, charge sharing에 취약하다.

## Monotonicity 제약

Dynamic gate input은 evaluation 동안 monotonically rising이어야 한다. Dynamic gate output은 evaluation 중 high에서 low로만 바뀔 수 있다. 그래서 dynamic gate가 dynamic gate를 바로 구동하면 다음 stage가 잘못 평가될 수 있다.

## Domino Gates

Domino gate는 dynamic stage 뒤에 static inverter를 붙인 구조이다.

```text
dynamic gate -> static inverter
```

Dynamic output은 evaluation 중 falling만 가능하지만, inverter를 거친 domino output은 rising만 가능해 다음 dynamic stage의 monotonic input 조건을 만족한다.

## Keeper

Dynamic node는 evaluation 중 floating 상태가 될 수 있어 leakage로 전압이 떨어진다. Keeper는 약한 feedback pMOS로 dynamic node high 값을 유지한다.

Tradeoff:

- Keeper가 강하면 noise margin이 좋아진다.
- Keeper가 강하면 pull-down evaluation이 느려진다.
- Keeper가 약하면 leakage와 noise에 취약하다.

## Charge Sharing

Dynamic gate 내부 node들이 evaluation 중 charge를 나누면 dynamic output voltage가 떨어질 수 있다. 이 droop가 inverter threshold를 넘으면 잘못된 출력이 생긴다.

해결책:

- internal node precharge
- keeper 강화
- gate 구조 단순화
- sizing 조정

## Circuit Pitfalls

자료는 pseudo-nMOS, latch, domino gate, dynamic gate와 latch 조합에서 leakage, power supply noise, delay variation, back-gate/coupling 문제가 발생할 수 있음을 강조한다. 고성능 회로일수록 이러한 비이상성을 layout과 timing까지 함께 고려해야 한다.

## 시험ㆍ복습 체크포인트

- AOI/OAI compound gate를 static CMOS network로 구성할 수 있어야 한다.
- Asymmetric gate와 skewed gate의 최적화 목표 차이를 설명해야 한다.
- Pseudo-nMOS가 static power를 소모하는 이유를 말할 수 있어야 한다.
- Dynamic gate의 precharge/evaluation phase와 monotonicity 제약을 이해해야 한다.
- Domino gate, keeper, charge sharing의 역할과 tradeoff를 설명할 수 있어야 한다.

