# CMOS 전력소모와 저전력기법

tags: #cmos-integrated-circuit #power #dynamic-power #static-power #leakage #power-gating #dual-vdd

관련 노트: [[03_1 Low Power Techniques - 저전력 설계 보강]], [[05 Wire - 배선 모델과 Crosstalk]]

## 핵심 요약

이 장은 CMOS 회로의 power와 energy를 정의하고, dynamic power와 static power의 발생 원인 및 저감 기법을 다룬다. Dynamic power는 switching activity, capacitance, VDD, frequency에 의해 결정되고, static power는 subthreshold, gate, junction leakage가 주요 원인이다.

## Power와 Energy

전력은 단위 시간당 에너지 소모이다.

```text
P(t) = i(t) * VDD
P_avg = E / T
```

CMOS 회로는 VDD pin에 연결된 전원에서 에너지를 공급받고, switching과 leakage를 통해 에너지를 소모한다.

## Average Switching Power

Capacitor가 0에서 VDD로 충전될 때 전원에서 끌어오는 에너지는 `C VDD^2`이고, 그 중 일부가 capacitor에 저장되며 나머지는 PMOS resistance에서 열로 소모된다. 방전 시 저장된 에너지는 NMOS network에서 소모된다.

평균 dynamic power:

```text
P_dynamic = alpha * C * VDD^2 * f
```

| 항 | 의미 |
|---|---|
| `alpha` | activity factor, 한 clock에 switching이 발생하는 비율 |
| `C` | switching node capacitance |
| `VDD` | supply voltage |
| `f` | clock frequency |

VDD를 낮추는 것은 제곱 효과 때문에 dynamic power 감소에 가장 강력하다.

## Short-Circuit Power

입력이 전이하는 동안 pMOS와 nMOS가 동시에 ON이 되는 순간이 있다. 이때 VDD에서 GND로 직접 current path가 생기며 short-circuit power가 발생한다.

입력 transition이 느리거나 rise/fall balance가 좋지 않으면 short-circuit current가 커질 수 있다.

## Power Dissipation Sources

| 종류 | 원인 |
|---|---|
| Dynamic power | load capacitance 충방전 |
| Short-circuit power | switching 중 pMOS/nMOS 동시 ON |
| Static power | leakage current |

Static leakage에는 subthreshold leakage, gate leakage, junction leakage가 포함된다.

## Dynamic Power Reduction

### Activity Factor 감소

- 불필요한 switching을 줄인다.
- glitch를 줄인다.
- clock gating으로 사용하지 않는 register의 clock을 멈춘다.
- operand isolation으로 비활성 block 입력 변화를 막는다.

### Capacitance 감소

- transistor sizing을 줄인다.
- 긴 wire를 줄인다.
- high-fanout node를 줄인다.
- critical path가 아닌 gate는 작게 만든다.

### VDD Scaling

`VDD^2` 항 때문에 전압 감소가 가장 효과적이다. 다만 VDD를 낮추면 delay가 증가하므로 timing slack이 있는 block이나 architecture-level 병렬화/파이프라인과 함께 사용한다.

### Frequency Scaling

Frequency를 낮추면 dynamic power가 선형적으로 감소한다. 목표 throughput이 낮거나 workload가 작을 때 유효하다.

## Dual VDD

Dual VDD는 빠른 timing이 필요한 gate에는 높은 VDD를, slack이 있는 gate에는 낮은 VDD를 사용하는 기법이다.

| 영역 | 사용 |
|---|---|
| VDDH | critical path, speed가 중요한 cell |
| VDDL | noncritical path, power saving이 큰 cell |

주의점은 낮은 VDD 출력이 높은 VDD gate를 직접 구동할 때 static current가 흐를 수 있다는 것이다. 이를 막기 위해 level converter가 필요할 수 있고, area/power overhead가 생긴다.

## Static Power

Static power는 chip이 quiescent 상태여도 leakage 때문에 소모된다.

```text
P_static = I_leakage * VDD
```

Scaling으로 threshold voltage와 oxide thickness가 줄어들면서 static power 비중이 커졌다.

## Subthreshold Leakage Control

Subthreshold leakage는 OFF transistor에서 흐르는 전류이다. Delay와 leakage는 tradeoff 관계가 있다.

저감 방법:

- 높은 threshold voltage 사용
- gate-source voltage를 낮게 유지
- source-body bias 조절
- stack effect 활용
- sleep mode에서 power gating 사용

Series OFF transistor가 여러 개 있으면 virtual node 전압이 올라가 leakage가 크게 줄어든다.

## Power Gating

Power gating은 sleep transistor로 logic block의 VDD 또는 GND 연결을 끊어 leakage를 줄인다.

| 방식 | 설명 |
|---|---|
| Header switch | VDD 쪽 pMOS sleep transistor |
| Footer switch | GND 쪽 nMOS sleep transistor |

Active mode에서는 sleep switch가 켜져 virtual rail이 실제 rail에 가까워진다. Sleep mode에서는 switch를 꺼서 leakage path를 차단한다.

### Tradeoff

- Sleep transistor가 delay를 증가시킨다.
- Area가 증가한다.
- Wake-up 시 virtual rail charging으로 rush current와 supply bounce가 생긴다.
- State retention이 필요하면 retention flip-flop이나 clamp가 필요하다.

## Gate Leakage와 Junction Leakage

Gate leakage는 thin oxide tunneling 때문에 발생하며, 65 nm 이하 공정에서 중요해진다. High-k dielectric은 effective oxide capacitance를 유지하면서 tunneling을 줄이는 방법이다.

Junction leakage는 reverse-biased p-n junction에서 발생한다. 특히 high-Vt transistor에서 다른 leakage가 작을 때 상대적으로 드러날 수 있고, GIDL이 drain 조건에서 악화시킬 수 있다.

## 시험ㆍ복습 체크포인트

- `P_dynamic = alpha C VDD^2 f`를 쓰고 각 항의 power 저감 방법을 연결할 수 있어야 한다.
- Dynamic power와 static power의 차이를 설명할 수 있어야 한다.
- Dual VDD에서 level converter가 필요한 이유를 말할 수 있어야 한다.
- Stack effect와 power gating이 leakage를 줄이는 원리를 이해해야 한다.
- Power gating의 delay, area, wake-up noise tradeoff를 설명할 수 있어야 한다.
