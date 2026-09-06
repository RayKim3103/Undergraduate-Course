# Scaling, Reliability, Variability

tags: #cmos-integrated-circuit #scaling #moores-law #reliability #variability #interconnect

관련 노트: [[05 Wire - 배선 모델과 Crosstalk]], [[06_1 MTCMOS and Power Gating - 한글날 보강]]

## 핵심 요약

이 장은 CMOS scaling이 device, interconnect, power, reliability, variability에 미치는 영향을 다룬다. Moore's law와 constant field scaling의 이상적 그림을 소개한 뒤, 실제 scaling에서는 oxide tunneling, wire delay, leakage, power density, design productivity 문제가 scaling의 한계로 떠오름을 설명한다.

## Moore's Law

Moore's law는 일정 기간마다 chip에 집적 가능한 transistor 수가 증가한다는 관찰이다. Scaling은 더 많은 transistor, 더 작은 gate, 더 높은 기능 집적을 가능하게 했지만 전력과 배선 문제를 함께 키웠다.

## Constant Field Scaling

이상적인 constant field scaling에서는 모든 치수와 전압을 같은 비율로 줄여 electric field를 일정하게 유지한다.

축소 효과:

- gate length 감소
- oxide thickness 감소
- capacitance 감소
- supply voltage 감소
- gate delay 감소
- energy 감소

이 모델에서는 성능과 전력이 함께 좋아지는 그림이 나오지만, 현실 scaling은 여러 물리적 한계 때문에 이상적으로 진행되지 않는다.

## Real Scaling

실제 공정에서는 oxide thickness scaling이 느려졌다. Gate oxide가 몇 원자층 수준까지 얇아지면 tunneling current가 커져 gate leakage가 급증한다. High-k dielectric은 물리적으로 두꺼운 막을 쓰면서도 높은 gate capacitance를 유지하기 위한 해결책이다.

## Wire Scaling

Interconnect는 local wire와 global wire가 다르게 scaling된다.

| Wire 종류 | 특징 |
|---|---|
| Local interconnect | transistor 근처 짧은 연결, device scaling과 함께 줄어듦 |
| Global interconnect | chip 전체를 가로지르는 긴 연결, 길이가 충분히 줄지 않음 |

Wire 단면적이 줄면 resistance per unit length가 증가한다. Total wire capacitance와 RC delay는 geometry와 coupling에 따라 복잡하게 변한다.

## Interconnect Delay

Unrepeated wire delay는 wire resistance와 capacitance에 의해 커진다. Repeated wire는 buffer를 넣어 delay 성장을 완화하지만, repeater area와 power가 추가된다.

Scaling이 진행되면서 transistor gate delay는 줄어도 global wire delay는 상대적으로 나빠질 수 있다. 이 때문에 chip 설계는 floorplanning, buffering, hierarchy, locality가 매우 중요해졌다.

## Scaling Implications

### Interconnect Woes

Wire delay는 특정 공정 세대 이후 전체 성능 병목이 되기 시작했다. Long wire는 transistor가 빨라져도 더 빨라지지 않거나 오히려 더 나빠질 수 있다.

### Power Woes

Dynamic power density는 clock frequency와 transistor 수 증가로 문제가 되었고, VDD scaling 둔화로 완화 폭이 줄었다. Static leakage도 threshold voltage 감소와 gate oxide tunneling 때문에 커졌다.

## Static Power와 Leakage 증가

낮은 VDD에서 성능을 유지하려면 threshold voltage를 낮추는 경향이 있다. 하지만 threshold가 낮아지면 OFF 상태 subthreshold leakage가 증가한다. Thin oxide는 gate leakage를 키운다. 결과적으로 standby power가 중요한 설계 문제가 된다.

## Reliability 문제

Scaling은 다음 reliability 이슈를 키운다.

- oxide breakdown
- hot carrier degradation
- electromigration
- bias temperature instability
- supply noise와 IR drop

작은 device는 전압과 온도, 공정 변화에 더 민감하다.

## Variability

공정 변동성은 transistor와 wire 특성을 chip마다, die 내부 위치마다 다르게 만든다.

주요 변동:

- effective channel length
- threshold voltage
- oxide thickness
- dopant fluctuation
- line edge roughness
- wire width/thickness

Variability가 커지면 timing closure와 yield가 어려워진다.

## Design Productivity

Transistor 수는 빠르게 증가하지만, 설계자가 검증하고 구현할 수 있는 gate 수는 같은 속도로 늘지 않는다. 그래서 HDL synthesis, reusable IP, hierarchical design, standard cell methodology가 필수적이다.

## 시험ㆍ복습 체크포인트

- Constant field scaling의 이상적 효과를 설명할 수 있어야 한다.
- Real scaling에서 oxide thickness가 계속 줄기 어려운 이유를 말할 수 있어야 한다.
- Interconnect delay가 scaling 이후 병목이 되는 이유를 이해해야 한다.
- Dynamic power, leakage, variability가 scaling의 한계와 어떻게 연결되는지 설명할 수 있어야 한다.
- Reliability와 yield가 variability에 민감한 이유를 말할 수 있어야 한다.

