# MTCMOS와 Power Gating 보강

tags: #cmos-integrated-circuit #mtcmos #power-gating #leakage #stack-effect #retention

관련 노트: [[06 Scaling Reliability Variability - 스케일링 신뢰성 변동성]], [[07 Gates - 고급 CMOS 게이트]]

## 핵심 요약

이 보강 자료는 leakage 저감을 위한 stack effect, gate replacement, MTCMOS, sleep transistor sizing, local/global footswitch, data holding, power gating transition 문제를 다룬다. 핵심은 active mode의 성능 저하를 제한하면서 sleep mode leakage를 크게 줄이는 구조를 설계하는 것이다.

## Stack Effect

여러 OFF transistor가 series로 연결되면 leakage가 줄어든다. 내부 virtual node 전압이 상승하거나 하강하면서 각 OFF transistor의 effective `Vgs`, `Vds`, body bias가 leakage를 줄이는 방향으로 바뀌기 때문이다.

```text
single OFF transistor leakage > stacked OFF transistor leakage
```

Forced-stack은 leakage를 줄이기 위해 transistor를 의도적으로 나누어 series stack을 만드는 방법이다. 단, series 저항이 증가하므로 delay가 늘어난다.

## Gate Replacement

Gate replacement는 worst leakage state에서 내부 logic gate를 기능은 유지하면서 leakage가 더 작은 library gate로 바꾸는 방법이다.

핵심 조건:

- 논리 기능이 유지되어야 한다.
- worst leakage input state에서 leakage가 감소해야 한다.
- delay와 area overhead가 허용 범위여야 한다.

## MTCMOS 기본 구조

MTCMOS(Multi-Threshold voltage CMOS)는 logic에는 빠른 low-Vt transistor를 쓰고, power rail 차단에는 leakage가 작은 high-Vt sleep transistor를 사용한다.

### Active Mode

```text
sleep transistor ON
virtual VDD ≈ VDD
virtual GND ≈ GND
```

Logic은 low-Vt device로 빠르게 동작한다.

### Sleep Mode

```text
sleep transistor OFF
virtual rails floating
leakage는 high-Vt sleep transistor가 제한
```

Block leakage가 크게 줄어든다.

## Performance Constraint

Sleep transistor는 active mode에서 series resistance처럼 동작한다. 따라서 virtual rail이 완전히 ideal rail과 같지 않고, logic gate의 effective supply voltage가 줄어 delay가 증가한다.

```text
Veff = VDD - voltage drop across sleep switch
```

Sleep switch width가 커지면 delay penalty는 줄지만 area와 gate capacitance가 증가한다.

## Headswitch와 Footswitch

| 방식 | 위치 | 특징 |
|---|---|---|
| Headswitch | VDD와 logic 사이, 보통 pMOS | leakage 차단, pMOS 면적 큼 |
| Footswitch | logic과 GND 사이, 보통 nMOS | drive가 강하고 면적 효율적 |

Local footswitch는 작은 block마다 switch를 두는 방식이고, global footswitch는 큰 영역을 하나의 switch network로 공유한다. Local 방식은 제어가 세밀하지만 면적 overhead가 크고, global 방식은 효율적이지만 rail bounce와 wake-up control이 중요하다.

## Sleep Transistor Sizing

Sizing 목표는 active delay 증가를 제한하면서 leakage saving을 충분히 얻는 것이다.

고려 요소:

- peak current
- virtual rail voltage drop
- wake-up time
- rush current
- area
- sleep switch gate capacitance

자료의 SOR 개념은 sleep switch의 상대 크기가 delay와 leakage에 영향을 준다는 점을 보여준다. 더 큰 switch는 delay를 줄이지만 power/area overhead가 증가한다.

## Data Holding

Power gating으로 block 전원이 꺼지면 내부 register state가 사라질 수 있다. 이를 막기 위해 state retention 기법을 사용한다.

예:

- data holding flip-flop
- balloon latch
- leakage feedback flip-flop
- intermittent power supply
- virtual power/ground clamp

Retention 회로는 sleep 중 필요한 상태만 저장하고, wake-up 후 정상 동작을 이어가게 한다.

## Power Gating Transition 문제

Conventional power gating은 sleep에서 active로 전환할 때 큰 current spike와 on-chip power distribution noise를 만들 수 있다.

문제:

- virtual rail 충전/방전으로rush current 발생
- supply bounce
- ground bounce
- wake-up delay
- retention state 복원 timing

## 개선된 Power Gating Scheme

보강 자료는 intermediate power saving mode, zigzag 또는 staged wake-up 같은 scheme을 소개한다. 공통 목표는 다음과 같다.

- sleep leakage를 낮춘다.
- active mode 성능을 유지한다.
- wake-up current를 분산한다.
- virtual rail fluctuation을 줄인다.

## 시험ㆍ복습 체크포인트

- Stack effect가 leakage를 줄이는 물리적 이유를 설명할 수 있어야 한다.
- MTCMOS에서 low-Vt logic과 high-Vt sleep transistor의 역할을 구분해야 한다.
- Sleep transistor sizing의 delay/area/leakage tradeoff를 말할 수 있어야 한다.
- Local footswitch와 global footswitch의 차이를 이해해야 한다.
- Power gating에서 data retention과 wake-up noise가 왜 중요한지 설명할 수 있어야 한다.

