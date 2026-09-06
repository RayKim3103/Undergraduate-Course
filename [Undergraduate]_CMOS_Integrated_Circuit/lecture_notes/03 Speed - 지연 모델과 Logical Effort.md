# 지연 모델과 Logical Effort

tags: #cmos-integrated-circuit #delay #rc-model #elmore-delay #logical-effort #gate-sizing

관련 노트: [[02 Devices - MOS 소자 모델과 비이상성]], [[03_1 Low Power Techniques - 저전력 설계 보강]]

## 핵심 요약

이 장은 CMOS gate의 속도를 propagation delay, contamination delay, RC delay model, Elmore delay, logical effort로 분석한다. 핵심은 transistor와 wire를 저항/커패시턴스로 근사해 delay를 빠르게 추정하고, multistage path에서 stage 수와 gate size를 선택하는 방법을 배우는 것이다.

## Delay 정의

| 기호 | 의미 |
|---|---|
| `tpdr` | rising propagation delay |
| `tpdf` | falling propagation delay |
| `tpd` | 평균 propagation delay |
| `tcdr` | rising contamination delay |
| `tcdf` | falling contamination delay |
| `tcd` | 평균 contamination delay |

```text
tpd = (tpdr + tpdf) / 2
tcd = (tcdr + tcdf) / 2
```

Propagation delay는 출력이 최종적으로 바뀌는 늦은 경우를 나타내고, contamination delay는 입력 변화가 출력에 처음 영향을 주기 시작하는 빠른 경우를 나타낸다.

## Delay Estimation

SPICE simulation이 가장 정확하지만, 초기 설계에서는 RC model로 빠르게 delay를 추정한다.

기본 가정:

- Unit nMOS는 resistance `R`, capacitance `C`
- Unit pMOS는 mobility가 낮아 더 큰 resistance를 가짐
- 더 넓은 transistor는 resistance가 작아지지만 capacitance가 커짐

Inverter fanout-of-1 delay를 기준 단위로 삼아 다른 gate delay를 비교한다.

## Elmore Delay

Pull-up 또는 pull-down network를 RC ladder로 모델링하면 Elmore delay를 사용할 수 있다.

```text
t_Elmore = sum_i R_common_i * C_i
```

각 capacitor `C_i`에 대해, 입력 source에서 그 capacitor까지 공유되는 resistance를 곱해 모두 더한다. Elmore delay는 정확한 waveform 해석보다 단순하지만 RC tree의 지연 직관을 잘 제공한다.

## Delay Components

Logical effort 모델에서 gate delay는 두 부분으로 나뉜다.

```text
d = f + p
f = g h
```

| 기호 | 의미 |
|---|---|
| `g` | logical effort, 같은 drive를 내는 inverter 대비 입력 capacitance |
| `h` | electrical effort, `Cout / Cin` |
| `f` | effort delay 또는 stage effort |
| `p` | parasitic delay |
| `d` | normalized stage delay |

## Logical Effort

Logical effort는 특정 gate가 inverter와 비교해 같은 output drive를 만들기 위해 얼마나 큰 input capacitance를 요구하는지 나타낸다.

일반적으로:

- NAND는 NOR보다 빠르다.
- NOR는 pMOS series 때문에 logical effort와 delay가 커지기 쉽다.
- Gate input마다 diffusion position과 stack 위치에 따라 parasitic delay가 다를 수 있다.

## FO4 Inverter

FO4는 fanout-of-4 inverter delay를 의미한다. Digital design에서 process-independent delay 단위처럼 자주 쓰인다.

```text
FO4 inverter: h = 4, g = 1
d = gh + p
```

FO4 delay는 서로 다른 공정이나 설계의 속도를 비교하는 rough metric으로 유용하다.

## Multistage Logical Effort

여러 gate path의 전체 effort는 다음 요소로 구성된다.

| 항목 | 의미 |
|---|---|
| Path logical effort `G` | 각 stage의 logical effort 곱 |
| Path electrical effort `H` | path 최종 load / path 입력 capacitance |
| Branching effort `B` | path 밖으로 갈라지는 load 영향 |
| Path effort `F` | `F = G B H` |

Path parasitic delay는 각 stage parasitic delay의 합이다.

## 최적 Stage Effort

N개 stage path에서 delay가 최소가 되려면 각 stage가 비슷한 effort를 부담하는 것이 좋다.

```text
f_hat = F^(1/N)
D = N*f_hat + P
```

실무적으로 stage effort가 약 4 근처일 때 빠른 경우가 많다. Delay는 최적 stage 수나 size에서 약간 벗어나도 크게 악화되지 않는 편이다.

## Gate Sizing 절차

1. Path logical effort `G`를 계산한다.
2. Electrical effort `H`를 계산한다.
3. Branch가 있으면 branching effort `B`를 포함한다.
4. `F = G B H`를 구한다.
5. Stage 수 `N`과 best stage effort를 정한다.
6. Load에서 입력 쪽으로 거꾸로 gate capacitance와 width를 계산한다.

## Logical Effort의 한계

- Delay model이 단순하다.
- Interconnect delay가 큰 회로에서는 반복적인 보정이 필요하다.
- 최소 delay를 주지만 최소 area 또는 최소 power를 보장하지 않는다.
- 매우 작은 공정에서는 velocity saturation, coupling, variation이 더 중요해진다.

## 시험ㆍ복습 체크포인트

- Propagation delay와 contamination delay의 차이를 설명할 수 있어야 한다.
- Elmore delay 계산 원리를 이해해야 한다.
- `d = gh + p`에서 `g`, `h`, `p`의 의미를 말할 수 있어야 한다.
- `F = GBH`와 best stage effort를 사용해 multistage path delay를 추정할 수 있어야 한다.
- NAND가 NOR보다 빠른 이유를 pMOS stack 관점에서 설명할 수 있어야 한다.

