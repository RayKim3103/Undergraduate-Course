# Sequential Circuit Design과 Timing

tags: #cmos-integrated-circuit #sequential-circuit #latch #flip-flop #setup-hold #clock-skew #time-borrowing

관련 노트: [[08 Datapaths - 가산기 시프터 곱셈기]]

## 핵심 요약

이 장은 순차회로의 sequencing element 설계와 timing constraint를 다룬다. Latch와 flip-flop 구조, C2MOS latch, pulsed latch, enabled/resettable storage element, setup/hold, max-delay/min-delay, time borrowing, clock skew가 핵심이다.

## Sequencing의 필요성

Combinational logic은 입력이 바뀌면 delay 후 출력이 바뀐다. Pipeline에서 모든 token이 항상 같은 속도로 움직인다면 sequencing element가 필요 없겠지만, 실제 회로는 logic delay가 stage마다 다르다.

Sequencing element는 빠른 token을 지연시켜 한 clock cycle에 정확히 한 stage씩 이동하도록 만든다.

```text
combinational logic + storage element + clock = synchronous system
```

## Sequencing Overhead

Flip-flop이나 latch는 slow token에도 추가 delay를 더한다. 따라서 cycle time은 combinational logic delay만이 아니라 storage element overhead까지 포함한다.

```text
T_cycle >= t_pcq + t_pd_logic + t_setup + skew
```

## Latch와 Flip-Flop

| 요소 | 동작 |
|---|---|
| Latch | clock level에 민감, transparent/opaque 상태 |
| Flip-flop | clock edge에 민감 |

Latch는 transparent 동안 입력이 출력으로 통과하고, opaque 동안 이전 값을 유지한다. Flip-flop은 clock edge 근처에서만 값을 캡처한다.

## Latch Design

### Pass Transistor Latch

Pass transistor로 data path를 열고 닫는다. 간단하지만 threshold drop과 degraded level 문제가 생길 수 있다.

### Transmission Gate Latch

nMOS와 pMOS를 함께 사용해 0과 1을 모두 잘 전달한다. CMOS latch 구현에서 기본적으로 많이 사용된다.

### Tristate Feedback Latch

입력 path와 feedback path를 clock에 따라 번갈아 켜서 transparent/hold 상태를 만든다.

### Buffered Input/Output

입력과 출력에 inverter buffer를 추가하면 driving strength와 noise margin을 개선할 수 있지만 delay와 capacitance가 증가한다.

## C2MOS Latch

C2MOS(Clocked CMOS) latch는 clocked transistor와 inverter 구조를 사용해 race를 줄이고 clock phase에 따라 값을 저장한다. Clocked device 배치에 따라 transparency와 hold 특성이 결정된다.

## Flip-Flop Design

Flip-flop은 back-to-back latch 두 개로 만든다.

```text
master latch + slave latch = edge-triggered flip-flop
```

Positive edge-triggered flip-flop에서는 clock이 바뀌는 순간 master가 닫히고 slave가 열리며, edge에서 값이 전달된다.

## Pulsed Latch

Pulsed latch는 짧은 clock pulse 동안만 transparent해지는 latch이다. 외부에서 보면 edge-triggered flip-flop처럼 동작하지만, latch 기반 구조라 일부 time borrowing이 가능하다.

Tradeoff:

- flip-flop보다 빠를 수 있다.
- hold time 요구가 커질 수 있다.
- pulse width 설계가 중요하다.

## Enabled, Resettable, Settable Elements

Enabled latch/FF는 enable이 켜졌을 때만 새 값을 캡처한다. 이는 clock gating이나 mux feedback으로 구현할 수 있다.

Resettable latch/FF는 reset 입력으로 상태를 0으로 초기화하고, settable element는 1로 초기화한다. Asynchronous set/reset은 clock과 무관하게 동작하므로 metastability와 release timing을 주의해야 한다.

## Incorporating Logic into Latches

간단한 logic을 latch 내부에 흡수하면 별도 gate stage를 줄일 수 있다. 하지만 storage node의 noise margin, setup/hold, clock loading이 달라지므로 timing characterization이 필요하다.

## Timing Delay 종류

| 기호 | 의미 |
|---|---|
| `tpcq` | clock edge에서 Q가 안정될 때까지의 propagation delay |
| `tccq` | clock edge 후 Q가 변하기 시작하는 contamination delay |
| `tsetup` | clock edge 전 D가 안정되어야 하는 시간 |
| `thold` | clock edge 후 D가 유지되어야 하는 시간 |
| `tpd` | combinational logic propagation delay |
| `tcd` | combinational logic contamination delay |

## Max-Delay Constraint

Max-delay failure는 data가 다음 sequencing element의 setup time 전에 도착하지 못할 때 발생한다.

Flip-flop 기반 system:

```text
T_cycle >= tpcq + tpd_logic + tsetup + tskew
```

Combinational logic delay가 너무 크면 receiving element가 잘못된 값을 sample한다.

## Min-Delay Constraint

Min-delay failure 또는 hold failure는 data가 너무 빨리 다음 element에 도착해 hold time을 깨는 경우이다.

```text
tccq + tcd_logic >= thold + tskew
```

두 flip-flop 사이에 combinational logic이 거의 없거나 clock skew가 불리하면 hold violation이 생긴다.

## 2-Phase Latch Timing

2-phase latch system은 서로 겹치지 않는 두 clock phase를 사용한다. Latch가 transparent인 동안 data가 통과할 수 있으므로 time borrowing이 가능하지만, race-through와 hold 조건을 주의해야 한다.

## Time Borrowing

Flip-flop system에서는 한 stage의 logic이 한 cycle 안에 끝나야 한다. Latch 기반 system에서는 한 stage가 조금 늦어도 다음 latch의 transparent window를 일부 빌려 쓸 수 있다.

```text
긴 logic stage가 다음 phase의 일부 시간을 borrow
전체 loop는 cycle time 안에 완료되어야 함
```

Time borrowing은 intentional하게 설계할 수도 있고, skew나 delay variation을 흡수하는 opportunistic 효과로 나타날 수도 있다.

## Clock Skew

Clock skew는 서로 다른 storage element에 clock이 도착하는 시간 차이다.

Flip-flop처럼 hard edge를 쓰는 system에서는 skew가 useful computation time을 줄이고 hold risk를 키울 수 있다. Latch나 pulsed latch처럼 softer edge를 가진 system은 transparency window 덕분에 일부 skew를 흡수할 수 있다.

## Sequencing 방식 비교

| 방식 | 장점 | 단점 |
|---|---|---|
| Flip-flop | 설계가 단순하고 timing 분석이 명확 | time borrowing 제한, clock overhead |
| 2-phase latch | 큰 time borrowing 가능 | clock phase 설계와 race 관리가 어려움 |
| Pulsed latch | 빠르고 일부 borrowing 가능 | hold time과 pulse width에 민감 |

## 시험ㆍ복습 체크포인트

- Latch와 flip-flop의 transparent/edge-triggered 차이를 설명할 수 있어야 한다.
- `tpcq`, `tccq`, `tsetup`, `thold`를 정의할 수 있어야 한다.
- Max-delay와 min-delay constraint 식을 쓸 수 있어야 한다.
- Time borrowing이 latch 기반 system에서 가능한 이유를 이해해야 한다.
- Clock skew가 setup과 hold 조건에 미치는 영향을 설명할 수 있어야 한다.

