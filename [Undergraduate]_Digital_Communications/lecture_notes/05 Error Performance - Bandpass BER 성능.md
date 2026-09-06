---
과목: Digital Communications
유형: Lecture Note
주제: Coherent/Noncoherent BER, M-ary systems, MPSK, MFSK
tags:
  - digital-communications
  - ber
  - ser
  - bpsk
  - bfsk
  - mpsk
  - mfsk
---

# Error Performance - Bandpass BER 성능

## 핵심 요약

이 강의는 bandpass digital modulation의 error performance를 `Eb/N0`, BER, SER 관점에서 비교한다. coherent BPSK/BFSK, noncoherent BFSK, DPSK, MPSK, MFSK가 핵심이며, power efficiency와 bandwidth efficiency의 trade-off를 정리한다.

## Eb/N0와 SNR

bit energy:

```text
E_b = S T_b = S / R_b
```

noise power:

```text
N = N_0 W
```

관계:

```text
E_b / N_0 = S / (R_b N_0)
S/N = S / (N_0 W)
```

따라서 같은 수신 power라도 bit rate와 bandwidth에 따라 `Eb/N0`와 SNR의 해석이 달라진다.

## Coherent Binary Systems

### Coherent BPSK

BPSK는 antipodal signaling이다.

```text
s_1(t) = sqrt(E_b) psi_1(t)
s_2(t) = -sqrt(E_b) psi_1(t)
```

BER:

```text
P_b = Q(sqrt(2E_b / N_0))
```

BPSK에서는 bit error probability와 symbol error probability가 같다.

### Coherent BFSK

BFSK는 orthogonal signaling이다.

BER:

```text
P_b = Q(sqrt(E_b / N_0))
```

BPSK와 BFSK의 성능 차:

- coherent BPSK는 antipodal이라 Euclidean distance가 더 크다.
- coherent BFSK는 orthogonal이라 같은 BER에 약 3 dB 더 많은 power가 필요하다.

## Binary Signaling의 일반식

두 equal-energy 신호의 cross-correlation coefficient를 `rho`라고 하자.

```text
rho = (1/E_b) integral_0^T s_1(t) s_2(t) dt
```

energy difference:

```text
E_d = 2E_b(1 - rho)
```

BER:

```text
P_b = Q(sqrt(E_b(1 - rho) / N_0))
```

특수 경우:

| rho | 신호 관계 | BER |
|---:|---|---|
| 1 | 동일 신호 | `1/2` |
| -1 | antipodal | `Q(sqrt(2E_b/N_0))` |
| 0 | orthogonal | `Q(sqrt(E_b/N_0))` |

## Noncoherent Binary Systems

### Noncoherent BFSK

수신기는 각 tone에 대한 BPF와 envelope detector를 사용한다.

```text
choose branch with larger envelope
```

송신 tone이 있는 branch의 envelope는 Rician 분포를 따르고, 없는 branch의 envelope는 Rayleigh 분포를 따른다.

BER:

```text
P_b = (1/2) exp(-E_b / (2N_0))
```

coherent BFSK에 비해 같은 BER 기준 약 1 dB 성능 열화가 나타난다.

### DPSK

DPSK는 이전 symbol과 현재 symbol의 phase difference로 bit를 판단한다.

BER:

```text
P_b = (1/2) exp(-E_b / N_0)
```

DPSK는 BPSK보다 성능이 떨어지지만 carrier phase recovery가 어려운 환경에서 구현이 쉽다.

## BER 성능 요약

| 방식 | BER 형태 | 해석 |
|---|---|---|
| Coherent BPSK | `Q(sqrt(2E_b/N_0))` | 가장 좋은 binary 성능 |
| Coherent BFSK | `Q(sqrt(E_b/N_0))` | BPSK보다 3 dB 손해 |
| Noncoherent BFSK | `(1/2) exp(-E_b/(2N_0))` | phase 정보 불필요 |
| DPSK | `(1/2) exp(-E_b/N_0)` | differential detection |

## M-ary Systems

M-ary modulation은 `k`개의 bit를 하나의 symbol로 묶는다.

```text
M = 2^k
R_b = (log2 M) R_s
```

성능 판단 기준:

- bandwidth efficiency
- bit error rate
- symbol error rate
- required power

## MPSK Bandwidth Efficiency

MPSK는 2차원 신호공간에서 더 많은 symbol을 phase로 배치한다.

대역폭:

```text
B_MPSK = 2R_s = 2R_b / log2 M
```

bandwidth efficiency:

```text
eta = 0.5 log2 M  [bps/Hz]
```

M이 커질수록 bandwidth efficiency는 증가하지만, 같은 energy에서 symbol 간 Euclidean distance가 줄어 BER/SER이 나빠진다.

## MFSK Bandwidth Efficiency

### Coherent MFSK

coherent MFSK의 최소 tone spacing:

```text
Delta f = 1 / (2T_s) = R_s / 2
```

대역폭 효율:

```text
eta = 2 log2 M / (M + 3)
```

### Noncoherent MFSK

noncoherent MFSK의 최소 tone spacing:

```text
Delta f = 1 / T_s = R_s
```

대역폭 효율:

```text
eta = log2 M / (M + 1)
```

## MPSK vs MFSK

| 방식 | 장점 | 단점 |
|---|---|---|
| MPSK | bandwidth efficient | M 증가 시 symbol 간 거리 감소, power inefficient |
| MFSK | power efficient | M 증가 시 필요한 bandwidth 증가 |

MPSK는 bandwidth-limited channel에 유리하고, MFSK는 power-limited channel에서 장점이 있다.

## Shannon Limit

Shannon limit은 reliable communication이 가능한 이론적 최소 `Eb/N0`가 약 `-1.6 dB`임을 보여준다.

해석:

- 이 한계보다 낮은 signal power에서는 어떤 coding을 써도 reliable communication을 보장할 수 없다.
- 실제 통신 시스템의 목표는 BER curve를 이상적인 Shannon limit에 가깝게 만드는 것이다.

## MPSK Error Performance

MPSK 수신 vector:

```text
r = s_m + n
```

AWGN에서 equal prior이면 다음 기준들이 동등해진다.

- MAP
- ML
- minimum distance detection
- maximum correlator output

Gray coding을 사용하면 인접 symbol error가 1-bit error로 이어지도록 만들 수 있다.

중요 결과:

- BPSK와 QPSK는 같은 BER 성능을 가진다.
- QPSK는 in-phase BPSK와 quadrature BPSK 두 개로 볼 수 있다.

## MFSK Error Performance

MFSK는 M개의 orthogonal dimension을 사용한다.

수신기는 각 correlator output을 비교하여 가장 큰 것을 선택한다.

M 증가 효과:

- `P_E vs SNR`: M이 커지면 SER이 커질 수 있다.
- `P_E vs Eb/N0`: M이 커지면 energy efficiency가 좋아질 수 있다.

SER과 BER의 관계는 M이 커질수록 대략 다음 형태로 수렴한다.

```text
P_b ≈ (1/2) P_s
```

## 시험 포인트

- BPSK, BFSK, DPSK, noncoherent BFSK의 BER 식을 비교한다.
- antipodal, orthogonal, identical signaling을 `rho`로 해석한다.
- MPSK는 bandwidth 효율, MFSK는 power 효율에서 장점이 있음을 이해한다.
- BPSK와 QPSK의 BER이 같은 이유를 I/Q 분해로 설명할 수 있어야 한다.
- Gray coding이 BER을 낮추는 구조적 이유를 안다.

## 같이 보면 좋은 노트

- [[03 Bandpass Transmission - 디지털 변조와 검파]]
- [[04 Noise and Decision - 잡음과 최적 검출]]
- [[08 QPSK vs 16QAM - 대역폭과 BER 비교]]
