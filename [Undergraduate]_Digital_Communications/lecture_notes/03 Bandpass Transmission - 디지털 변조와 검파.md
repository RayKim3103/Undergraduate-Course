---
과목: Digital Communications
유형: Lecture Note
주제: Bandpass modulation, PSK, FSK, ASK, QAM, vector space, coherent/noncoherent detection
tags:
  - digital-communications
  - bandpass
  - psk
  - fsk
  - qam
  - coherent-detection
  - noncoherent-detection
---

# Bandpass Transmission - 디지털 변조와 검파

## 핵심 요약

이 강의는 디지털 정보를 passband carrier의 amplitude, phase, frequency 변화로 표현하는 방법과, 수신 신호를 vector space에서 검출하는 방법을 다룬다. PSK, FSK, ASK, APK/QAM의 기본 신호 모델을 비교하고, matched filter/correlator, coherent detection, noncoherent detection의 차이를 설명한다.

## 변조 방식 분류

### Linearity에 따른 분류

| 분류 | 의미 | 예 |
|---|---|---|
| Linear modulation | 변조 신호 사이에 superposition이 적용됨 | PSK |
| Non-linear modulation | superposition이 직접 적용되지 않음 | FSK |

### Memory에 따른 분류

| 분류 | 의미 | 예 |
|---|---|---|
| Memoryless | 현재 symbol이 이전 data와 독립 | BPSK |
| Memory | 현재 symbol이 이전 symbol에 의존 | DBPSK |

DBPSK에서는 절대 phase가 아니라 이전 symbol과의 phase difference에 정보를 싣는다.

## Bandpass Modulation

Bandpass modulation은 정보 신호를 baseband에서 원하는 passband로 옮기는 과정이다.

일반적인 bandpass 신호:

```text
s(t) = A(t) cos(omega_i t + phi(t))
```

정보를 싣는 위치에 따라 다음처럼 나뉜다.

- amplitude 변화: ASK, AM
- phase 변화: PSK
- frequency 변화: FSK
- amplitude + phase 변화: QAM/APK

## Digital Modulation Schemes

### PSK

PSK는 phase를 바꾸어 symbol을 표현한다.

```text
s_i(t) = sqrt(2E/T) cos(omega_0 t + phi_i), 0 <= t <= T
phi_i = 2 pi i / M
```

BPSK:

- bit 0: phase 0
- bit 1: phase pi
- 두 신호는 antipodal 관계이다.

### FSK

FSK는 서로 다른 carrier frequency로 symbol을 표현한다.

```text
s_i(t) = sqrt(2E/T) cos(omega_i t + phi), 0 <= t <= T
```

특징:

- 서로 다른 frequency 성분을 orthogonal하게 배치한다.
- BFSK는 두 개의 직교 basis function으로 표현된다.
- MFSK는 보통 M차원 신호공간을 사용한다.

### ASK

ASK는 amplitude level로 symbol을 표현한다.

```text
s_i(t) = sqrt(2E_i/T) cos(omega_0 t + phi)
```

Binary ASK의 대표 예는 on-off keying이다.

### APK and QAM

APK는 amplitude와 phase를 모두 바꾼다. QAM은 직교하는 in-phase와 quadrature 성분을 이용하여 사각형 constellation을 만든다.

예:

- 16-QAM
- 64-QAM

QAM은 bandwidth efficiency가 좋지만 constellation point 간 거리가 좁아질수록 noise에 민감하다.

## 신호의 Vectorial View

신호는 직교 basis function의 선형결합으로 표현할 수 있다.

```text
s_i(t) = sum_j a_ij psi_j(t)
a_ij = integral s_i(t) psi_j(t) dt / K_j
```

직교 조건:

```text
integral_0^T psi_j(t) psi_k(t) dt = K_j delta_jk
```

신호의 energy:

```text
E_i = integral_0^T s_i^2(t) dt
```

직교 basis를 사용하면 energy가 vector 성분의 제곱합으로 표현된다.

## Minimum Error Decision in Vector Space

AWGN, equal prior probability 조건에서 최적 결정은 가장 가까운 signal point를 선택하는 것이다.

```text
choose s_i that minimizes ||r - s_i||
```

수신 과정:

```text
r(t)
-> matched filter 또는 correlator
-> vector point r
-> decision region 판정
```

## Matched Filter and Correlator

Matched filter는 sampling time에서 SNR을 최대화하는 filter이다.

```text
h_opt(t) = s(T - t)
```

수신 신호와 기준 신호의 상관값을 계산하는 correlator와 sampling time에서 같은 출력을 준다.

```text
z(T) = integral_0^T r(t) s(t) dt
```

M개의 waveform을 직접 비교할 수도 있지만, 신호공간의 basis 수 `N`이 작으면 `N`개의 correlator만으로 충분하다.

## Coherent Detection

Coherent detection은 carrier phase와 symbol timing을 정확히 알고 있다고 가정한다.

### BPSK Coherent Detection

BPSK basis:

```text
psi_1(t) = sqrt(2/T) cos(omega_0 t)
s_1(t) = sqrt(E) psi_1(t)
s_2(t) = -sqrt(E) psi_1(t)
```

correlator output `z(T)`를 threshold와 비교한다.

### MPSK Coherent Detection

MPSK는 두 basis로 표현된다.

```text
psi_1(t) = sqrt(2/T) cos(omega_0 t)
psi_2(t) = sqrt(2/T) sin(omega_0 t)
```

수신기는 in-phase projection `X`와 quadrature projection `Y`를 계산한다.

```text
X = integral r(t) psi_1(t) dt
Y = integral r(t) psi_2(t) dt
```

그 뒤 추정 phase를 constellation phase와 비교해 가장 가까운 symbol로 결정한다.

### FSK Coherent Detection

MFSK는 보통 `M`개의 orthogonal frequency를 사용한다.

coherent FSK의 최소 tone spacing:

```text
Delta f_min = 1 / (2T)
```

수신기는 M개의 correlator output 중 가장 큰 값을 선택한다.

## Noncoherent Detection

Noncoherent detection은 carrier phase를 정확히 모르는 상황에서 사용한다.

필요성:

- 이동통신 채널에서 phase ambiguity가 발생한다.
- QPSK constellation이 회전할 수 있다.
- FSK의 orthogonality가 손상될 수 있다.

### DPSK

DPSK는 절대 phase가 아니라 연속된 두 symbol의 phase difference를 검출한다.

```text
received phase at 2T - received phase at T
```

공통 phase ambiguity가 두 symbol 사이에서 거의 일정하면 차분 과정에서 사라진다.

일반적으로 DPSK는 BPSK보다 BER 성능이 약 3 dB 나쁘다. 두 수신 신호를 비교하므로 noise가 더 크게 영향을 주기 때문이다.

### Noncoherent FSK

Noncoherent FSK는 energy detector 또는 envelope detector를 사용한다.

특징:

- phase 정보가 필요 없다.
- coherent detection보다 구현이 쉽다.
- 같은 orthogonality를 유지하려면 더 넓은 tone spacing이 필요하다.

noncoherent FSK 최소 tone spacing:

```text
Delta f_min = 1 / T
```

따라서 coherent FSK가 noncoherent FSK보다 bandwidth 효율이 2배 좋다.

## 시험 포인트

- PSK, FSK, ASK, QAM이 carrier의 어떤 요소를 바꾸는지 비교한다.
- MPSK는 2차원 I/Q basis로 표현된다.
- MFSK는 M차원 orthogonal basis가 필요하다.
- matched filter와 correlator가 sampling time에서 동등하다는 점을 이해한다.
- coherent FSK와 noncoherent FSK의 tone spacing 차이 `1/(2T)` vs `1/T`를 기억한다.

## 같이 보면 좋은 노트

- [[04 Noise and Decision - 잡음과 최적 검출]]
- [[05 Error Performance - Bandpass BER 성능]]
- [[08 QPSK vs 16QAM - 대역폭과 BER 비교]]
