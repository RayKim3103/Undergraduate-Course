---
과목: Digital Communications
유형: Project Report
주제: Rayleigh fading, QPSK BER, Hamming coding, 16-QAM audio receiver
tags:
  - digital-communications
  - project-report
  - rayleigh-fading
  - ber
  - hamming-code
  - equalization
---

# Project Report - Rayleigh Fading 음성 통신 구현

## 핵심 요약

이 보고서는 Rayleigh fading 채널을 설명하고, QPSK BER 시뮬레이션, `(7,4) Hamming code`의 coding gain, 16-QAM 음성 복원 수신기, 복원 신호 재송신기, 2-tap 채널 개선 방법을 다룬다.

보고서의 중심 결론은 다음과 같다.

- AWGN 채널에서는 `Eb/N0`가 증가할수록 BER이 급격히 감소한다.
- Rayleigh fading 채널에서는 다중경로에 따른 amplitude/phase 왜곡 때문에 BER 개선이 더 완만하다.
- Hamming coding은 redundancy를 사용해 BER을 낮추며, 특히 fading 채널에서 단순 power 증가 외의 개선 수단이 된다.
- 1-tap equalization과 16-QAM minimum distance detection, Hamming decoding, dequantization을 결합하면 손상된 음성 데이터를 복원할 수 있다.

## Rayleigh Fading Channel

### Large-scale Fading

Large-scale fading은 송수신기 사이의 거리가 길어지거나 큰 장애물에 의해 나타나는 장기적인 신호 감쇠이다.

대표 요인:

- Path loss: 거리가 증가할수록 수신 전력이 감소한다.
- Shadowing: 건물, 산, 지형지물 등 큰 장애물이 신호를 가려 수신 전력이 불규칙하게 변한다.

### Small-scale Fading

Small-scale fading은 수신기 주변의 반사/산란체 위치 변화 때문에 짧은 거리에서도 수신 신호 전력이 빠르게 변하는 현상이다.

대표 모델:

| 모델 | 조건 | 특징 |
|---|---|---|
| Rician | 직접 경로가 있음 | LOS 성분 + 산란 성분 |
| Rayleigh | 직접 경로가 없음 | 산란/반사 경로들의 합 |

Rayleigh fading에서는 여러 산란 성분의 합이 복소 Gaussian처럼 모델링되고, envelope는 Rayleigh 분포를 따른다.

## Multipath Channel Impulse Response

다중경로 채널은 여러 지연 경로의 합으로 표현된다.

```text
c(tau; t) = sum_i alpha_i(t) delta(tau - tau_i(t))
```

의미:

- `alpha_i(t)`: i번째 경로의 복소 감쇠 계수
- `tau_i(t)`: i번째 경로의 지연
- 시간 `t`에 따라 채널이 변할 수 있으므로 시변 채널이다.

1-tap Rayleigh fading은 discrete domain에서 다음처럼 단순화할 수 있다.

```text
h[n] = alpha_0 delta[n]
y[n] = h x[n] + w[n]
```

## QPSK BER 비교

### AWGN Channel

AWGN 채널에서는 신호에 Gaussian 잡음만 더해진다.

```text
y = x + n
```

따라서 `Eb/N0`가 증가하면 noise 대비 bit energy가 커지고 BER이 빠르게 감소한다.

### 1-tap Rayleigh Fading Channel

Rayleigh fading에서는 수신 신호가 다음처럼 표현된다.

```text
y = h x + n
```

채널을 알고 있을 때 equalization을 하면 다음과 같다.

```text
x_hat = y / h = x + n / h
```

주의할 점은 `|h|`가 작을 때 잡음이 크게 증폭된다는 것이다. 따라서 송신단의 `Eb/N0`가 높아져도 수신단에서 경험하는 instantaneous SNR은 채널 상태에 따라 크게 흔들린다.

## Hamming Code 성능 분석

보고서는 `(7,4) Hamming code`를 사용하여 coded BER과 uncoded BER을 비교한다.

### 핵심 구조

```text
4-bit message -> 7-bit codeword
```

4개의 정보 비트에 3개의 parity 비트를 추가하여 1-bit error correction을 수행한다.

### BER 결과 해석

Rayleigh 채널 예시 결과:

| Eb/N0 | Coded BER | Uncoded BER |
|---:|---:|---:|
| 1 dB | 0.1034 | 0.1267 |
| 4 dB | 0.0498 | 0.0774 |
| 7 dB | 0.0214 | 0.0434 |
| 10 dB | 0.0086 | 0.0235 |

AWGN 채널 예시 결과:

| Eb/N0 | Coded BER | Uncoded BER |
|---:|---:|---:|
| 1 dB | 0.02494 | 0.0561 |
| 4 dB | 0.00138 | 0.0124 |
| 7 dB | 약 `7e-6` | 0.0067 |
| 10 dB | 매우 작음 | 약 `5e-6` |

해석:

- coded BER은 uncoded BER보다 전반적으로 낮다.
- coding은 bandwidth를 더 쓰는 대신 error correction 능력을 얻는다.
- 같은 BER을 얻기 위해 필요한 `Eb/N0`가 줄어드는 효과가 coding gain이다.
- Rayleigh fading에서는 단순히 power를 키우는 것보다 coding/equalization의 조합이 중요하다.

## 음성 데이터 수신기

보고서에서 복원한 곡은 Black Pink의 "Shut Down"으로 기록되어 있다.

수신 과정:

```text
received symbol
-> equalization
-> 16-QAM demodulation
-> syndrome decoding
-> dequantization
-> reconstruction
```

### 1. Equalization

1-tap 채널에서는 수신 심볼을 estimated channel로 나눠 보상한다.

```text
x_hat = y / h_hat
```

### 2. 16-QAM Demodulation

equalized symbol과 16-QAM constellation point 사이의 거리를 계산하여 가장 가까운 점을 선택한다.

이때 Gray coding을 적용하면 인접 심볼 오류가 발생해도 bit error 수를 줄일 수 있다.

### 3. Error Correction and Decoding

수신 bitstream을 7-bit씩 나누고 syndrome을 계산한다.

- syndrome = 0: 오류 없음
- syndrome이 `H^T`의 특정 열과 일치: 해당 위치의 1-bit error correction
- correction 후 parity bit를 제거하고 4-bit message를 복원

### 4. Dequantization

5-bit 단위로 0부터 31까지의 level을 얻고, `[-1, 1]` 범위로 복원한다.

```text
analog = 2 * level / 31 - 1
```

sampling frequency는 `44.1 kHz`이다.

## 복원 신호 재송신

보고서는 복원된 음성을 다시 송신하는 송신기를 설계한다.

송신기:

```text
audio
-> 5-bit uniform quantization
-> (7,4) Hamming encoding
-> 16-QAM Gray-coded modulation
-> 1-tap Rayleigh channel
-> AWGN at Eb/N0 = 10 dB
```

재수신 결과는 첫 번째 복원보다 잡음이 더 커진다고 분석한다. 이유는 다음과 같다.

- 첫 번째 복원 신호에도 residual noise가 남아 있다.
- 새 AWGN이 추가된다.
- equalization 과정에서 `1/h`가 noise 성분도 함께 키울 수 있다.

## 2-tap Channel 개선 아이디어

2-tap 채널은 다음처럼 모델링된다.

```text
h[n] = alpha_0 delta[n] + alpha_1 delta[n-1]
```

이는 이전 심볼이 현재 심볼에 영향을 주는 ISI를 유발한다.

보고서의 개선 방향:

- 주파수 영역에서 `H(e^jw) = alpha_0 + alpha_1 e^{-jw}`를 추정한다.
- `1/H(e^jw)` 형태의 equalizer를 설계한다.
- transversal equalizer를 사용해 ISI를 줄인다.
- pilot signal로 채널 상태를 추정한 뒤 equalization을 적용한다.

## 같이 보면 좋은 노트

- [[00 Final Project - 디지털 통신 시스템 시뮬레이션]]
- [[05 Error Performance - Bandpass BER 성능]]
- [[06 Channel Coding - 오류 제어 부호]]
- [[09 Channel Model - Multipath Fading과 Equalization]]
