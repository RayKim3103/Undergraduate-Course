---
과목: Digital Communications
유형: Lecture Note
주제: Baseband transmission, sampling, PCM, quantization, ISI
tags:
  - digital-communications
  - baseband
  - sampling
  - pcm
  - quantization
  - isi
---

# Baseband Transmission - 샘플링 PCM Quantization ISI

## 핵심 요약

이 강의는 아날로그 정보를 baseband digital signal로 바꾸고 이를 pulse waveform으로 전송하는 과정을 다룬다. 핵심 흐름은 sampling, quantization, coding, waveform encoding이며, 실제 필터 때문에 발생하는 ISI와 이를 줄이기 위한 Nyquist 조건 및 raised cosine filter도 소개한다.

```text
analog signal
-> sampling
-> quantization
-> coding
-> PCM bit stream
-> pulse waveform
-> baseband channel
```

## Baseband System

Baseband transmission은 정보를 carrier로 올리기 전 또는 carrier 없이 low-frequency 영역에서 전송하는 방식이다.

송신 쪽 주요 블록:

- sample
- quantizer
- coder
- waveform encoder
- baseband modulator

수신 쪽 주요 블록:

- waveform detector
- decoder
- low-pass filter

## Messages, Characters, Symbols

디지털 통신에서는 정보를 bit로 바꾼 뒤 symbol 단위로 묶어 waveform에 mapping한다.

```text
M = symbol 수
k = 한 symbol이 표현하는 bit 수
M = 2^k
```

예시:

- `k = 1`: `M = 2`, binary system
- `k = 2`: `M = 4`, quaternary system
- `k = 3`: `M = 8`, 8-ary system

M-ary system에서는 `M`개의 서로 다른 waveform이 필요하다.

## Sampling Theorem

### Uniform Sampling Theorem

대역 제한 신호가 `f_m` Hz 이상의 성분을 갖지 않으면, 다음 조건에서 원 신호를 유일하게 복원할 수 있다.

```text
f_s >= 2 f_m
T_s <= 1 / (2 f_m)
```

용어:

- `f_s`: sampling rate
- `f_m`: signal bandwidth 또는 최대 주파수
- `2 f_m`: Nyquist rate

## Impulse Sampling

이상적인 sampling은 원 신호에 impulse train을 곱하는 것으로 표현된다.

```text
x_s(t) = sum_n x(nT_s) delta(t - nT_s)
```

주파수 영역에서는 원래 spectrum이 `f_s` 간격으로 반복된다.

```text
X_s(f) = (1 / T_s) sum_n X(f - n f_s)
```

## Aliasing

`f_s < 2 f_m`이면 반복된 spectrum이 겹치고, 서로 다른 주파수 성분이 같은 샘플 sequence로 보이게 된다. 이것이 aliasing이다.

aliasing 방지 방법:

- sampling rate를 충분히 크게 한다.
- sampling 전에 prefiltering으로 입력 대역을 제한한다.
- sampling 후 postfiltering을 적용한다.

실제 filter는 ideal하지 않으므로 보통 `f_s = 2 f_m`보다 여유 있게 sampling한다.

## Pulse Code Modulation

PCM은 아날로그 amplitude를 discrete level로 quantization한 뒤 bit stream으로 표현하는 방식이다.

흐름:

```text
sampled value x[n]
-> quantized value x_hat[n]
-> binary code
```

PCM은 analog signal을 digital bit stream으로 바꾸는 핵심 format conversion이다.

## Quantization

Quantization은 sampled amplitude를 유한한 대표값 중 하나로 표현하는 과정이다.

```text
x[n] in R
x_hat[n] in {x_1, x_2, ..., x_L}
```

### Uniform vs Nonuniform Quantization

| 방식 | 특징 | 적합한 경우 |
|---|---|---|
| Uniform | 모든 구간 간격이 동일 | 신호 amplitude 분포가 균일할 때 |
| Nonuniform | 특정 amplitude 영역을 더 촘촘히 표현 | 음성처럼 작은 magnitude가 자주 나올 때 |

음성 신호는 낮은 amplitude 값이 자주 나타나므로 nonuniform quantization이 유리할 수 있다.

### Quantization Error

uniform quantizer에서 quantization interval을 `q`라고 하면 error는 대략 `[-q/2, q/2]`에 균일하게 분포한다고 볼 수 있다.

평균 quantization noise power:

```text
sigma_e^2 = q^2 / 12
```

level 수가 `L`이면 peak signal power to average quantization noise power ratio는 대략 다음처럼 증가한다.

```text
SQNR_peak ≈ 3 L^2
```

즉 quantization level이 많을수록 quantization noise가 줄어든다.

## Baseband Transmission

PCM bit sequence를 baseband channel로 보내려면 electrical pulse waveform으로 바꾸어야 한다.

표현 방식:

- pulse의 유무로 1/0 표현
- pulse amplitude level로 여러 bit를 한 symbol로 표현
- pulse waveform 형태를 약속하여 symbol mapping

## Multi-level Baseband Transmission

Binary PCM은 bandwidth 요구량이 클 수 있다. 이를 줄이기 위해 여러 bit를 하나의 symbol로 묶는 multi-level signaling을 사용한다.

장점:

- 같은 bit rate에서 symbol rate를 낮출 수 있다.
- 필요한 bandwidth를 줄일 수 있다.

단점:

- symbol 간 거리가 줄어 noise에 약해진다.
- 같은 오류 성능을 얻으려면 더 큰 signal power가 필요하다.

따라서 power와 bandwidth 사이에 trade-off가 생긴다.

## 예제 흐름

아날로그 정보:

- `f_m = 3 kHz`
- `M = 16` level PCM
- quantization error가 peak-to-peak 범위의 1%보다 작아야 함

핵심 결과:

- 필요한 bits/sample: `l = 6`
- Nyquist sampling rate: `f_s = 2 f_m = 6000 samples/s`
- bit rate: `R = l f_s = 36000 bits/s`
- 16-level pulse라면 `k = log2 16 = 4` bit/symbol
- symbol rate: `R_s = R / 4 = 9000 symbols/s`

## Inter-Symbol Interference

ISI는 한 symbol의 pulse tail이 다른 symbol의 sampling 순간에 영향을 주는 현상이다.

이상적인 경우:

- 각 symbol이 impulse처럼 존재한다.
- sampling time에서 다른 symbol의 값은 0이다.

실제 경우:

- transmitting filter와 receiving filter 때문에 pulse가 퍼진다.
- sampling time `t = kT`에서 이웃 symbol의 값이 0이 아니면 간섭이 발생한다.

## Nyquist Criterion for Zero ISI

symbol rate가 `R_s = 1/T`일 때 ISI 없이 detection하려면 이상적인 경우 최소 system bandwidth가 필요하다.

```text
B_min = R_s / 2
```

이는 ideal low-pass filter에서 가능한 이론적 한계이다.

## Raised Cosine Filter

실제 시스템에서는 ideal filter를 만들 수 없으므로 raised cosine filter를 사용한다.

roll-off factor `r`에 대해 bandwidth는 다음과 같다.

```text
B = (1 + r) R_s / 2
```

해석:

- `r = 0`: minimum bandwidth, ideal case, `2 symbols/s/Hz`
- `r = 1`: bandwidth가 두 배가 되어 `1 symbol/s/Hz`

## 시험 포인트

- sampling theorem과 aliasing 조건을 정확히 설명할 수 있어야 한다.
- quantization error variance `q^2/12`의 의미를 이해한다.
- multi-level signaling은 bandwidth를 줄이는 대신 power 요구량을 키운다.
- ISI는 pulse shaping과 sampling time의 문제로 이해한다.
- raised cosine filter의 roll-off factor가 bandwidth에 미치는 영향을 기억한다.

## 같이 보면 좋은 노트

- [[01 Basic of Communications - 디지털 통신 개요]]
- [[03 Bandpass Transmission - 디지털 변조와 검파]]
- [[04 Noise and Decision - 잡음과 최적 검출]]

