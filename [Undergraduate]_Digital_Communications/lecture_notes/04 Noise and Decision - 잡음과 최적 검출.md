---
과목: Digital Communications
유형: Lecture Note
주제: Noise, spectral density, Gaussian decision, MAP/ML, matched filter
tags:
  - digital-communications
  - noise
  - awgn
  - detection
  - map
  - ml
  - matched-filter
---

# Noise and Decision - 잡음과 최적 검출

## 핵심 요약

이 강의는 디지털 수신기에서 잡음을 확률적으로 모델링하고, 수신 신호를 threshold와 비교해 bit를 결정하는 과정을 다룬다. 에너지/전력 스펙트럼 밀도, bandpass noise의 I/Q 표현, Gaussian random process, Bayes decision, MAP/ML decision, matched filter, unipolar/bipolar signaling의 BER이 핵심이다.

## Digital Signal Detection의 두 단계

수신기는 보통 두 단계를 거친다.

```text
r(t)
-> linear filter h(t)
-> sample at t = nT
-> threshold comparison
-> decision
```

수신 신호:

```text
r(t) = s_i(t) + n(t)
```

filter 출력:

```text
z(t) = a_i(t) + n_0(t)
```

sampling 후 `z(nT)`가 어떤 신호 평균 주변에 있는지 보고 decision을 한다.

## Spectral Density

### Energy Spectral Density

total energy가 finite한 pulse-like signal에서는 energy spectral density를 사용한다.

```text
E_g = integral |g(t)|^2 dt
```

시스템 `G(w) = F(w)H(w)`에서 출력 에너지 스펙트럼은 magnitude response에 의해 결정된다.

```text
|G(w)|^2 = |F(w)|^2 |H(w)|^2
```

### Power Spectral Density

시간상 계속 존재하는 power signal에는 power spectral density를 사용한다.

```text
P = lim_{T->infty} (1/T) integral_{-T/2}^{T/2} |f(t)|^2 dt
```

PSD는 주파수별 power density이며, 적분하면 평균 전력을 얻는다.

```text
P = integral S_f(f) df
```

### Autocorrelation과 PSD

PSD는 autocorrelation function의 Fourier transform 관계로 구할 수 있다.

```text
R_f(tau) = lim_{T->infty} (1/T) integral f*(t) f(t + tau) dt
S_f(w) <-> R_f(tau)
```

## Noise

Noise는 시스템 내부에 필연적으로 존재하는 unwanted electrical signal이다.

주요 값:

- mean value
- mean square value
- AC component
- noise PSD

## Bandpass Noise의 I/Q 표현

bandpass noise는 in-phase와 quadrature 성분으로 표현할 수 있다.

```text
n(t) = n_c(t) cos(2 pi f_c t) - n_s(t) sin(2 pi f_c t)
```

여기서:

- `n_c(t)`: in-phase noise
- `n_s(t)`: quadrature noise

이 표현을 사용하면 빠르게 진동하는 carrier 성분 대신 baseband envelope/phasor 관점에서 noise를 분석할 수 있다.

## Thermal Noise and White Noise

통신 시스템의 thermal noise는 관심 주파수 대역에서 PSD가 거의 일정하므로 white noise로 모델링한다.

two-sided PSD:

```text
S_n(f) = N_0 / 2
```

autocorrelation:

```text
R_n(tau) = (N_0 / 2) delta(tau)
```

low-pass white noise가 bandwidth `B`를 통과하면 noise power는 다음과 같다.

```text
P_n = N_0 B
```

## SNR

SNR은 평균 신호 제곱값과 평균 잡음 제곱값의 비율이다.

```text
SNR = signal power / noise power
SNR_dB = 10 log10(SNR)
```

## Gaussian Distribution and Q Function

Gaussian random variable:

```text
X ~ N(m, sigma^2)
```

Q function은 standard normal tail probability이다.

```text
Q(x) = P[X > x], X ~ N(0, 1)
Q(0) = 0.5
Q(-x) = 1 - Q(x)
```

통신의 BER 공식 대부분은 Gaussian tail probability로 귀결되므로 `Q(.)` 형태로 나타난다.

## Random Variable vs Random Process

| 개념 | 의미 |
|---|---|
| Random variable | 불확실한 사건을 하나의 숫자로 모델링 |
| Random process | 시간에 따라 변하는 불확실한 신호를 모델링 |

random process `X(t)`에서 특정 시간 `t_i`를 고정하면 `X(t_i)`는 random variable이 된다.

## Binary Signal Detection

수신 신호:

```text
z = a_i + n_0, i = 1, 2
```

`n_0`가 Gaussian이면 `z`는 평균이 `a_i`인 Gaussian random variable이다.

decision:

```text
z > gamma_0 -> H_1
z < gamma_0 -> H_2
```

prior가 같고 variance가 같으면 threshold는 두 평균의 중간이다.

```text
gamma_0 = (a_1 + a_2) / 2
```

## Bayes, MAP, ML

Bayes theorem:

```text
P(S_i | Z) = P(Z | S_i) P(S_i) / P(Z)
```

MAP criterion:

```text
choose S_i maximizing P(S_i | Z)
```

ML criterion:

prior가 같으면 `P(S_i)`가 동일하므로 MAP은 ML과 같다.

```text
choose S_i maximizing P(Z | S_i)
```

likelihood ratio test:

```text
L(Z) = P(Z | S_1) / P(Z | S_2)
```

## Bit Error Probability

binary detection에서 error probability:

```text
P_e = P(H_2 | S_1) P(S_1) + P(H_1 | S_2) P(S_2)
```

equal prior이면:

```text
P_e = Q((a_1 - a_2) / (2 sigma_0))
```

따라서 평균 separation이 클수록, noise standard deviation이 작을수록 BER이 낮아진다.

## Matched Filter

matched filter의 목적은 sampling time에서 SNR을 최대화하는 것이다.

최적 impulse response:

```text
h(t) = k s(T - t)
```

최대 출력 SNR:

```text
SNR_max = 2E / N_0
```

binary signaling에서 `s_1(t) - s_2(t)`에 matched된 filter를 사용하면 error probability를 줄이는 방향으로 `a_1 - a_2`를 키울 수 있다.

## Matched Filter와 Correlator의 동등성

sampling time `t = T`에서 matched filter output은 correlator output과 같다.

```text
z(T) = integral_0^T r(t) s(t) dt
```

따라서 수신기 구현은 matched filter bank 또는 correlator bank로 볼 수 있다.

## Unipolar vs Bipolar Signaling

### Unipolar

```text
s_1(t) = A, 0 <= t <= T
s_2(t) = 0, 0 <= t <= T
```

threshold:

```text
gamma_0 = A^2 T / 2
```

BER:

```text
P_B = Q(sqrt(E_b / N_0))
```

### Bipolar

```text
s_1(t) = A
s_2(t) = -A
```

threshold:

```text
gamma_0 = 0
```

BER:

```text
P_B = Q(sqrt(2E_b / N_0))
```

같은 BER을 얻기 위해 bipolar signaling은 unipolar signaling보다 약 3 dB의 power를 절약할 수 있다.

## 시험 포인트

- PSD와 autocorrelation의 관계를 이해한다.
- AWGN의 two-sided PSD `N_0/2`와 noise power `N_0 B`를 기억한다.
- MAP과 ML이 equal prior에서 같아지는 이유를 설명한다.
- binary Gaussian decision threshold가 평균의 중간이 되는 조건을 안다.
- matched filter가 SNR을 최대화하고 correlator와 동등하다는 점을 이해한다.
- unipolar와 bipolar BER 차이 및 3 dB 이득을 기억한다.

## 같이 보면 좋은 노트

- [[03 Bandpass Transmission - 디지털 변조와 검파]]
- [[05 Error Performance - Bandpass BER 성능]]
- [[10 Amplitude Modulation - AM DSB SSB VSB]]
