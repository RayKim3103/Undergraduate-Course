---
title: "03. Review of Signals and Systems"
pages: 32
tags: [DSP, lecture-note, signals, systems, DTFT, sampling]
---

# 03. Review of Signals and Systems

> 이전: [[02 DSP Introduction]]
> 다음: [[04 Sampling Rate Change]]

## 자료의 범위

이 강의는 DSP 본론에 들어가기 전 Signals and Systems의 핵심을 복습한다.

- 기본 이산시간 수열과 연산
- discrete-time system의 성질
- LTI system과 convolution
- DTFS, DTFT
- DTFT의 주요 성질
- sampling/reconstruction
- continuous-time signal의 discrete-time processing

## 기본 수열

### Unit Sample

$$
\delta[n] =
\begin{cases}
1, & n=0 \\
0, & n\neq0
\end{cases}
$$

모든 이산 신호는 shifted impulse의 선형결합으로 표현된다.

$$
x[n] = \sum_{k=-\infty}^{\infty} x[k]\delta[n-k]
$$

### Unit Step

$$
u[n] =
\begin{cases}
1, & n\ge 0 \\
0, & n<0
\end{cases}
$$

관계식:

$$
u[n] = \sum_{k=-\infty}^{n}\delta[k],
\qquad
\delta[n] = u[n]-u[n-1]
$$

### Complex Exponential

$$
x[n] = Ae^{j\omega_0 n}
$$

이산시간 복소지수는 LTI 시스템의 eigenfunction이다. 즉 입력이 $e^{j\omega n}$이면 출력도 같은 주파수의 복소지수이고, 크기와 위상만 바뀐다.

## Discrete-Time System의 성질

### Memory

출력 $y[n]$이 같은 시점의 입력 $x[n]$에만 의존하면 memoryless이다. 과거/미래 입력에 의존하면 memory가 있다.

### Linearity

시스템 $T$가 선형이면 다음을 만족한다.

$$
T\{a x_1[n] + b x_2[n]\}
= aT\{x_1[n]\}+bT\{x_2[n]\}
$$

### Time Invariance

입력을 $n_0$만큼 지연하면 출력도 동일하게 지연되어야 한다.

$$
x[n] \rightarrow y[n]
\quad\Rightarrow\quad
x[n-n_0] \rightarrow y[n-n_0]
$$

### Causality

출력 $y[n]$이 현재와 과거 입력에만 의존하면 causal이다.

### Stability

BIBO 안정성:

$$
|x[n]| \le B_x < \infty
\quad\Rightarrow\quad
|y[n]| \le B_y < \infty
$$

## LTI System과 Convolution

LTI 시스템은 impulse response $h[n]$으로 완전히 결정된다.

$$
y[n] = \sum_{k=-\infty}^{\infty} x[k]h[n-k]
= x[n]*h[n]
$$

### Convolution 계산 절차

1. $h[k]$를 원점 기준으로 뒤집어 $h[-k]$를 만든다.
2. $n$만큼 이동시켜 $h[n-k]$를 만든다.
3. $x[k]h[n-k]$를 곱한다.
4. 모든 $k$에 대해 합산한다.

### LTI 안정성과 인과성

- causal LTI: $h[n]=0$ for $n<0$
- BIBO stable LTI:

$$
\sum_{n=-\infty}^{\infty}|h[n]| < \infty
$$

FIR 시스템은 impulse response 길이가 유한하므로 일반적으로 안정하다. IIR 시스템은 pole/ROC에 따라 안정성이 결정된다.

## Frequency-Domain Representation

LTI 시스템에 복소지수를 넣으면:

$$
x[n]=e^{j\omega n}
$$

$$
y[n]
= \sum_{k=-\infty}^{\infty} h[k]e^{j\omega(n-k)}
= e^{j\omega n}\sum_{k=-\infty}^{\infty}h[k]e^{-j\omega k}
$$

따라서

$$
H(e^{j\omega}) = \sum_{k=-\infty}^{\infty}h[k]e^{-j\omega k}
$$

는 frequency response이며 eigenvalue 역할을 한다.

## DTFS

주기 $N$인 이산시간 신호는 discrete-time Fourier series로 표현된다.

합성식:

$$
x[n] = \sum_{k=0}^{N-1} a_k e^{j\frac{2\pi}{N}kn}
$$

분석식:

$$
a_k = \frac{1}{N}\sum_{n=0}^{N-1}x[n]e^{-j\frac{2\pi}{N}kn}
$$

## DTFT

비주기 이산시간 신호의 주파수 표현:

$$
X(e^{j\omega}) = \sum_{n=-\infty}^{\infty}x[n]e^{-j\omega n}
$$

역변환:

$$
x[n] = \frac{1}{2\pi}\int_{-\pi}^{\pi}X(e^{j\omega})e^{j\omega n}d\omega
$$

DTFT는 $\omega$에 대해 $2\pi$ 주기이다.

## DTFT 성질

| 성질 | 시간 영역 | 주파수 영역 |
|---|---|---|
| 선형성 | $ax_1[n]+bx_2[n]$ | $aX_1+bX_2$ |
| convolution | $x[n]*h[n]$ | $X(e^{j\omega})H(e^{j\omega})$ |
| 시간 이동 | $x[n-n_0]$ | $e^{-j\omega n_0}X(e^{j\omega})$ |
| 변조 | $e^{j\omega_0n}x[n]$ | $X(e^{j(\omega-\omega_0)})$ |
| 시간 반전 | $x[-n]$ | $X(e^{-j\omega})$ |
| Parseval | $\sum|x[n]|^2$ | $\frac1{2\pi}\int |X|^2d\omega$ |

## Frequency Selective Filter

이상적 low-pass filter:

$$
H_d(e^{j\omega}) =
\begin{cases}
1, & |\omega| \le \omega_c \\
0, & \omega_c < |\omega| \le \pi
\end{cases}
$$

이상적 응답의 impulse response는 sinc 형태이고 무한 길이이다. 실제 FIR 필터 설계에서는 window를 곱해 유한 길이로 자른다.

## Sampling과 Reconstruction

연속시간 신호 $x_c(t)$를 주기 $T$로 샘플링하면

$$
x[n] = x_c(nT)
$$

샘플링 각주파수:

$$
\Omega_s = \frac{2\pi}{T}
$$

Nyquist 조건:

$$
\Omega_s > 2\Omega_N
$$

여기서 $\Omega_N$은 원 신호의 최고 주파수이다. 조건을 만족하지 않으면 spectrum replica가 겹쳐 aliasing이 발생한다.

## 재구성

이상적 reconstruction filter는 샘플 스펙트럼의 중심 replica만 통과시킨다. 시간 영역에서는 sinc interpolation으로 볼 수 있다.

$$
x_c(t) = \sum_{n=-\infty}^{\infty}x[n]\operatorname{sinc}\left(\frac{t-nT}{T}\right)
$$

## 체크포인트

- LTI 시스템은 impulse response와 convolution으로 완전히 기술된다.
- convolution은 주파수 영역에서 곱셈이므로 필터 해석의 핵심 도구이다.
- DTFT는 연속 주파수 함수이고 $2\pi$ 주기이다.
- 샘플링은 주파수 영역에서 spectrum replica를 만들며, replica가 겹치면 aliasing이다.
- 이후 강의의 z-transform, DFT, Wiener filter는 모두 이 복습 내용을 기반으로 한다.
