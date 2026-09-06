---
title: "04. Sampling Rate Change"
pages: 14
tags: [DSP, lecture-note, sampling, multirate]
---

# 04. Sampling Rate Change

> 이전: [[03 Review of Signals and Systems]]
> 다음: [[05 LTI System to Linear Algebra]]

## 학습 목표

이 자료는 discrete-time processing으로 sampling rate를 바꾸는 방법을 다룬다.

- down-sampling/decimation
- anti-aliasing filter
- up-sampling/interpolation
- reconstruction/interpolation filter
- 비정수 배율 sampling rate 변환

## Sampling Rate Reduction

### Down-sampling

정수 $M$만큼 샘플링률을 낮추는 decimator는 다음과 같이 정의된다.

$$
y[n] = x[nM]
$$

즉 원래 수열에서 $M$번째 샘플만 남긴다.

### 주파수 영역 효과

down-sampling은 주파수축에서 spectrum을 $M$배 확장하고, $2\pi$ 주기 때문에 겹쳐 더해지는 효과를 만든다.

$$
Y(e^{j\omega})
= \frac{1}{M}\sum_{k=0}^{M-1}
X\left(e^{j(\omega-2\pi k)/M}\right)
$$

따라서 원 신호가 충분히 bandlimited가 아니면 aliasing이 발생한다.

## Anti-Aliasing Filter

decimation 전에 반드시 low-pass filtering을 해야 한다.

이상적인 조건:

$$
H_{aa}(e^{j\omega}) =
\begin{cases}
1, & |\omega|\le \pi/M \\
0, & \pi/M < |\omega| \le \pi
\end{cases}
$$

절차:

1. 원 신호 $x[n]$을 low-pass filter에 통과시킨다.
2. cutoff를 $\pi/M$ 이하로 제한한다.
3. 그 결과를 $M$배 down-sampling한다.

핵심은 decimation 자체가 aliasing을 없애지 못한다는 점이다. aliasing을 막는 역할은 앞단의 anti-aliasing filter가 담당한다.

## Increasing Sampling Rate

### Up-sampling

정수 $L$배 sampling rate를 높일 때는 샘플 사이에 $L-1$개의 0을 삽입한다.

$$
x_u[n] =
\begin{cases}
x[n/L], & n = 0,\pm L,\pm 2L,\ldots \\
0, & \text{otherwise}
\end{cases}
$$

### 주파수 영역 효과

zero insertion은 시간축을 늘리므로 주파수 영역에서는 spectrum image가 생긴다.

$$
X_u(e^{j\omega}) = X(e^{j\omega L})
$$

원 spectrum이 $L$개 image로 반복되어 보이므로, interpolation filter가 필요하다.

## Interpolation Filter

up-sampling 후 low-pass filter로 image를 제거한다.

이상적 interpolation filter:

$$
H_i(e^{j\omega}) =
\begin{cases}
L, & |\omega|\le \pi/L \\
0, & \pi/L < |\omega| \le \pi
\end{cases}
$$

gain이 $L$인 이유는 zero insertion으로 평균 에너지가 희석된 것을 보상하기 위해서이다.

## 비정수 배율 Sampling Rate 변환

샘플링률을 $L/M$배로 바꾸려면 다음 순서를 사용한다.

$$
x[n] \xrightarrow{\uparrow L}
x_u[n] \xrightarrow{\text{LPF}}
v[n] \xrightarrow{\downarrow M}
y[n]
$$

low-pass filter cutoff는 두 조건을 동시에 만족해야 한다.

$$
\omega_c = \min\left(\frac{\pi}{L}, \frac{\pi}{M}\right)
$$

실제 구현에서는 interpolation filter와 anti-aliasing filter를 하나의 low-pass filter로 합쳐 계산량을 줄인다.

## Simulation에서 관찰할 점

강의자료의 simulation 그림들은 다음을 보여준다.

- filtering 없이 down-sampling하면 spectrum이 겹쳐 aliasing이 생긴다.
- anti-aliasing filter를 먼저 적용하면 down-sampling 후에도 원래 저주파 성분이 보존된다.
- up-sampling 직후에는 샘플 사이에 0이 들어간 형태라 실제 interpolation된 부드러운 신호가 아니다.
- interpolation low-pass filter를 거쳐야 image spectrum이 제거되고 자연스러운 고샘플률 신호가 된다.

## 체크포인트

- decimation 전: anti-aliasing filter
- interpolation 후: anti-imaging filter
- down-sampling은 spectrum folding/aliasing 위험을 만든다.
- up-sampling은 spectrum image를 만든다.
- 비정수 변환은 항상 “먼저 up-sample, filter, 나중에 down-sample” 순서로 생각한다.
