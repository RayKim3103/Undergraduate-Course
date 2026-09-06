---
title: "13. 2D Digital Signal Processing Overview"
pages: 13
tags: [DSP, lecture-note, 2D-DSP, image-processing]
---

# 13. 2D DSP Overview

> 이전: [[12 DFT and FFT]]

## 학습 목표

이 자료는 1차원 DSP 개념을 2차원 디지털 신호, 특히 영상 신호로 확장한다.

- 2D signal의 종류
- 2D digital image
- 2D system
- LSI system과 2D convolution
- DSFT
- 2D sampling/reconstruction
- image processing의 네 영역

## 2D Signals

### Analog 2D Signal

공간과 진폭이 모두 연속인 신호이다.

예:

- 필름 영상
- 실제 장면의 광 intensity
- 지진파 공간 분포
- radar/medical analog image

### Discrete-Space Signal

공간은 샘플링되어 이산이지만 진폭은 연속인 신호이다.

$$
x[n_1,n_2]
$$

analog image를 공간 sampling하면 얻어진다.

### Digital Signal

공간과 진폭이 모두 이산인 신호이다. 일반적인 디지털 이미지는 여기에 해당한다.

8-bit grayscale image:

$$
x[n_1,n_2]\in\{0,1,\ldots,255\}
$$

0은 가장 어두운 값, 255는 가장 밝은 값이다.

## 기본 2D 수열

### 2D Impulse

$$
\delta[n_1,n_2]=
\begin{cases}
1, & n_1=0,\ n_2=0\\
0, & \text{otherwise}
\end{cases}
$$

### Line Impulse

특정 행 또는 열 전체가 impulse처럼 활성화된 형태이다. 영상에서는 edge, scan line, projection을 설명할 때 유용하다.

### 2D Step

$$
u[n_1,n_2]=
\begin{cases}
1, & n_1\ge0,\ n_2\ge0\\
0, & \text{otherwise}
\end{cases}
$$

### Separable Sequence

2D 신호가 두 1D 신호의 곱으로 분해되면 separable이다.

$$
x[n_1,n_2]=x_1[n_1]x_2[n_2]
$$

separable filter는 계산량을 크게 줄일 수 있다.

## Digital Image와 Quantization

디지털 이미지는 pixel 또는 pel의 격자이다.

예:

- 512 x 512 pixels, 8 bits/pixel
- 256 gray levels

자료의 예시는 bit depth와 spatial resolution 변화가 영상 품질에 어떤 영향을 주는지 보여준다.

- bit 수 감소: intensity quantization artifact 증가
- pixel 수 감소: spatial detail 손실

## 2D System

2D 시스템은 입력 영상 $x[n_1,n_2]$를 출력 영상 $y[n_1,n_2]$로 mapping한다.

$$
y[n_1,n_2]=T\{x[n_1,n_2]\}
$$

## Linear and Shift-Invariant System

2D에서 LTI에 해당하는 개념을 LSI(linear shift-invariant) 또는 space-invariant system이라고 한다.

### Linearity

$$
T\{a x_1 + b x_2\}=aT\{x_1\}+bT\{x_2\}
$$

### Shift Invariance

입력이 공간적으로 shift되면 출력도 같은 양만큼 shift된다.

$$
x[n_1-m_1,n_2-m_2]
\rightarrow
y[n_1-m_1,n_2-m_2]
$$

## 2D Convolution

LSI 시스템은 impulse response 또는 PSF(point-spread function) $h[n_1,n_2]$로 완전히 결정된다.

$$
y[n_1,n_2]
=
\sum_{k_1=-\infty}^{\infty}
\sum_{k_2=-\infty}^{\infty}
x[k_1,k_2]h[n_1-k_1,n_2-k_2]
$$

축약:

$$
y=x*h
$$

영상 blur, sharpening, denoising filter는 모두 이 관점으로 표현할 수 있다.

## DSFT

2D discrete-space Fourier transform:

$$
X(e^{j\omega_1},e^{j\omega_2})
=
\sum_{n_1=-\infty}^{\infty}
\sum_{n_2=-\infty}^{\infty}
x[n_1,n_2]
e^{-j(\omega_1 n_1+\omega_2 n_2)}
$$

역변환:

$$
x[n_1,n_2]
=
\frac{1}{(2\pi)^2}
\int_{-\pi}^{\pi}\int_{-\pi}^{\pi}
X(e^{j\omega_1},e^{j\omega_2})
e^{j(\omega_1 n_1+\omega_2 n_2)}
d\omega_1d\omega_2
$$

2D LSI 시스템에서는

$$
Y(e^{j\omega_1},e^{j\omega_2})
=
H(e^{j\omega_1},e^{j\omega_2})X(e^{j\omega_1},e^{j\omega_2})
$$

가 된다.

## 2D Sampling과 Reconstruction

analog 2D signal $x_c(t_1,t_2)$를 sampling period $T_1,T_2$로 샘플링하면

$$
x[n_1,n_2]=x_c(n_1T_1,n_2T_2)
$$

각 축에서 Nyquist 조건을 만족해야 aliasing 없이 복원할 수 있다.

$$
\Omega_{s1}>2\Omega_{N1},
\qquad
\Omega_{s2}>2\Omega_{N2}
$$

2D에서는 한 축이라도 sampling rate가 부족하면 해당 방향으로 aliasing이 생긴다.

## Image Processing의 네 영역

| 영역 | 목적 | 예 |
|---|---|---|
| Enhancement | 사람이 보기 좋게 개선 | contrast enhancement, TV |
| Restoration | degradation 제거/감소 | deblurring, superresolution |
| Coding | 적은 bit로 표현 | JPEG, MPEG |
| Understanding | 의미/기호 추출 | computer vision, robotics, target identification |

## 체크포인트

- 1D의 time index가 2D에서는 spatial index $(n_1,n_2)$로 확장된다.
- LTI는 2D에서 LSI/space-invariant system이 된다.
- impulse response는 영상처리에서 PSF로 불린다.
- 2D convolution은 blur/filtering의 기본 모델이다.
- DSFT는 2D frequency 성분을 분석하는 도구이다.
- sampling/reconstruction과 aliasing 개념은 축별로 적용된다.
