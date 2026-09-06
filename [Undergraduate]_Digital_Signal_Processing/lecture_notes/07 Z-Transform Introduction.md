---
title: "07. Z-Transform Introduction"
pages: 16
tags: [DSP, lecture-note, z-transform, ROC]
---

# 07. Z-Transform Introduction

> 이전: [[06 Linear Algebra Inverse]]
> 다음: [[08 Z-Transform Analysis of LTI Systems]]

## 학습 목표

이 자료는 z-transform의 정의, ROC, inverse z-transform, 주요 성질을 다룬다.

## Z-Transform 정의

양방향 z-transform:

$$
X(z)=\sum_{n=-\infty}^{\infty}x[n]z^{-n}
$$

여기서

$$
z=re^{j\omega}
$$

로 두면 z-transform은 DTFT를 확장한 형태이다. unit circle $|z|=1$ 위에서 수렴하면

$$
X(e^{j\omega})=X(z)\big|_{z=e^{j\omega}}
$$

가 DTFT가 된다.

## ROC

ROC(region of convergence)는 z-transform 급수가 수렴하는 z-plane 영역이다.

중요 성질:

- ROC는 pole을 포함하지 않는다.
- ROC는 connected region이다.
- finite-duration sequence의 ROC는 보통 z-plane 전체이다. 단, $z=0$ 또는 $z=\infty$는 예외가 될 수 있다.
- right-sided sequence의 ROC는 가장 바깥 pole의 바깥쪽이다.
- left-sided sequence의 ROC는 가장 안쪽 pole의 안쪽이다.
- two-sided sequence의 ROC는 두 pole 사이의 ring이다.
- unit circle이 ROC에 포함되면 DTFT가 존재한다.
- LTI 시스템에서 unit circle이 ROC에 포함되면 stable이다.

## 같은 식, 다른 ROC

예를 들어

$$
X(z)=\frac{1}{1-az^{-1}}
$$

는 ROC에 따라 다른 시간 신호를 나타낸다.

### Right-sided

$$
x[n]=a^n u[n],
\qquad ROC: |z|>|a|
$$

### Left-sided

$$
x[n]=-a^n u[-n-1],
\qquad ROC: |z|<|a|
$$

따라서 inverse z-transform에서는 algebraic expression만 보면 안 되고 ROC를 반드시 같이 봐야 한다.

## Finite-Duration Sequence

길이가 유한한 수열:

$$
x[n]=a^n,\quad 0\le n\le N-1
$$

z-transform:

$$
X(z)=\sum_{n=0}^{N-1}a^nz^{-n}
=\frac{1-a^Nz^{-N}}{1-az^{-1}}
$$

분자와 분모의 cancellation 때문에 pole-zero 해석에서 겉보기 식만으로 ROC를 오해하지 않도록 주의한다.

## Inverse Z-Transform 방법

### 1. 표준 쌍 이용

자주 쓰는 쌍:

$$
\delta[n] \leftrightarrow 1
$$

$$
a^n u[n] \leftrightarrow \frac{1}{1-az^{-1}},\quad |z|>|a|
$$

$$
-a^n u[-n-1] \leftrightarrow \frac{1}{1-az^{-1}},\quad |z|<|a|
$$

### 2. Partial Fraction Expansion

유리함수 $X(z)$를 1차 항들의 합으로 분해한 뒤 표준쌍을 적용한다. 이때 각 항의 ROC가 전체 ROC와 일관되어야 한다.

### 3. Power Series Expansion

$X(z)$를 $z^{-1}$ 또는 $z$의 급수로 전개해 계수에서 $x[n]$을 읽는다.

right-sided ROC이면 보통 $z^{-1}$의 양의 거듭제곱으로 전개된다.

## Z-Transform 성질

| 성질 | 시간 영역 | z 영역 |
|---|---|---|
| 선형성 | $ax_1[n]+bx_2[n]$ | $aX_1(z)+bX_2(z)$ |
| 시간 이동 | $x[n-n_0]$ | $z^{-n_0}X(z)$ |
| 지수 곱 | $a^nx[n]$ | $X(z/a)$ |
| 미분 | $nx[n]$ | $-z\frac{dX(z)}{dz}$ |
| conjugation | $x^*[n]$ | $X^*(z^*)$ |
| 시간 반전 | $x[-n]$ | $X(z^{-1})$ |
| convolution | $x_1[n]*x_2[n]$ | $X_1(z)X_2(z)$ |

## Stable/Causal LTI와 ROC

시스템 함수:

$$
H(z)=\sum_{n=-\infty}^{\infty}h[n]z^{-n}
$$

causal LTI:

- impulse response가 right-sided
- ROC가 가장 바깥 pole 바깥쪽

stable LTI:

- unit circle이 ROC에 포함

causal and stable LTI:

- 모든 pole이 unit circle 내부에 있어야 한다.

## 체크포인트

- z-transform은 식 + ROC가 하나의 쌍이다.
- 같은 $X(z)$라도 ROC가 다르면 완전히 다른 $x[n]$이다.
- DTFT는 z-transform의 unit circle evaluation이다.
- causal/stable 판단은 pole 위치와 ROC를 함께 봐야 한다.
