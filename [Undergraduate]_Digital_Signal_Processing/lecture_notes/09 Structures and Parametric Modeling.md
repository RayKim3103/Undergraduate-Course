---
title: "09. Structures and Parametric Modeling"
pages: 18
tags: [DSP, lecture-note, filter-structure, AR, MA, ARMA]
---

# 09. Structures and Parametric Modeling

> 이전: [[08 Z-Transform Analysis of LTI Systems]]
> 다음: [[10 Wiener Optimal Filter]]

## 학습 목표

이 자료는 discrete-time system의 구현 구조와 parametric signal modeling을 다룬다.

- AR, MA, ARMA 구조
- linear constant coefficient difference equation
- Direct Form I, Direct Form II
- cascade/parallel connection
- AR modeling과 Yule-Walker equation
- MA/ARMA modeling의 근사와 least squares

## AR, MA, ARMA의 차이

### MA: Moving Average

feed-forward 구조만 가진다.

$$
y[n]=\sum_{k=0}^{M} b_k x[n-k]
$$

시스템 함수:

$$
H(z)=\sum_{k=0}^{M} b_k z^{-k}
$$

특징:

- all-zero system
- FIR
- feedback이 없어 구조적으로 안정적

### AR: Auto-Regressive

현재 출력이 과거 출력에 의해 재귀적으로 결정된다.

$$
y[n] = x[n] - \sum_{k=1}^{N} a_k y[n-k]
$$

시스템 함수:

$$
H(z)=\frac{1}{1+\sum_{k=1}^{N}a_k z^{-k}}
$$

특징:

- all-pole system
- feedback 구조
- pole 위치에 따라 안정성 결정

### ARMA

feed-forward와 feedback을 모두 포함한다.

$$
y[n] = \sum_{k=0}^{M}b_k x[n-k]
- \sum_{k=1}^{N}a_k y[n-k]
$$

시스템 함수:

$$
H(z)=
\frac{\sum_{k=0}^{M}b_k z^{-k}}
{1+\sum_{k=1}^{N}a_k z^{-k}}
$$

특징:

- pole과 zero를 모두 가짐
- IIR 일반형
- Direct Form I/II로 구현 가능

## Direct Form I

분자 $B(z)$와 분모 $A(z)$를 각각 별도 delay line으로 구현한다.

장점:

- 구조가 직관적
- feed-forward와 feedback part가 분리되어 이해하기 쉽다.

단점:

- delay element 수가 많을 수 있다.

## Direct Form II

Direct Form I에서 두 delay line을 공유해 canonical structure로 만든다.

장점:

- 필요한 memory가 줄어든다.
- 실시간 구현에서 효율적이다.

주의:

- 내부 상태 값이 커질 수 있어 finite wordlength effect에 더 민감할 수 있다.

## LTI System 연결 성질

LTI 시스템은 다음 성질을 갖는다.

### Commutation

두 LTI 시스템의 cascade 순서는 바뀌어도 전체 impulse response가 같다.

$$
h_1*h_2=h_2*h_1
$$

### Cascade

연속 연결:

$$
H(z)=H_1(z)H_2(z)
$$

### Parallel

병렬 연결:

$$
H(z)=H_1(z)+H_2(z)
$$

이 성질은 복잡한 필터를 1차/2차 section으로 쪼개 구현할 때 중요하다.

## AR Modeling

AR 모델은 신호를 과거 샘플들의 선형결합과 prediction error로 표현한다.

$$
x[n] = -\sum_{k=1}^{P}a_k x[n-k] + e[n]
$$

예측값:

$$
\hat{x}[n] = -\sum_{k=1}^{P}a_k x[n-k]
$$

prediction error:

$$
e[n]=x[n]-\hat{x}[n]
$$

목표는 error energy를 최소화하는 $a_k$를 찾는 것이다.

## Orthogonality Principle

최소제곱 최적해에서는 오차가 predictor를 구성하는 모든 과거 샘플과 직교한다.

$$
E\{e[n]x^*[n-m]\}=0,
\qquad m=1,\ldots,P
$$

이를 autocorrelation으로 정리하면 Yule-Walker equation이 된다.

## Yule-Walker Equation

autocorrelation

$$
r_x[m]=E\{x[n]x^*[n-m]\}
$$

에 대해 AR 계수는 다음 선형방정식을 만족한다.

$$
\begin{bmatrix}
r_x[0] & r_x[1] & \cdots & r_x[P-1]\\
r_x[1] & r_x[0] & \cdots & r_x[P-2]\\
\vdots & \vdots & \ddots & \vdots\\
r_x[P-1] & r_x[P-2] & \cdots & r_x[0]
\end{bmatrix}
\begin{bmatrix}
a_1\\a_2\\ \vdots\\ a_P
\end{bmatrix}
=
-
\begin{bmatrix}
r_x[1]\\r_x[2]\\ \vdots\\ r_x[P]
\end{bmatrix}
$$

이 행렬은 Toeplitz 구조를 가지므로 효율적인 해법을 사용할 수 있다.

## MA Modeling

MA 모델은 all-zero system이다.

$$
H(z)=\sum_{k=0}^{Q}b_k z^{-k}
$$

강의에서는 MA 모델을 직접 풀기보다, all-zero system의 역수를 all-pole system으로 근사해 계산하는 아이디어를 다룬다.

$$
\frac{1}{B(z)} \approx G(z)
$$

long division을 사용하면 zero 하나도 무한 개의 pole을 가진 all-pole 모델처럼 근사될 수 있다.

## ARMA Modeling

ARMA는 pole과 zero를 모두 추정해야 하므로 AR보다 복잡하다. 전형적인 접근:

1. 먼저 AR 부분 $A(z)$를 추정한다.
2. $B(z)/A(z)$ 형태를 이용해 impulse response 또는 보조 수열을 계산한다.
3. 남은 MA 계수 $b_k$를 least squares로 추정한다.

## 체크포인트

- MA = FIR = all-zero = feed-forward
- AR = all-pole = feedback
- ARMA = pole-zero = feed-forward + feedback
- Direct Form II는 memory를 줄이지만 수치적 민감성을 확인해야 한다.
- AR modeling의 핵심은 orthogonality principle과 Yule-Walker equation이다.
