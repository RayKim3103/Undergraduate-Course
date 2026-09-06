---
title: "05. LTI System to Linear Algebra"
pages: 18
tags: [DSP, lecture-note, linear-algebra, convolution]
---

# 05. LTI System to Linear Algebra

> 이전: [[04 Sampling Rate Change]]
> 다음: [[06 Linear Algebra Inverse]]

## 학습 목표

이 강의는 convolution과 LTI 시스템을 선형대수의 행렬-벡터 형태로 바꾸는 방법을 다룬다.

- linear system equation
- lexicographical ordering
- Toeplitz matrix
- circulant matrix
- matrix squarization
- similarity, diagonalization
- DFT 행렬과 circulant matrix의 관계

## 왜 행렬로 바꾸는가

LTI 시스템은 시간 영역에서 convolution으로 표현된다.

$$
y[n] = h[n]*x[n]
$$

이를 벡터와 행렬로 쓰면

$$
\mathbf{y}=H\mathbf{x}
$$

가 된다. 이 표현은 다음 문제들을 같은 수학적 틀에서 다룰 수 있게 한다.

- filtering: $H$와 $\mathbf{x}$가 주어졌을 때 $\mathbf{y}$ 계산
- filter design: $\mathbf{x}$와 $\mathbf{y}$가 주어졌을 때 $H$ 추정
- inverse filtering: $H$와 $\mathbf{y}$가 주어졌을 때 $\mathbf{x}$ 복원
- blind deconvolution: $\mathbf{y}$만 주어지고 $H,\mathbf{x}$를 동시에 추정

## 1D Convolution의 Toeplitz Matrix 표현

FIR impulse response

$$
h[0],h[1],\ldots,h[M-1]
$$

와 입력

$$
\mathbf{x}=[x[0],x[1],\ldots,x[N-1]]^T
$$

가 있을 때 linear convolution은 다음과 같은 Toeplitz matrix로 표현된다.

$$
\begin{bmatrix}
y[0]\\
y[1]\\
y[2]\\
\vdots
\end{bmatrix}
=
\begin{bmatrix}
h[0] & 0 & 0 & \cdots\\
h[1] & h[0] & 0 & \cdots\\
h[2] & h[1] & h[0] & \cdots\\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
\begin{bmatrix}
x[0]\\
x[1]\\
x[2]\\
\vdots
\end{bmatrix}
$$

Toeplitz matrix는 각 대각선 성분이 일정한 행렬이다. convolution의 shift-invariant 성질이 행렬의 반복 구조로 나타난 것이다.

## Lexicographical Ordering

2D 신호는 행렬 형태이지만 선형대수 처리를 위해 벡터로 펼쳐야 한다. 이때 일정한 순서로 pixel을 나열하는 방식을 lexicographical ordering이라고 한다.

예:

$$
X =
\begin{bmatrix}
x[0,0] & x[0,1]\\
x[1,0] & x[1,1]
\end{bmatrix}
\quad\rightarrow\quad
\mathbf{x}=[x[0,0],x[0,1],x[1,0],x[1,1]]^T
$$

2D convolution은 이 ordering을 통해 block Toeplitz 또는 block circulant matrix로 표현된다.

## Squarization

linear convolution matrix는 입력 길이와 출력 길이가 달라 non-square가 되는 경우가 많다. 역행렬이나 고유분해를 사용하려면 square matrix 형태가 필요할 수 있다.

Squarization은 zero padding 또는 boundary condition 설정을 통해 행렬을 정방행렬로 만드는 과정이다.

대표적 선택:

- zero boundary: 바깥 값을 0으로 가정
- periodic boundary: 신호가 주기적으로 반복된다고 가정

periodic boundary를 쓰면 convolution matrix가 circulant matrix가 된다.

## Circulant Matrix

Circulant matrix는 각 행이 이전 행의 circular shift로 구성되는 행렬이다.

예:

$$
C =
\begin{bmatrix}
c_0 & c_{N-1} & \cdots & c_1\\
c_1 & c_0 & \cdots & c_2\\
\vdots & \vdots & \ddots & \vdots\\
c_{N-1} & c_{N-2} & \cdots & c_0
\end{bmatrix}
$$

circulant matrix는 DFT 행렬로 대각화된다.

$$
C = W^{-1}\Lambda W
$$

여기서 $W$는 DFT matrix이고, $\Lambda$의 대각 성분은 impulse response의 DFT 값이다.

## Similarity와 Diagonalization

두 행렬 $A,B$가

$$
B = P^{-1}AP
$$

를 만족하면 similar하다고 한다. $A$가 충분한 eigenvector를 가지면

$$
A=PDP^{-1}
$$

로 대각화할 수 있다.

LTI/circulant 시스템에서 Fourier basis가 eigenvector 역할을 하므로, 주파수 영역에서는 convolution이 대각 행렬 곱으로 단순해진다.

## 핵심 연결

시간 영역:

$$
y[n]=h[n]*x[n]
$$

행렬 영역:

$$
\mathbf{y}=H\mathbf{x}
$$

주파수 영역:

$$
Y[k]=H[k]X[k]
$$

세 식은 같은 현상을 다른 표현으로 쓴 것이다.

## 체크포인트

- Toeplitz: linear convolution
- Circulant: circular convolution, periodic boundary
- DFT matrix: circulant matrix의 eigenvector 행렬
- convolution을 행렬로 쓰면 inverse problem과 least squares가 자연스럽게 등장한다.
