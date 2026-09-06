---
title: "12. DFT and FFT"
pages: 18
tags: [DSP, lecture-note, DFT, FFT]
---

# 12. DFT and FFT

> 이전: [[11 Digital Filter Design]]
> 다음: [[13 2D DSP Overview]]

## 학습 목표

이 자료는 DFT와 FFT를 다룬다.

- DTFT sampling으로부터 DFT 도입
- DFS와 DFT의 관계
- DFT 성질
- circular shift/convolution
- FFT 계산량
- Decimation-in-Time
- Decimation-in-Frequency

## DFS 복습

주기 $N$인 이산시간 신호는 discrete Fourier series로 표현된다.

$$
x[n]=\sum_{k=0}^{N-1}a_k e^{j\frac{2\pi}{N}kn}
$$

$$
a_k=\frac1N\sum_{n=0}^{N-1}x[n]e^{-j\frac{2\pi}{N}kn}
$$

## DFT 정의

길이 $N$의 finite-duration sequence $x[n]$에 대해 N-point DFT:

$$
X[k]=\sum_{n=0}^{N-1}x[n]W_N^{kn},
\qquad
W_N=e^{-j2\pi/N}
$$

역 DFT:

$$
x[n]=\frac1N\sum_{k=0}^{N-1}X[k]W_N^{-kn}
$$

DFT는 DTFT를 $N$개의 등간격 주파수에서 sampling한 것으로 볼 수 있다.

$$
X[k]=X(e^{j\omega})\big|_{\omega=2\pi k/N}
$$

## Frequency Sampling의 의미

DTFT를 주파수 영역에서 $N$점 sampling하면 시간 영역에서는 원 신호가 $N$주기로 반복된다. 따라서 DFT는 finite sequence 하나를 다루는 동시에, 그 sequence가 주기적으로 반복된다고 보는 관점을 포함한다.

이 관점 때문에 DFT에서는 circular shift와 circular convolution이 자연스럽게 등장한다.

## DFT 성질

### Linearity

$$
ax_1[n]+bx_2[n]
\leftrightarrow
aX_1[k]+bX_2[k]
$$

길이가 다른 신호를 더할 때는 공통 DFT 길이를 맞추기 위해 zero padding을 고려한다.

### Circular Shift

시간 영역에서 $n_0$만큼 circular shift하면

$$
x[(n-n_0)_N]
\leftrightarrow
W_N^{kn_0}X[k]
$$

주파수 영역에서 phase factor가 곱해진다.

### Duality

DFT는 시간과 주파수 사이에 대칭성이 있다.

만약

$$
x[n]\leftrightarrow X[k]
$$

이면 적절한 index reversal과 scale factor를 포함해

$$
X[n]\leftrightarrow N x[(-k)_N]
$$

관계가 성립한다.

### Symmetry

실수 신호 $x[n]$에 대해

$$
X[k]=X^*[(-k)_N]
$$

즉 magnitude는 even symmetry, phase는 odd symmetry를 갖는다.

### Circular Convolution

N-point circular convolution:

$$
y[n]=\sum_{m=0}^{N-1}x_1[m]x_2[(n-m)_N]
$$

DFT 영역:

$$
Y[k]=X_1[k]X_2[k]
$$

linear convolution을 DFT로 계산하려면 결과 길이 이상으로 zero padding해야 circular aliasing을 피할 수 있다.

## DFT와 Circulant Matrix

circular convolution은 circulant matrix 곱과 같다.

$$
\mathbf{y}=C\mathbf{x}
$$

circulant matrix는 DFT matrix로 대각화된다.

$$
C = W^{-1}\Lambda W
$$

따라서 DFT 영역에서는 convolution이 각 frequency bin별 scalar multiplication으로 분리된다.

## FFT가 필요한 이유

DFT 직접 계산은 각 $k$마다 $N$개의 곱셈이 필요하므로 전체 복소 곱셈 수가 $N^2$ 수준이다.

FFT는 대칭성과 주기성을 이용해 계산량을

$$
O(N^2)\rightarrow O(N\log_2N)
$$

으로 줄인다.

예:

- 256-point DFT: 직접 계산은 약 65536 complex multiplications
- 256-point FFT: 훨씬 적은 butterfly 연산
- 1024-point에서는 차이가 더 커진다.

## Decimation-in-Time FFT

입력 sequence를 짝수 index와 홀수 index로 나눈다.

$$
X[k]
=\sum_{n=0}^{N-1}x[n]W_N^{kn}
$$

짝수/홀수 분해:

$$
X[k]
=\sum_{r=0}^{N/2-1}x[2r]W_{N/2}^{kr}
 + W_N^k\sum_{r=0}^{N/2-1}x[2r+1]W_{N/2}^{kr}
$$

즉

$$
X[k]=E[k]+W_N^kO[k]
$$

$$
X[k+N/2]=E[k]-W_N^kO[k]
$$

여기서 $E[k]$와 $O[k]$는 각각 even/odd subsequence의 $N/2$-point DFT이다.

## Decimation-in-Frequency FFT

출력 주파수 index를 짝수/홀수로 나눈다. 입력의 앞 절반과 뒤 절반을 더하고 빼는 방식으로 작은 DFT로 분해한다.

DIT가 “시간 index를 먼저 쪼개는 방식”이라면 DIF는 “주파수 index를 먼저 쪼개는 방식”으로 이해하면 된다.

## Butterfly

FFT의 기본 계산 단위는 butterfly이다.

$$
A = E + W O
$$

$$
B = E - W O
$$

여기서 $W$는 twiddle factor이다.

## 체크포인트

- DFT는 DTFT의 sampling이며 시간 영역 주기화를 동반한다.
- DFT에서 convolution은 circular convolution이다.
- linear convolution을 원하면 zero padding 길이를 충분히 잡아야 한다.
- FFT는 새로운 변환이 아니라 DFT를 빠르게 계산하는 알고리즘이다.
- DIT: time index 분해, DIF: frequency index 분해.
