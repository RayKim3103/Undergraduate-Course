---
title: "08. Z-Transform Analysis of LTI Systems"
pages: 15
tags: [DSP, lecture-note, z-transform, LTI, minimum-phase]
---

# 08. Z-Transform Analysis of LTI Systems

> 이전: [[07 Z-Transform Introduction]]
> 다음: [[09 Structures and Parametric Modeling]]

## 학습 목표

이 자료는 z-transform을 이용해 LTI 시스템을 분석한다.

- inverse system
- ROC와 inverse system의 안정성/인과성
- $H(z)$와 frequency response
- magnitude와 phase의 관계
- minimum phase system
- all-pass system

## Inverse System

LTI 시스템 $H(z)$에 inverse system $H_i(z)$를 cascade로 연결했을 때 전체 응답이 identity가 되려면

$$
H(z)H_i(z)=1
$$

따라서

$$
H_i(z)=\frac{1}{H(z)}
$$

이다.

시간 영역에서는

$$
h[n]*h_i[n]=\delta[n]
$$

을 만족해야 한다.

## ROC의 중요성

inverse system에서도 ROC 선택이 중요하다. convolution theorem을 적용하려면 원 시스템의 ROC와 inverse system의 ROC가 겹쳐야 한다.

또한 같은 $1/H(z)$라도 ROC 선택에 따라 inverse impulse response가 causal/stable/noncausal/unstable로 달라질 수 있다.

## Zero와 Inverse Pole

$H(z)$의 zero는 $H_i(z)$의 pole이 된다.

따라서 원 시스템의 zero가 unit circle 밖에 있으면 causal inverse를 만들 때 inverse pole도 unit circle 밖에 생긴다. 이 경우 causal inverse는 unstable할 수 있다.

핵심:

- 원 시스템 zero가 unit circle 내부: causal stable inverse 가능
- 원 시스템 zero가 unit circle 외부: causal inverse는 불안정, stable inverse는 noncausal이 될 수 있음

## Frequency Response와 Pole-Zero Plot

frequency response는 unit circle 위의 시스템 함수 값이다.

$$
H(e^{j\omega}) = H(z)\big|_{z=e^{j\omega}}
$$

pole-zero plot에서 unit circle 위의 점 $e^{j\omega}$와 zero/pole 사이의 거리와 각도를 보면 magnitude와 phase를 해석할 수 있다.

유리 시스템:

$$
H(z)=C
\frac{\prod_k(1-z_k z^{-1})}
{\prod_m(1-p_m z^{-1})}
$$

unit circle에서 magnitude는 대략

$$
|H(e^{j\omega})|
= |C|\frac{\prod_k|e^{j\omega}-z_k|}
{\prod_m|e^{j\omega}-p_m|}
$$

로 볼 수 있다.

- zero 근처 주파수: magnitude 감소
- pole 근처 주파수: magnitude 증가

## Magnitude와 Phase

일반적으로 magnitude만 알면 phase가 유일하게 정해지지 않는다. pole/zero 개수가 주어져도 가능한 phase 선택은 여러 개일 수 있다.

하지만 minimum phase system이면 magnitude와 phase가 서로 유일하게 연결된다.

## Minimum Phase System

discrete-time minimum phase system은 보통 다음 조건을 만족한다.

- causal
- stable
- 모든 zero가 unit circle 내부
- inverse system도 causal and stable

minimum phase system은 같은 magnitude response를 갖는 stable causal 시스템 중 phase delay가 가장 작다.

## All-Pass System

all-pass system은 모든 주파수에서 magnitude가 1인 시스템이다.

$$
|H_{ap}(e^{j\omega})|=1
$$

1차 all-pass의 전형적 형태:

$$
H_{ap}(z)=\frac{z^{-1}-a^*}{1-az^{-1}},
\qquad |a|<1
$$

특징:

- magnitude는 변하지 않는다.
- phase만 바꾼다.
- pole과 zero가 unit circle에 대해 reciprocal conjugate 위치에 놓인다.

## Nonminimum Phase를 Minimum Phase로 만들기

unit circle 밖 zero를 안쪽 reciprocal conjugate 위치로 반사시키면 magnitude response는 all-pass factor 때문에 유지할 수 있고 phase 특성은 minimum phase 쪽으로 바뀐다.

즉 같은 magnitude를 갖는 시스템을

$$
H(z)=H_{\min}(z)H_{ap}(z)
$$

처럼 minimum phase part와 all-pass part로 분해해 이해할 수 있다.

## 체크포인트

- inverse system은 단순히 $1/H(z)$가 아니라 ROC까지 포함해 판단해야 한다.
- $H(z)$의 zero는 inverse system의 pole이 된다.
- causal stable inverse가 필요하면 zero가 unit circle 안에 있어야 한다.
- all-pass는 magnitude를 바꾸지 않고 phase만 바꾼다.
- minimum phase system은 inverse도 stable/causal인 가장 다루기 좋은 시스템이다.
