---
title: "01. 연구실 소개와 DSP 응용 맥락"
pages: 3
tags: [DSP, lecture-note, applications, image-processing]
---

# 01. 연구실 소개와 DSP 응용 맥락

> 이전: [[00 강의계획과 개요]]
> 다음: [[02 DSP Introduction]]

## 자료의 위치

이 PDF는 본격적인 수식 강의라기보다, Yonsei University SuperResolution Image Processing Lab의 연구 분야를 통해 DSP가 실제로 어디에 쓰이는지 보여주는 소개 자료이다. 텍스트 추출상 한글 인코딩이 많이 깨져 있지만, 자료의 중심 주제는 초해상도 영상처리, 복원, 의료영상, 원격탐사, 특수 센서 영상, 자율주행용 영상 인식 등이다.

## 연구실 핵심 키워드

- Superresolution image processing
- Digital image restoration
- Low-resolution/low-SNR image enhancement
- Multi-sensor image reconstruction
- Medical imaging
- Remote sensing
- Camera ISP
- ToF, light-field camera, LWIR 등 특수 목적 센서

## DSP 관점에서 보는 연구 주제

### 초해상도 영상처리

초해상도는 낮은 해상도의 관측 영상에서 더 높은 해상도의 영상을 복원하는 문제이다. DSP 관점에서는 다음 모델로 이해할 수 있다.

$$
\mathbf{y}_k = D_k B_k M_k \mathbf{x} + \mathbf{n}_k
$$

- $\mathbf{x}$: 복원하고 싶은 고해상도 영상
- $\mathbf{y}_k$: $k$번째 저해상도 관측 영상
- $M_k$: motion/warping
- $B_k$: blur
- $D_k$: down-sampling
- $\mathbf{n}_k$: noise

이 모델은 뒤의 [[05 LTI System to Linear Algebra]]와 직접 연결된다. 영상처리 문제를 행렬 방정식으로 쓰면 복원은 inverse problem이 된다.

### 영상 복원

흐림, 잡음, 센서 한계로 손상된 영상을 복구하는 문제이다.

대표적인 관측 모델:

$$
y[m,n] = h[m,n] * x[m,n] + v[m,n]
$$

여기서 $h[m,n]$은 blur point-spread function, $v[m,n]$은 noise이다. 복원은 단순히 $1/H$를 곱하는 문제가 아니라, 잡음 증폭과 안정성 문제를 함께 다뤄야 한다. 이 때문에 [[10 Wiener Optimal Filter]]가 중요해진다.

### 의료영상

Digital angiography, sonography, MRI/CT 같은 의료영상 장비는 대부분 DSP 기반이다.

- Fourier Slice Theorem 기반 재구성
- 저선량 영상의 noise reduction
- 저해상도/저 SNR 환경에서 영상 품질 향상
- 다중 센서 또는 다중 프레임을 이용한 복원

의료영상에서는 원본 신호를 직접 볼 수 없고 관측된 투영/샘플/잡음 신호로부터 내부 구조를 추정한다. 따라서 sampling, inverse problem, regularization이 모두 중요하다.

### 원격탐사와 특수 센서

Remote sensing, multi-spectral/hyperspectral imaging, ToF, LWIR 등의 센서는 일반 RGB 카메라와 다른 물리적 특성을 갖는다.

- 공간 해상도와 스펙트럼 해상도의 trade-off
- 센서 잡음과 결측 데이터 보정
- 다중 센서 fusion
- 저조도/극저조도 환경의 영상 복원

DSP 관점에서는 서로 다른 sampling grid, point-spread function, spectral response를 하나의 관측 모델로 통합하는 문제가 된다.

## 이 자료가 강의 전체와 연결되는 방식

| 응용 문제 | 강의 개념 |
|---|---|
| 초해상도 복원 | LTI/LSI system, inverse filtering, least squares |
| 잡음 제거 | Wiener filter, power spectrum, correlation |
| 영상 압축 | DFT/FFT, frequency-domain representation |
| 센서 영상 처리 | sampling, aliasing, reconstruction |
| 2D 영상 분석 | 2D convolution, DSFT, 2D sampling |

## 핵심 정리

- DSP는 오디오, 통신, 영상, 의료, 센서 등 거의 모든 공학 시스템의 기반 도구이다.
- 영상처리에서는 “실제 장면”과 “센서 관측” 사이에 blur, sampling, noise가 개입한다.
- 많은 복원 문제는 $\mathbf{y}=H\mathbf{x}+\mathbf{n}$ 형태의 inverse problem으로 정식화된다.
- 이 강의의 수학적 도구는 연구실의 초해상도/복원 문제를 이해하기 위한 언어이기도 하다.
