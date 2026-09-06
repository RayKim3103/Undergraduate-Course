---
title: "02. Introduction to Digital Signal Processing"
pages: 11
tags: [DSP, lecture-note, introduction]
---

# 02. DSP Introduction

> 이전: [[01 연구실 소개와 DSP 응용 맥락]]
> 다음: [[03 Review of Signals and Systems]]

## 학습 목표

이 자료는 DSP의 기본 개념과 응용 분야를 소개한다.

- signal과 system의 정의
- analog, discrete-time, digital signal의 차이
- A/D, D/A 변환의 역할
- 1차원 DSP와 다차원 DSP의 구분
- speech, acoustic, image, medical, HDTV, DVP, satellite 등 응용 사례

## Signal과 System

### Signal

신호는 하나 이상의 독립변수에 대한 함수이다.

예:

- 음성 신호: 시간 $t$ 또는 sample index $n$의 함수
- 영상: 공간 좌표 $(m,n)$의 함수
- 동영상: 공간과 시간 $(m,n,t)$의 함수

### System

시스템은 입력 신호를 출력 신호로 변환하는 과정이다.

$$
x \xrightarrow{T\{\cdot\}} y
$$

DSP에서 시스템은 필터, 압축기, 복원 알고리즘, 인식기처럼 신호를 목적에 맞게 바꾸는 연산으로 이해할 수 있다.

## Analog, Discrete-Time, Digital

| 구분 | 독립변수 | 진폭 | 예 |
|---|---|---|---|
| Continuous-time signal | 연속 | 연속 | 마이크 전압, 아날로그 영상 |
| Discrete-time signal | 이산 | 연속 | 샘플링된 신호 |
| Digital signal | 이산 | 이산 | 양자화된 오디오/이미지 픽셀 |

### A/D 변환

A/D converter는 두 단계를 포함한다.

1. sampling: 시간 또는 공간 축을 이산화한다.
2. quantization: 진폭 값을 유한한 level로 근사한다.

예를 들어 영상에서는 픽셀 위치가 공간 sampling이고, 8-bit grayscale의 0-255 값이 amplitude quantization이다.

### D/A 변환

D/A converter는 디지털 샘플에서 연속시간/연속공간 신호를 재구성한다. 이상적 재구성은 sinc interpolation과 low-pass reconstruction filter로 설명된다.

## 왜 DSP가 중요한가

DSP가 널리 쓰이는 이유:

- 디지털 데이터는 저장, 복사, 전송에서 품질 유지가 쉽다.
- 알고리즘을 software로 바꿔 다양한 처리를 적용할 수 있다.
- adaptive processing처럼 상황에 따라 filter를 바꾸는 처리가 가능하다.
- noise reduction, compression, enhancement, recognition 등 고급 처리가 가능하다.

## DSP의 분류

### 1차원 DSP

독립변수가 하나인 경우이다. 주로 시간축 신호를 다룬다.

- speech signal processing
- acoustical signal processing
- music synthesis
- sonar/hydrophone

### 다차원 DSP

독립변수가 둘 이상인 경우이다.

- 2D DSP: 이미지, 공간 영역
- 3D DSP: 동영상, spatio-temporal domain
- stereo/multiview image processing

## 주요 응용 사례

### 음성 처리

- 음성 인식
- 화자 식별
- speech enhancement
- analog 처리보다 훨씬 유연한 알고리즘 적용 가능

### 음향 신호 처리

- digital music synthesizer
- digital instruments
- hydrophone/sonograph
- 군사/의료 목적 음향 분석

### 영상 및 동영상 처리

CCD/CMOS 센서로 획득한 디지털 영상은 필터링, 향상, 복원, 압축, 인식에 모두 DSP가 적용된다.

### Hubble Space Telescope 사례

초기 Hubble telescope은 광학계 결함 때문에 blurred image를 생성했다. 가능한 해결책은 다음이었다.

- primary mirror 교체: 거의 불가능
- 보정 광학 장치 설치: 우주왕복선 발사 필요
- digital image processing으로 광학 결함 보정

이 사례는 물리적 hardware 문제 일부를 DSP 알고리즘으로 보상할 수 있음을 보여준다.

### 의료 영상

- NMR-CT, CAT scan
- digital X-ray angiography
- Fourier Slice Theorem 기반 영상 재구성
- 비침습적으로 내부 구조 분석 가능

### HDTV와 Digital Video-phone

고해상도 영상과 움직이는 영상은 데이터량이 매우 크므로 압축이 필수이다.

- HDTV: MPEG2 등 압축 기술 필요
- DVP: 기존 전화선에서 음성과 영상을 같이 전송하기 위해 MPEG4 같은 압축 기술 필요

## 핵심 정리

- DSP는 신호를 digital domain으로 가져온 뒤 원하는 목적에 맞게 변환하는 학문이다.
- A/D 변환에서 sampling과 quantization을 구분해야 한다.
- digital processing의 장점은 유연성, 반복성, 저장/전송 안정성이다.
- 1D DSP는 시간 신호, 2D/3D DSP는 영상과 동영상으로 자연스럽게 확장된다.
