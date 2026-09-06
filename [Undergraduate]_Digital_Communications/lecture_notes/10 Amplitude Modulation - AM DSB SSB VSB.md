---
과목: Digital Communications
유형: Lecture Note
주제: Amplitude modulation, DSB-SC, DSB-LC, FDM, SSB, VSB, AM SNR
tags:
  - digital-communications
  - amplitude-modulation
  - dsb-sc
  - dsb-lc
  - ssb
  - vsb
  - fdm
---

# Amplitude Modulation - AM DSB SSB VSB

## 핵심 요약

이 강의는 continuous wave modulation 중 amplitude modulation 계열을 다룬다. DSB-SC, DSB-LC, FDM, SSB, VSB의 생성/복조 방식과 동기 오차, carrier power 효율, envelope detector, heterodyne receiver, bandpass noise, AM 수신 SNR이 핵심이다.

## AM의 기본 형태

일반 sinusoidal carrier:

```text
s(t) = A(t) cos(omega_c t + theta(t))
```

AM에서는 phase가 아니라 amplitude가 message signal에 비례하도록 만든다.

```text
s_AM(t) = message-dependent amplitude * cos(omega_c t)
```

## DSB-SC

DSB-SC는 Double-Sideband Suppressed-Carrier modulation이다.

기본 형태:

```text
s(t) = m(t) cos(omega_c t)
```

주파수 영역:

- baseband spectrum `M(w)`가 `+omega_c`, `-omega_c` 주변으로 복사된다.
- upper sideband와 lower sideband가 모두 존재한다.
- carrier 성분 자체는 보내지 않는다.

장점:

- carrier에 power를 낭비하지 않는다.
- DSB-LC보다 power efficiency가 좋다.

단점:

- 수신단에서 carrier frequency와 phase를 정확히 알아야 한다.
- synchronous detection이 필요하다.

## DSB-SC Demodulation

복조는 수신 신호에 같은 carrier를 다시 곱하고 LPF를 통과시키는 방식이다.

```text
r(t) cos(omega_c t)
-> LPF
-> (1/2) m(t)
```

동기 오류:

- phase error가 있으면 출력 amplitude가 줄거나 왜곡된다.
- frequency error가 있으면 low-frequency beat가 곱해져 심각한 distortion이 생긴다.

따라서 DSB-SC에서는 synchronization이 핵심 문제이다.

## Quadrature Multiplexing

cos carrier와 sin carrier는 직교하므로 같은 carrier frequency 대역에 두 신호를 동시에 실을 수 있다.

```text
s(t) = m_1(t) cos(omega_c t) + m_2(t) sin(omega_c t)
```

수신단에서 각각 cos와 sin을 곱하고 LPF를 통과시키면 두 신호를 분리할 수 있다.

필수 조건:

- carrier phase synchronization
- I/Q orthogonality 유지

## DSB-SC Generation

DSB-SC를 만들기 위해 단순 LTI 시스템만으로는 frequency translation을 구현할 수 없다. modulation은 nonlinear 또는 time-varying 시스템을 필요로 한다.

생성 방법:

- chopper modulator
- switching circuit
- ring modulator
- square-law modulator
- balanced modulator

## Costas Receiver

Costas receiver는 DSB-SC에서 carrier phase synchronization을 복원하기 위한 구조이다.

구성:

- I-channel product detector
- Q-channel product detector
- low-pass filters
- phase discriminator
- voltage-controlled oscillator

Q-channel이 phase error 정보를 제공하고, VCO를 조절해 carrier 동기를 맞춘다.

## DSB-LC

DSB-LC는 Double-Sideband Large-Carrier AM으로, 보통 말하는 commercial AM이다.

기본 형태:

```text
s(t) = [A + m(t)] cos(omega_c t)
```

carrier를 함께 보내므로 수신기가 envelope detection만으로 복조할 수 있다.

장점:

- 수신기 구조가 간단하다.
- 정확한 coherent carrier recovery가 없어도 된다.

단점:

- carrier가 정보를 담지 않으므로 power efficiency가 낮다.
- `A`가 충분히 크지 않으면 envelope가 원 message를 따라가지 못한다.

## Modulation Index

sinusoidal message에서 modulation index는 carrier amplitude 대비 message amplitude의 비율이다.

```text
mu = A_m / A_c
```

조건:

- `mu <= 1`: envelope detection 가능
- `mu > 1`: overmodulation, envelope distortion 발생

## Carrier and Sideband Power

DSB-LC에서는 total power가 carrier power와 sideband power로 나뉜다.

핵심 해석:

- carrier term은 정보를 담지 않지만 큰 power를 차지한다.
- modulation index가 1이어도 AM efficiency는 33%를 넘기 어렵다.
- DSB-SC는 carrier power가 없으므로 이상적으로 information-bearing power efficiency가 높다.

## AM Demodulation

DSB-LC 복조 방식:

- envelope detection
- synchronous detection
- rectifier detection

envelope detector는 diode와 RC 회로로 구현할 수 있어 간단하다.

RC 선택 조건:

- carrier cycle보다는 충분히 느리게 방전
- message 변화보다는 충분히 빠르게 따라감

rectifier detector는 동작 가능하지만 일반적으로 envelope detector보다 비효율적이다.

## FDM

Frequency Division Multiplexing은 여러 baseband signal을 서로 다른 carrier frequency에 올려 한 매체로 동시에 보내는 방식이다.

commercial AM:

- 대략 540 kHz부터 1600 kHz
- channel spacing 약 10 kHz

## Superheterodyne Receiver

superheterodyne receiver는 수신 신호를 intermediate frequency(IF)로 변환한 뒤 복조한다.

구성:

```text
RF amplifier
-> mixer + local oscillator
-> IF amplifier
-> demodulator
-> audio amplifier
```

장점:

- 다양한 station carrier를 같은 IF로 옮겨 동일한 증폭/복조 회로를 사용할 수 있다.
- sensitivity와 selectivity를 개선한다.

주의:

- image frequency 문제가 발생할 수 있다.
- image rejection을 위해 front-end bandpass filtering이 필요하다.

## SSB

SSB는 Single-Sideband modulation이다.

아이디어:

- DSB는 upper sideband와 lower sideband가 같은 정보를 중복해서 가진다.
- 한쪽 sideband만 보내도 message를 복원할 수 있다.
- bandwidth를 거의 절반으로 줄일 수 있다.

생성 방법:

- filtering method
- phase-shift method
- analytic signal과 Hilbert transform 이용

단점:

- sharp filter 또는 정확한 phase shift가 필요해 구현이 어렵다.
- demodulation에서 frequency/phase error에 민감하다.

## Analytic Signal and Hilbert Transform

analytic signal은 real signal을 real part로 갖는 complex signal이며, one-sided spectrum을 갖는다.

SSB 생성에서 Hilbert transform은 원 신호와 90도 phase-shift된 신호를 만들기 위해 사용된다.

개념적 형태:

```text
m_a(t) = m(t) + j m_hat(t)
```

여기서 `m_hat(t)`는 Hilbert transform이다.

## SSB Demodulation

SSB-SC도 synchronous demodulation이 필요하다.

오류 영향:

- phase error: amplitude scaling 및 quadrature leakage
- frequency error: spectrum이 밀려 음성 왜곡 또는 frequency scramble 발생

## VSB

VSB는 Vestigial-Sideband modulation이다.

아이디어:

- SSB처럼 bandwidth를 줄이고 싶지만 완벽한 sideband filter 구현은 어렵다.
- 한 sideband는 거의 모두 보내고, 반대 sideband 일부를 vestige로 보낸다.
- 수신 필터가 두 sideband의 합성 응답을 보상하도록 설계한다.

VSB는 SSB와 DSB의 절충안이다.

특징:

- bandwidth는 SSB보다 약간 크다.
- 구현은 SSB보다 쉽다.
- TV video transmission 등에 사용된다.

## Bandpass Noise

bandpass noise는 in-phase와 quadrature 성분으로 표현한다.

```text
n(t) = n_c(t) cos(omega_c t) - n_s(t) sin(omega_c t)
```

이 표현은 passband noise를 baseband equivalent 형태로 분석하게 해준다.

## AM Reception SNR

### DSB-SC Coherent Detection

coherent detection은 quadrature noise 성분을 제거하므로 SNR 관점에서 유리하다.

### SSB-SC

SSB는 signal과 noise가 같은 방식으로 detector를 통과하므로 DSB의 sideband redundancy를 줄인 형태로 해석할 수 있다.

### DSB-LC Envelope Detection

envelope detector는 nonlinear이다.

- high SNR에서는 coherent detection과 비슷한 결과를 준다.
- low SNR에서는 envelope가 noise에 크게 흔들려 성능이 급격히 나빠진다.

## 시험 포인트

- DSB-SC와 DSB-LC의 carrier 유무, power efficiency, receiver complexity를 비교한다.
- DSB-SC에서 carrier synchronization이 왜 필요한지 설명한다.
- AM envelope detection 조건 `A > |m(t)|` 또는 `mu <= 1`을 기억한다.
- FDM과 superheterodyne receiver의 IF 개념을 이해한다.
- SSB는 bandwidth 절약, VSB는 구현 난이도와 bandwidth 사이 절충이라는 점을 안다.
- bandpass noise의 I/Q 표현을 사용할 수 있어야 한다.

## 같이 보면 좋은 노트

- [[01 Basic of Communications - 디지털 통신 개요]]
- [[04 Noise and Decision - 잡음과 최적 검출]]
- [[11 Angle Modulation - FM PM과 SNR]]
