---
과목: Digital Communications
유형: Lecture Note
주제: 통신 시스템 개요, modulation, demodulation, AM, angle modulation
tags:
  - digital-communications
  - modulation
  - demodulation
  - am
  - fm
  - pm
---

# Basic of Communications - 디지털 통신 개요

## 핵심 요약

이 강의는 디지털 통신 시스템의 큰 흐름을 소개한다. 음성 같은 아날로그 정보는 sampling, quantization, bit 변환을 거쳐 baseband digital signal이 되고, 실제 채널에 맞게 passband로 변조되어 전송된다. 수신기는 반대로 down conversion, detection, decision을 거쳐 원래 정보를 복원한다.

전체 관점:

```text
source
-> sampling
-> quantization
-> bit stream
-> modulation
-> channel + noise
-> demodulation
-> decision
-> recovered information
```

## 디지털 통신 시스템의 기본 흐름

송신단에서는 정보 신호를 전송 가능한 형태로 바꾼다.

- 사람 목소리: 대략 `50 Hz ~ 4000 Hz`
- 휴대폰 전파: 수백 MHz에서 수천 MHz
- 실제 시스템 대역폭: 수십 MHz 수준

baseband 신호를 그대로 보내기 어려우므로 carrier를 곱해 passband로 올린다.

수신단에서는 passband 신호를 다시 baseband로 내리고, 수신 파형이 어떤 symbol에 가까운지 결정한다.

## Modulation

### 정의

Modulation은 신호 정보를 전송 매체의 특성에 맞는 파형으로 변환하는 과정이다.

### 변조를 하는 이유

1. 장거리 전송
2. 안테나 길이 및 크기 단축
3. 신호 대역폭 증가
4. 사용 가능한 주파수 영역으로 이동
5. 여러 사용자/채널을 주파수 영역에서 분리

안테나 크기 예시는 변조의 필요성을 잘 보여준다.

```text
lambda = c / f
```

- 4 kHz 음성 신호를 직접 보내면 파장이 매우 길어져 현실적인 안테나가 어렵다.
- 1 GHz 대역으로 변조하면 파장이 약 `0.3 m`가 되어 소형 기기에 적합해진다.

## Analog Modulation

아날로그 변조는 연속적인 source signal `f(t)`를 carrier에 실어 보낸다.

```text
s(t) = f(t) cos(2 pi f_c t)
```

여기서 `cos(2 pi f_c t)`는 carrier이다.

특징:

- source signal 값이 연속적이므로 가능한 변조 파형도 연속적으로 많다.
- 대표적으로 AM, FM, PM이 있다.

## Digital Modulation

디지털 변조는 bit sequence를 미리 약속한 유한 개의 고주파 패턴에 mapping한다.

예시:

- phase modulation: bit `0`, `1`을 서로 다른 phase로 표현
- frequency modulation: `00`, `01`, `10`, `11`을 서로 다른 frequency로 표현

디지털 복조는 다음 두 단계로 이해할 수 있다.

```text
received waveform
-> 가장 가까운 송신 패턴 선택
-> pattern-to-bit 변환
```

## Demodulation

Demodulation은 변조되어 수신된 신호를 원래 정보로 되돌리는 과정이다.

아날로그 demodulation:

- envelope detection
- synchronous detection

디지털 demodulation:

- 수신 신호를 후보 패턴과 비교
- 가장 비슷한 symbol 선택
- symbol을 bit sequence로 변환

## Amplitude Modulation 개요

AM은 전달하려는 정보에 따라 carrier의 amplitude를 바꾸는 변조 방식이다.

일반 형태:

```text
s(t) = A(t) cos(2 pi f_c t + theta)
```

특징:

- phase는 고정되거나 중요하지 않다.
- 정보는 amplitude 변화에 들어간다.
- envelope detection으로 비교적 간단히 복조할 수 있다.

## Angle Modulation 개요

Angle modulation은 amplitude를 일정하게 유지하고 phase 또는 instantaneous frequency에 정보를 싣는 방식이다.

일반 형태:

```text
s(t) = A cos(2 pi f_c t + theta(t))
```

### Instantaneous Frequency

phase가 시간에 따라 변하면 instantaneous angular frequency와 instantaneous frequency를 정의할 수 있다.

```text
omega_i(t) = d theta(t) / dt
f_i(t) = (1 / 2 pi) d theta(t) / dt
```

## FM

FM은 instantaneous frequency가 입력 신호 `m(t)`에 비례한다.

```text
f_i(t) = k_FM m(t) + f_c
```

phase는 instantaneous frequency를 적분해 얻는다.

```text
theta(t) = 2 pi integral f_i(t) dt
```

## PM

PM은 phase가 입력 신호 `m(t)`에 비례한다.

```text
theta(t) = 2 pi k_PM m(t) + 2 pi f_c t + theta_0
```

PM의 instantaneous frequency는 phase를 미분해서 얻는다.

```text
f_i(t) = k_PM dm(t)/dt + f_c
```

## AM vs FM vs PM

| 구분 | 정보가 실리는 위치 | amplitude | phase/frequency |
|---|---|---|---|
| AM | amplitude | 변함 | 거의 고정 |
| FM | instantaneous frequency | 일정 | 입력 신호에 따라 frequency 변화 |
| PM | phase | 일정 | 입력 신호에 따라 phase 변화 |

## 시험 포인트

- modulation이 필요한 이유를 안테나 길이, 장거리 전송, 주파수 이동 관점에서 설명할 수 있어야 한다.
- analog modulation과 digital modulation의 차이를 구분해야 한다.
- AM, FM, PM에서 정보가 어떤 parameter에 실리는지 비교할 수 있어야 한다.
- instantaneous frequency가 phase의 시간 미분이라는 점을 기억한다.

## 같이 보면 좋은 노트

- [[02 Baseband Transmission - 샘플링 PCM Quantization ISI]]
- [[03 Bandpass Transmission - 디지털 변조와 검파]]
- [[10 Amplitude Modulation - AM DSB SSB VSB]]
- [[11 Angle Modulation - FM PM과 SNR]]
