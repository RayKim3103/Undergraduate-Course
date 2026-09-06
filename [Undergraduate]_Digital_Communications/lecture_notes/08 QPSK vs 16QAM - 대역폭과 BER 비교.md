---
과목: Digital Communications
유형: Supplement
주제: QPSK, 16-QAM, constellation, spectrum, BER comparison
tags:
  - digital-communications
  - qpsk
  - 16qam
  - constellation
  - spectrum
  - ber
---

# QPSK vs 16QAM - 대역폭과 BER 비교

## 핵심 요약

이 자료는 QPSK와 16-QAM을 constellation, modulated signal, spectrum, BER 관점에서 비교한다. 같은 bit rate에서 16-QAM은 한 symbol에 4 bit를 싣기 때문에 QPSK보다 symbol rate가 낮고 bandwidth가 작다. 대신 constellation point가 조밀해 noise와 fading에 더 취약하여 BER 성능은 QPSK보다 나쁘다.

## 비교 조건

자료의 specification:

| 항목 | 값 |
|---|---|
| Sampling frequency | `f_s = 4.8 kHz` |
| Bit rate | `R_b = 240 bps` |
| Average bit energy | `E_b = 1` |
| Roll-off factor | `r = 0` |

비교 항목:

- constellation
- modulated signal
- spectrum
- bit error rate

## Constellation 비교

### QPSK

QPSK는 2차원 I/Q 평면에 4개의 symbol을 배치한다.

```text
M = 4
bits/symbol = log2 4 = 2
```

특징:

- symbol point 간 거리가 비교적 넓다.
- noise margin이 크다.
- BPSK와 같은 BER 성능을 가질 수 있다.

### 16-QAM

16-QAM은 I/Q 평면의 amplitude와 phase를 모두 사용해 16개의 symbol을 배치한다.

```text
M = 16
bits/symbol = log2 16 = 4
```

특징:

- 같은 symbol rate에서 QPSK보다 bit rate가 2배 높다.
- 같은 bit rate에서는 QPSK보다 symbol rate가 절반이다.
- constellation point가 더 조밀해 noise에 민감하다.

## Modulated Signal 비교

같은 bit rate를 맞추면:

- QPSK: 2 bit/symbol이므로 symbol을 더 자주 전송해야 한다.
- 16-QAM: 4 bit/symbol이므로 symbol rate가 QPSK의 절반이다.

자료에서는 QPSK가 동일 bit rate를 위해 16-QAM보다 2배 빠르게 symbol을 전송한다고 설명한다.

## Spectrum 비교

roll-off factor가 0이면 bandwidth는 symbol rate에 직접 비례한다.

동일 bit rate에서:

```text
R_s,QPSK = R_b / 2
R_s,16QAM = R_b / 4
```

따라서:

```text
B_QPSK ≈ 2 B_16QAM
```

자료의 spectrum 비교에서도 QPSK 대역폭은 약 120 Hz, 16-QAM 대역폭은 약 60 Hz로 나타난다.

## BER 비교

16-QAM의 BER 성능은 QPSK보다 떨어진다.

이유:

- QPSK는 4개 점이 상대적으로 멀리 떨어져 있다.
- 16-QAM은 같은 average bit energy에서 16개 점을 더 좁은 영역에 배치한다.
- minimum Euclidean distance가 작아져 AWGN과 fading에 더 취약하다.

비교 결과의 방향:

| 채널 | QPSK | 16-QAM |
|---|---|---|
| AWGN | 더 낮은 BER | 더 높은 BER |
| Fading | 더 낮은 BER | 더 높은 BER |

fading 환경에서는 두 방식 모두 AWGN보다 BER이 나빠지지만, 16-QAM의 취약성이 더 두드러진다.

## Trade-off 정리

| 항목 | QPSK | 16-QAM |
|---|---|---|
| bits/symbol | 2 | 4 |
| 같은 bit rate에서 bandwidth | 큼 | 작음 |
| constellation 간격 | 넓음 | 좁음 |
| BER 성능 | 좋음 | 나쁨 |
| power 효율 | 상대적으로 좋음 | 같은 BER에 더 높은 SNR 필요 |
| bandwidth 효율 | 낮음 | 높음 |

## 시험 포인트

- QPSK와 16-QAM의 bits/symbol 차이를 계산한다.
- 같은 bit rate에서 16-QAM의 bandwidth가 QPSK의 절반이 되는 이유를 설명한다.
- 16-QAM이 BER에서 불리한 이유를 constellation distance로 설명한다.
- bandwidth efficiency와 error performance의 trade-off를 정리한다.

## 같이 보면 좋은 노트

- [[03 Bandpass Transmission - 디지털 변조와 검파]]
- [[05 Error Performance - Bandpass BER 성능]]
- [[09 Channel Model - Multipath Fading과 Equalization]]
