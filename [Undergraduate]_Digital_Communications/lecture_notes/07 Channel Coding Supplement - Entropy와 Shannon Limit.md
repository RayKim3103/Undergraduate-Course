---
과목: Digital Communications
유형: Supplement
주제: Entropy, mutual information, channel capacity, Shannon limit, coding trade-off
tags:
  - digital-communications
  - information-theory
  - entropy
  - mutual-information
  - channel-capacity
  - shannon-limit
---

# Channel Coding Supplement - Entropy와 Shannon Limit

## 핵심 요약

이 보조자료는 channel coding의 이론적 배경인 entropy, mutual information, channel capacity, Shannon-Hartley theorem, Shannon limit을 다룬다. 핵심은 coding이 error probability 자체를 없애는 마법이 아니라, capacity 이하의 rate에서 충분히 복잡한 coding을 사용하면 임의로 작은 error probability에 접근할 수 있다는 점이다.

## Entropy

Entropy는 확률변수 `X`를 관측하기 전의 불확실성 또는 관측했을 때 얻는 평균 정보량이다.

```text
H(X) = - sum_x P(X=x) log2 P(X=x)
```

해석:

- 결과가 거의 확실하면 entropy가 작다.
- 결과가 균등하게 불확실하면 entropy가 크다.
- 단위는 bit이다.

## Kullback-Leibler Distance

두 probability mass function `P(X)`, `Q(X)` 사이의 차이를 측정한다.

```text
D(P || Q) = sum_x P(x) log(P(x) / Q(x))
```

성질:

```text
D(P || Q) >= 0
```

등호는 두 분포가 같을 때만 성립한다.

## Conditional Entropy

`Y`를 관측한 뒤에도 `X`에 남아 있는 평균 불확실성이다.

```text
H(X|Y)
```

`Y`가 `X`를 잘 설명할수록 `H(X|Y)`는 작아진다.

## Mutual Information

Mutual information은 `Y`를 관측함으로써 `X`에 대해 얻는 정보량이다.

```text
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
```

성질:

- `I(X;Y) >= 0`
- `X`와 `Y`가 independent이면 `I(X;Y) = 0`
- 통신에서는 입력 `X`가 출력 `Y`에 얼마나 잘 보존되는지를 나타낸다.

## Channel Capacity

채널 capacity `C`는 가능한 입력분포 중 mutual information을 최대로 만드는 값이다.

```text
C = max_{p(x)} I(X;Y)
```

의미:

- error-free communication이 이론적으로 가능한 최대 data rate
- capacity보다 낮은 rate에서는 충분히 좋은 code로 error probability를 임의로 작게 만들 수 있다.
- capacity보다 높은 rate에서는 어떤 code도 임의로 작은 error probability를 보장할 수 없다.

## AWGN Channel Capacity

Shannon-Hartley theorem:

```text
C = W log2(1 + S/N)  [bits/s]
```

여기서:

- `W`: bandwidth
- `S`: average received signal power
- `N = N_0 W`: average noise power

bit energy와 연결하면:

```text
S = E_b C
```

## Shannon Limit

bandwidth를 무한히 키우는 한계에서 reliable communication에 필요한 최소 `Eb/N0`가 나온다.

```text
(E_b/N_0)_min = ln 2 ≈ 0.693 ≈ -1.6 dB
```

의미:

- 이 값보다 낮으면 어떤 정보율에서도 error-free communication이 불가능하다.
- bandwidth만 무한히 키운다고 capacity를 마음대로 크게 만들 수는 없다.

## Channel Coding의 Trade-off

channel coding은 redundancy를 추가해 성능을 바꾸는 도구이다.

대표 trade-off:

| trade-off | 의미 |
|---|---|
| Error performance vs bandwidth | BER을 낮추기 위해 bandwidth를 더 사용 |
| Power vs bandwidth | 송신 power를 줄이는 대신 bandwidth 증가 |
| Data rate vs bandwidth | data rate를 높이면서 coding으로 BER 보완 |
| Capacity vs bandwidth | Shannon capacity 한계 내에서 설계 |

## Low Eb/N0에서의 Code 성능

모든 code는 정정할 수 있는 error 수가 제한되어 있다. 한 block 안에 정정 가능 개수보다 많은 error가 들어오면 성능이 급격히 나빠질 수 있다.

따라서 BER 곡선에는 crossover가 나타날 수 있다.

- 낮은 `Eb/N0`: coding이 오히려 부담이 될 수 있음
- 충분한 `Eb/N0`: coding gain이 나타남
- turbo code 같은 강력한 code는 crossover point가 더 낮을 수 있음

## 통신 표준의 Channel Coding

자료는 4G LTE와 5G NR에서 coding이 표준적으로 사용됨을 언급한다.

일반적 흐름:

- 4G LTE: turbo code 중심
- 5G NR: LDPC, Polar code 등 사용

핵심은 실제 이동통신 표준에서도 channel coding이 power, bandwidth, reliability trade-off를 조절하는 핵심 블록이라는 점이다.

## 시험 포인트

- entropy와 mutual information의 정의를 구분한다.
- channel capacity는 `max I(X;Y)`라는 점을 기억한다.
- Shannon-Hartley capacity 식 `C = W log2(1 + S/N)`을 이해한다.
- Shannon limit `-1.6 dB`의 의미를 설명할 수 있어야 한다.
- coding은 bandwidth 비용으로 BER/power/data-rate 성능을 조절한다.

## 같이 보면 좋은 노트

- [[06 Channel Coding - 오류 제어 부호]]
- [[05 Error Performance - Bandpass BER 성능]]
