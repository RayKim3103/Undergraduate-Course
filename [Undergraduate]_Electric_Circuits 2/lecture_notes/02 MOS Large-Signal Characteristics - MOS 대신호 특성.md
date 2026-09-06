---
과목: Electric Circuits 2
유형: Lecture Note
주제: MOSFET 대신호 I-V, triode, saturation, cutoff, channel-length modulation, body effect
tags:
  - electric-circuits-2
  - mosfet
  - large-signal
  - saturation
  - triode
---

# MOS Large-Signal Characteristics - MOS 대신호 특성

## 핵심 요약

이 강의는 NMOS/PMOS의 large-signal I-V 특성을 복습한다. MOSFET은 gate oxide를 사이에 둔 capacitor 구조와 source-channel-drain 구조를 가지며, `VGS`, `VDS`, `VTH` 조건에 따라 cutoff, triode, saturation 영역으로 나뉜다. 실제 소자에서는 subthreshold leakage, channel-length modulation, body effect, temperature effect가 ideal model에서 벗어나게 만든다.

## MOSFET 구조

MOSFET:

- Metal-Oxide-Semiconductor Field-Effect Transistor
- NMOS와 PMOS로 구분
- 수직 방향: metal-oxide-semiconductor capacitor
- 수평 방향: source-channel-drain conduction path

처음에는 channel이 없고, gate voltage가 threshold 이상이 되면 inversion channel이 형성되어 전류가 흐른다.

## NMOS 동작 영역

### Cutoff

```text
VGS < VTH
ID = 0
```

ideal model에서는 전류가 없지만 실제 modern MOSFET에서는 subthreshold current가 존재한다.

### Triode

조건:

```text
VGS > VTH
VDS < VGS - VTH
```

전류:

```text
ID = mu_n Cox (W/L) [(VGS - VTH)VDS - VDS^2/2]
```

강의 슬라이드에서는 핵심적으로 `VDS`에 따라 current가 증가하는 영역으로 설명한다.

### Saturation

조건:

```text
VGS > VTH
VDS > VGS - VTH
```

pinch-off 조건:

```text
VGD = VTH
VDS = VGS - VTH
```

전류:

```text
ID = (1/2) mu_n Cox (W/L) (VGS - VTH)^2
```

`VOV = VGS - VTH`를 overdrive voltage라고 한다.

## PMOS 동작 영역

PMOS는 NMOS와 polarity가 반대이다. 보통 source에서 나오는 drain current 기준으로 정의하고, `VSG`, `VSD`, `|VTH|`를 사용한다.

### Cutoff

```text
VSG < |VTH|
ID = 0
```

### Triode

```text
VSG > |VTH|
VSD < VSG - |VTH|
```

### Saturation

```text
VSG > |VTH|
VSD > VSG - |VTH|
```

전류:

```text
ID = (1/2) mu_p Cox (W/L) (VSG - |VTH|)^2
```

## Ideal Model에서 벗어나는 효과

### Subthreshold Current

cutoff에서도 source-drain 사이에 leakage current가 흐른다.

특징:

- 작은 MOSFET일수록 더 중요하다.
- modern digital circuits에서 큰 문제가 된다.
- 이 과목의 손계산에서는 주로 무시하지만 simulation에서는 반영된다.

### Channel-Length Modulation

saturation에서도 `VDS`가 증가하면 실제 channel length가 줄어들어 `ID`가 증가한다.

모델:

```text
ID = (1/2) mu_n Cox (W/L) (VGS - VTH)^2 (1 + lambda VDS)
```

이는 BJT의 Early effect와 비슷하다. 필요할 때 회로 해석에 포함한다.

### Body Effect

body voltage가 source와 다르면 threshold voltage가 변한다.

```text
VTH = VTH0 + gamma (sqrt(2 phi_F + VSB) - sqrt(2 phi_F))
```

IC에서는 body가 여러 transistor와 공유되므로 source와 항상 묶을 수 없다.

- NMOS body: 가장 낮은 전위에 연결
- PMOS body: 가장 높은 전위에 연결

해석에서는 단순화를 위해 무시할 때가 많지만 simulation에는 반영한다.

### Temperature Effect

MOSFET parameter들은 온도에 의존한다. 보통 온도가 올라가면 mobility 감소 등으로 `ID`가 줄어드는 효과가 나타난다.

## Two-Track Approach

현대 MOSFET은 모델 파라미터가 매우 많다. 강의에서는 두 접근을 병행한다.

- 손계산: 단순하고 직관적인 model
- simulation: 복잡하지만 정확한 SPICE model

## 시험 포인트

- NMOS/PMOS의 cutoff, triode, saturation 조건을 정확히 쓴다.
- pinch-off 조건 `VDS = VGS - VTH`를 이해한다.
- overdrive voltage `VOV`의 의미를 안다.
- channel-length modulation이 `ro`와 small-signal gain에 연결됨을 기억한다.
- body effect는 `VSB`가 threshold를 바꾸는 현상이다.

## 같이 보면 좋은 노트

- [[03 MOS Small-Signal Characteristics - MOS 소신호 모델]]
- [[04 Common-Source Amplifier - CS 증폭기]]
- [[07 Bias Circuits and Current Mirrors - 바이어스와 전류미러]]

