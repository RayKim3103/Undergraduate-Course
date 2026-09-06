# PN Diode Operation and Special Diodes - PN 다이오드

tags: #ElectronicMaterials #PNDiode #ZenerDiode #Varicap #PIN #LED #TunnelDiode

이전: [[11 Diffusion Optical Absorption Contacts PN Junction - 확산 흡수 접촉 PN접합]]  
다음: [[13 MOS Capacitor and MOSFET - MOS 커패시터와 MOSFET]]

## 핵심 요약

- PN junction diode fabrication은 oxidation, lithography, diffusion, metallization으로 이어진다.
- forward bias는 built-in potential과 depletion width를 줄여 전류를 증가시킨다.
- reverse bias는 potential barrier와 depletion width를 키우고, minority carrier 중심의 작은 전류만 흐르게 한다.
- 실제 diode 응용에는 Zener, varicap, PIN, LED, tunnel diode가 있으며, 각자 다른 물리 현상을 이용한다.
- tunnel diode는 heavy doping으로 매우 얇은 depletion region을 만들고 quantum tunneling에 의해 negative differential resistance를 보인다.

## PN junction diode fabrication

1. damage-free single-crystal Si wafer에서 시작한다.
2. thermal oxidation으로 SiO2 diffusion barrier를 만든다.
3. 첫 lithography로 oxide에 diffusion window를 연다.
4. phosphorus diffusion으로 표면에 n+-p junction을 만든다.
5. sputtering으로 Al metallization을 형성해 외부 연결을 만든다.
6. 두 번째 lithography로 junction 영역 외부의 불필요한 metal을 제거한다.

## Ideal diode operation

### Equilibrium, `V_A = 0`

- p-type과 n-type 사이에는 built-in potential `V_bi`와 depletion region이 존재한다.
- 그러나 열평형에서는 Fermi level이 같기 때문에 net current는 0이다.
- diffusion과 drift가 서로 균형을 이룬다.

### Forward bias, `V_A = V_f > 0`

- 외부 전압이 built-in potential을 낮춘다.

```text
barrier energy = e (V_bi - V_f)
```

- depletion region width가 감소한다.
- majority carrier가 junction을 넘어 상대 영역으로 주입된다.
- 주입된 carrier는 반대쪽 quasi-neutral region에서 minority carrier가 되고, 확산하면서 recombination으로 감소한다.

### Reverse bias, `V_A = V_r < 0`

- 외부 전압이 built-in potential을 증가시킨다.

```text
barrier energy = e (V_bi + |V_r|)
```

- depletion region width가 증가한다.
- majority carrier injection은 억제된다.
- depletion region은 minority carrier를 수집하는 sink처럼 동작한다.

## Diode equation

강의 슬라이드에서는 이상 다이오드 전류의 지수 형태가 반복적으로 제시된다.

```text
I = I_0 [exp(V_A / V_ref) - 1]
```

- `I_0`는 reverse saturation current에 해당한다.
- `V_ref`는 thermal voltage 또는 ideality를 포함한 기준 전압으로 해석할 수 있다.
- forward bias에서는 지수적으로 전류가 커지고, reverse bias에서는 거의 `-I_0`에 접근한다.

## Junction voltage drop

- 외부에서 가한 전압 `V_A`는 대부분 depletion region에 걸린다고 근사한다.
- forward bias에서는 junction barrier가 `V_bi - V_A`가 된다.
- reverse bias에서는 barrier가 증가한다.
- quasi-neutral region은 전하 중성이므로 전위 변화가 상대적으로 작다고 본다.

## Current density in PN diode

- forward-biased PN diode 내부 전류는 electron current와 hole current의 합이다.
- 위치에 따라 electron current와 hole current 비율이 달라져도 total current density는 일정해야 한다.
- depletion region을 지난 minority carrier는 quasi-neutral region에서 diffusion하며 recombination한다.

## Carrier concentration under bias

### Forward bias

- barrier가 낮아져 p쪽 hole과 n쪽 electron이 각각 반대쪽으로 많이 주입된다.
- 반대쪽으로 넘어간 majority carrier는 그 영역에서 minority carrier가 된다.
- minority carrier 농도는 접합 근처에서 크고, 깊이 들어갈수록 recombination으로 감소한다.

### Reverse bias

- depletion region이 minority carrier를 빠르게 쓸어가므로 접합 근처 minority carrier 농도가 낮아진다.
- reverse current는 주로 thermal generation으로 생긴 minority carrier 수집에 의해 결정된다.

## Various types of diode

| Diode | 이용 특성 | 대표 응용 |
|---|---|---|
| Zener | reverse breakdown에서 거의 일정 전압 | regulator, ESD protection |
| Varicap | reverse bias에 따른 junction capacitance 변화 | RF tuning |
| PIN | intrinsic layer의 저항과 capacitance 차이 | RF switch, photodiode |
| LED | electron-hole recombination light emission | display, lamp |
| Tunnel diode | tunneling과 negative differential resistance | high-speed switching, high-frequency device |

## Zener diode

- heavily doped PN junction으로 reverse bias에서 정해진 breakdown voltage `V_BR`에 도달하면 전류가 흐른다.
- 일정한 reverse voltage를 유지하는 특성 때문에 voltage regulator로 사용한다.
- 높은 field를 견디며 ESD 보호 소자로도 활용된다.

## Varicap diode

- reverse-biased PN junction의 depletion region은 capacitor처럼 동작한다.
- reverse bias `V_R`가 커지면 depletion width `W_d`가 증가한다.
- capacitance는 depletion width에 반비례하므로 감소한다.

```text
V_R up -> W_d up -> C_d down
```

- N- layer를 capacitance control layer로 넣어 전압에 따른 capacitance 변화를 설계한다.

## PIN diode

- p-region과 n-region 사이에 undoped intrinsic region이 들어간 diode다.
- intrinsic layer는 reverse bias에서 넓은 depletion/absorption 영역을 제공한다.
- forward bias에서는 낮은 RF resistance, reverse bias에서는 높은 resistance와 낮은 capacitance를 이용할 수 있다.
- photodiode에서는 intrinsic layer가 photo-absorption layer로 작동한다.

## Light Emitting Diode

- forward bias에서 전자와 정공이 접합 영역으로 주입되어 recombination한다.
- recombination energy가 photon으로 방출되면 빛이 나온다.
- 빛의 색은 compound semiconductor의 band gap에 의해 결정된다.

```text
photon energy ≈ E_g
shorter wavelength -> larger E_g
```

- brightness는 대체로 current에 비례해 증가한다.
- InGaN은 청색/녹색 계열, AlGaAs는 적색 계열 LED 재료로 다뤄진다.

## PN diode와 Schottky diode 비교

- p+-n diode에서는 작은 forward bias에서 depletion region recombination이, 더 큰 forward bias에서 p+쪽에서 n쪽으로의 hole injection이 중요한 전류 성분이 된다.
- MS Schottky diode에서는 semiconductor에서 metal로 넘어가는 majority carrier thermionic emission이 주된 전류다.
- Schottky diode는 minority carrier storage가 작아 switching이 빠르다.

## Tunnel diode

### 구조와 특징

- tunnel diode는 p와 n 영역이 모두 매우 높게 doping된 degenerate PN junction이다.
- depletion region이 매우 얇아져 전자가 장벽을 통과할 tunneling probability가 커진다.
- threshold voltage가 일반 PN diode보다 낮고 응답이 빠르다.

### Negative differential resistance

- 작은 forward bias에서 n-region conduction band의 전자 상태와 p-region valence band의 빈 상태가 에너지상 잘 맞아 direct tunneling current가 증가한다.
- 특정 bias에서 overlap이 최대가 되어 peak current에 도달한다.
- bias를 더 키우면 대응 가능한 빈 상태와 전자 상태의 overlap이 줄어 tunneling current가 감소한다.
- 이 구간에서 전압이 증가해도 전류가 감소하므로 negative differential resistance가 나타난다.
- 더 큰 forward bias에서는 일반 PN diode의 injection current가 지배적이 되어 전류가 다시 증가한다.

## 시험 포인트

- PN diode fabrication의 주요 공정 순서.
- forward bias와 reverse bias에서 barrier height와 depletion width 변화.
- total current density가 diode 내부에서 일정해야 하는 이유.
- Zener, varicap, PIN, LED, tunnel diode의 이용 물리와 응용.
- Schottky diode와 PN diode의 dominant current 차이.
- tunnel diode의 I-V 곡선에서 NDR이 생기는 band overlap 설명.

