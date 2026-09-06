# MOS Capacitor and MOSFET - MOS 커패시터와 MOSFET

tags: #ElectronicMaterials #MOSCapacitor #MOSFET #ThresholdVoltage #Scaling

이전: [[12 PN Diode Operation and Special Diodes - PN 다이오드]]  
다음: 없음

## 핵심 요약

- MOS capacitor는 metal, oxide, semiconductor 구조에서 gate voltage로 semiconductor surface charge를 조절하는 소자다.
- ideal MOS는 완전 절연막, 계면 전하 없음, 균일 doping, field-free bulk, Ohmic back contact를 가정한다.
- 실제 MOS에서는 metal과 semiconductor work function 차이와 interface charge 때문에 flat-band voltage가 생긴다.
- gate bias에 따라 accumulation, depletion, inversion이 나타난다.
- MOSFET은 gate field가 source-drain 사이 channel을 만들고 조절하는 field effect transistor다.
- scaling은 집적도와 전력 측면에서 유리하지만 SCE, HCE, DIBL, gate leakage를 유발한다.

## Ideal MOS capacitor

### 구조

- MIM capacitor: metal-insulator-metal 구조다.
- MOS capacitor: metal-insulator-semiconductor 구조다.
- oxide는 보통 SiO2 같은 절연막으로 생각한다.

```text
C = epsilon A / d = dQ / dV
```

- MOS capacitance는 gate voltage 변화에 대한 semiconductor surface charge 변화로 해석한다.
- metal에는 자유전자가 많고, semiconductor에는 mobile carrier와 fixed ionized dopant가 함께 존재한다.

### Ideal MOS 가정

- metallic gate가 충분히 두꺼워 bias를 잘 전달한다.
- oxide가 완전한 절연체다.
- SiO2 내부와 SiO2/semiconductor interface에 charge center가 없다.
- semiconductor는 균일하게 doped되어 있다.
- substrate가 충분히 두꺼워 내부에 field-free region이 존재한다.
- metal과 semiconductor back contact가 Ohmic contact다.

## Real MOS capacitor

- 실제 MOS에서는 `Phi_M`과 `Phi_S`가 같지 않아 `V_G = 0`에서도 band bending이 생길 수 있다.
- 강의의 p-type 예에서는 `Phi_M < Phi_S`인 경우를 다룬다.
- band를 평탄하게 만드는 gate voltage를 flat-band voltage `V_FB`라고 한다.

```text
V_FB = Phi_MS - Q_i / C_ox
Phi_MS = Phi_M - Phi_S
Q_i: interface state charge density
C_ox: oxide capacitance
```

- interface charge `Q_i`가 있으면 work function 차이만으로 예측한 flat-band voltage에서 추가 shift가 생긴다.

## MOS capacitor operation

### p-type substrate

| Gate bias | 동작 | 표면 carrier |
|---|---|---|
| `V_G < 0` | accumulation | hole 축적 |
| 작은 `V_G > 0` | depletion | hole 감소, ionized acceptor 노출 |
| 큰 `V_G > 0` | inversion | electron inversion layer 형성 |

- 음의 gate bias는 p-type 표면에 hole을 끌어모아 accumulation을 만든다.
- 양의 gate bias는 hole을 밀어내 depletion region을 만들고, 충분히 크면 minority electron이 표면에 모여 inversion layer를 만든다.

### n-type substrate

| Gate bias | 동작 | 표면 carrier |
|---|---|---|
| `V_G > 0` | accumulation | electron 축적 |
| 작은 `V_G < 0` | depletion | electron 감소, ionized donor 노출 |
| 큰 `V_G < 0` | inversion | hole inversion layer 형성 |

### Depletion width

- depletion에서는 majority carrier concentration이 semiconductor background doping concentration보다 낮아진다.
- inversion에 가까워질수록 depletion width가 증가하지만, strong inversion 이후에는 최대 depletion width 근처에서 inversion charge가 주로 증가한다.
- doping concentration이 커질수록 maximum depletion width는 작아진다.

## MOSFET fundamentals

### 발명 흐름

- Audion vacuum tube는 정류와 증폭이 가능했지만 크기, 전력소모, 발열, 신뢰성 문제가 컸다.
- Bell Labs의 Bardeen, Brattain, Shockley는 Ge point-contact transistor를 만들었다.
- 이후 BJT, JFET을 거쳐 1959년 Kahng과 Atalla가 MOSFET을 발명했다.
- MOSFET은 낮은 전력소모, 높은 집적도, field effect 제어 때문에 현대 집적회로의 기본 소자가 되었다.

### MOSFET 구조

- 기본 terminal은 source, drain, gate다.
- gate는 oxide를 통해 channel과 전기적으로 절연되어 있지만 전기장으로 channel charge를 제어한다.
- accumulation 또는 depletion만으로는 source-drain 사이 conducting channel이 충분하지 않다.
- inversion이 형성되어야 current가 흐른다.

## NMOSFET과 PMOSFET

| 항목 | NMOSFET | PMOSFET |
|---|---|---|
| substrate | p-type 또는 P-well | n-type 또는 N-well |
| source/drain | n+ | p+ |
| channel | electron inversion channel | hole inversion channel |
| 동작 bias | `V_GS > V_TH`, `V_DS > 0` | `V_GS < V_TH`, `V_DS < 0` |
| conventional current | drain to source | source to drain |
| 전류 수준 | PMOS보다 큼 | hole mobility 때문에 작음 |

- 강의에서는 NMOS current가 PMOS current보다 약 2배 크다고 설명한다.
- 이는 electron mobility가 hole mobility보다 크기 때문이다.

## MOSFET operation

### Linear region

- `V_D < V_D,sat`에서는 channel이 resistor처럼 동작한다.
- drain voltage가 증가하면 drain current가 거의 비례해 증가한다.

```text
V_D,sat = V_G - V_TH
```

### Pinch-off와 saturation

- `V_D = V_D,sat`가 되면 drain 끝의 inversion charge가 거의 0이 되어 pinch-off가 시작된다.
- `V_D > V_D,sat`에서는 pinch-off region이 생기지만, source에서 drain 쪽으로 도달하는 전자 수가 크게 변하지 않아 이상적으로 `I_D`가 포화된다.
- pinch-off 지점에서는 electric field와 electron velocity가 매우 크다.

### Inversion charge

```text
Q_i(x) = -C_ox [V_GS - V_T - V(x)]
```

- `V(x) = V_GS - V_T`인 위치에서는 `Q_i(x) ≈ 0`이 되어 pinch-off 조건이 된다.
- 실제로 charge가 완전히 0이라기보다 매우 작아지고, electron은 high-field region을 빠르게 통과한다.

## I-V characteristics

- transfer characteristics: `V_D`를 고정하고 `V_G`를 변화시키며 `I_D`를 측정한다.
- output characteristics: `V_G`를 고정하고 `V_D`를 변화시키며 `I_D`를 측정한다.
- 낮은 `V_DS`에서는 linear region parameter를, 높은 `V_DS`에서는 saturation parameter를 추출한다.

## Subthreshold swing

```text
S.S = dV_G / dlog(I_D)  [V/dec]
```

- drain current를 한 자리수, 즉 한 decade 변화시키는 데 필요한 gate voltage 변화량이다.
- S.S가 작을수록 off에서 on으로 더 급격히 전환된다.
- 작은 S.S는 낮은 전압 동작과 낮은 off leakage 측면에서 유리하다.

## Linear region parameter extraction

```text
I_D = mu_lin C_ox (W/L) [(V_G - V_TH) V_D - V_D^2 / 2]
G_m = dI_D / dV_G = mu_lin C_ox (W/L) V_D
mu_lin = G_m,max / [C_ox V_D (W/L)]
V_TH,lin = V_G - I_D/G_m - V_D/2  at G_m,max
```

- low drain voltage 조건에서 channel이 resistor처럼 동작한다고 보고 mobility와 threshold voltage를 추출한다.
- `G_m`은 gate voltage 변화가 drain current를 얼마나 잘 제어하는지 나타내는 transconductance다.

## Saturation region parameter extraction

```text
I_D = mu_sat C_ox (W / 2L) (V_G - V_TH)^2
Grad = d sqrt(I_D) / dV_G = [mu_sat C_ox (W / 2L)]^1/2
mu_sat = (1/C_ox) (2L/W) (Grad_max)^2
V_TH,sat = V_G - sqrt(I_D) / Grad  at Grad_max
```

- saturation 영역에서는 `sqrt(I_D)`와 `V_G`의 선형성을 이용해 threshold voltage를 추출한다.

## High drain current를 얻는 방법

```text
I_D = (mu C_ox / 2) (W/L) (V_GS - V_TH)^2
```

- `V_TH`를 낮추면 on-current가 증가하지만 off-current와 standby power가 증가할 수 있다.
- `W`를 키우면 current가 증가하지만 면적이 커져 scaling에 불리하다.
- `L`을 줄이면 current가 증가하지만 short-channel effect와 leakage 문제가 커진다.

## MOSFET scaling issues

### Scaling의 필요성

- channel length와 oxide thickness를 줄이면 더 많은 transistor를 같은 면적에 집적할 수 있다.
- parasitic capacitance와 동작 전압을 낮추면 power consumption을 줄일 수 있다.

### Short Channel Effect

- channel length가 짧아지면 같은 `V_DS`에서도 channel electric field가 커진다.
- source-substrate와 drain-substrate depletion region이 가까워지면 punch-through leakage가 생길 수 있다.
- short channel에서는 gate뿐 아니라 drain도 channel barrier를 강하게 흔든다.

### Hot Carrier Effect

- high electric field에서 electron이 큰 에너지를 얻으면 hot carrier가 된다.
- hot electron은 gate oxide에 주입되어 oxide trap과 interface state를 만들 수 있다.
- hot hole은 substrate 쪽으로 주입될 수 있다.
- 결과적으로 `S.S`, `V_th`, mobility 같은 electrical performance가 열화된다.

### Drain-Induced Barrier Lowering

- MOSFET channel은 source에서 drain으로 넘어가는 potential barrier를 gate voltage로 조절하는 구조다.
- short-channel transistor에서는 높은 drain voltage가 이 barrier를 낮출 수 있다.
- gate bias가 없어도 barrier가 낮아져 `V_th`가 감소하고 S.S가 악화된다.

### Gate leakage current

| 메커니즘 | 조건 | 의미 |
|---|---|---|
| Thermionic emission | 온도 의존 | 열에너지 큰 전자가 barrier를 넘음 |
| Fowler-Nordheim tunneling | 높은 electric field | triangular barrier를 통해 oxide로 터널링 |
| Direct tunneling | ultra-thin oxide | 얇은 oxide를 직접 터널링 |

- oxide가 얇아질수록 gate leakage가 증가하고 standby power와 reliability 문제가 커진다.

## 시험 포인트

- ideal MOS capacitor 가정과 real MOS의 flat-band voltage.
- p-type과 n-type substrate에서 accumulation, depletion, inversion bias 조건.
- MOSFET이 inversion channel에서만 전류를 잘 흘리는 이유.
- NMOS와 PMOS의 bias, carrier, current level 차이.
- pinch-off와 saturation의 물리적 의미.
- subthreshold swing의 정의와 작을수록 좋은 이유.
- linear/saturation mobility와 threshold voltage 추출식.
- `V_TH` 감소, `W` 증가, `L` 감소가 high current와 leakage에 주는 trade-off.
- SCE, HCE, DIBL, gate leakage의 원인과 결과.

