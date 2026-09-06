# Diffusion Optical Absorption Contacts PN Junction - 확산 흡수 접촉 PN접합

tags: #ElectronicMaterials #Diffusion #OpticalAbsorption #SchottkyContact #OhmicContact #PNJunction

이전: [[10 Intrinsic Extrinsic Semiconductors - 반도체 캐리어]]  
다음: [[12 PN Diode Operation and Special Diodes - PN 다이오드]]

## 핵심 요약

- carrier diffusion은 농도가 높은 곳에서 낮은 곳으로 이동하는 현상이다.
- 전기장과 농도구배가 동시에 있으면 carrier는 drift와 diffusion을 함께 겪는다.
- 비균일 doping은 diffusion을 만들고, 이에 대응하는 built-in electric field가 생겨 steady state를 만든다.
- optical absorption은 `h nu >= E_g` 조건에서 electron-hole pair를 만들며, absorption coefficient로 band gap을 추정할 수 있다.
- metal-semiconductor contact는 work function 관계에 따라 Schottky contact 또는 Ohmic contact가 된다.
- PN junction은 carrier diffusion과 recombination으로 depletion region과 built-in potential을 형성한다.

## Carrier diffusion

### Electron diffusion

- 전자 농도 `n(x,t)`가 위치에 따라 다르면 전자는 높은 농도에서 낮은 농도로 확산한다.
- 어떤 위치 `x0`에서 왼쪽 농도가 오른쪽보다 높으면 왼쪽에서 오른쪽으로 넘어오는 전자가 더 많아진다.
- 전자 flux 방향과 conventional current 방향은 전하 부호 때문에 반대다.

```text
electron diffusion flux proportional to -dn/dx
electron diffusion current proportional to +dn/dx
```

### Hole diffusion

- 정공 농도 `p(x,t)`가 높은 곳에서 낮은 곳으로 정공이 확산한다.
- 정공은 양전하 carrier이므로 hole diffusion 방향과 conventional current 방향이 같다.

```text
hole diffusion flux proportional to -dp/dx
hole diffusion current proportional to -dp/dx
```

## Drift와 diffusion의 결합

- 전기장이 있으면 carrier는 drift한다.
- 농도구배가 있으면 carrier는 diffusion한다.
- 광생성으로 excess electron-hole pair가 특정 영역에 많이 생기면 `n`, `p` 농도구배가 커져 diffusion current가 증가한다.

```text
total current = drift current + diffusion current
```

## Doping variation과 built-in potential

- donor 농도가 위치에 따라 감소하면 전자는 농도가 높은 영역에서 낮은 영역으로 확산한다.
- 전자가 빠져나간 쪽에는 ionized donor가 남아 양전하가 드러난다.
- 이 전하분리가 built-in electric field를 만든다.
- steady state에서는 diffusion이 만드는 이동과 built-in field가 만드는 drift가 균형을 이룬다.

## Optical absorption

### Electron-hole pair 생성

- photon energy가 band gap 이상이면 valence band 전자가 conduction band로 올라가 electron-hole pair가 생성된다.

```text
h nu >= E_g
```

- 여기된 전자는 excess energy를 lattice vibration으로 잃고, conduction band 바닥 근처 평균 에너지로 thermalize된다.
- 강의에서는 conduction band 내 평균 에너지를 `3/2 kT` 수준으로 설명한다.

### Absorption coefficient와 band gap

- absorption coefficient `alpha`는 photon energy 또는 wavelength에 따라 달라진다.
- `h nu`가 `E_g`보다 커지기 시작하는 구간에서 흡수가 급격히 증가한다.
- density of states는 band edge에서부터 증가하므로 photon energy가 높아질수록 가능한 전이 상태가 많아진다.
- `alpha`와 photon energy의 관계를 측정하면 band gap을 추정할 수 있다.

## Piezoresistivity

- piezoresistivity는 stress가 가해질 때 재료의 resistivity가 변하는 현상이다.
- 반도체에서는 band structure와 carrier mobility가 stress에 민감해 효과가 크다.
- force sensor, pressure sensor, strain gauge, accelerometer, microphone에 사용된다.
- cantilever support 부근의 stress를 piezoresistor가 저항 변화로 읽어 힘을 추정할 수 있다.

## Metal-Semiconductor contact

### 분류 기준

- metal work function은 금속 고유 성질로 본다.
- semiconductor work function은 doping과 Fermi level 위치에 따라 달라진다.
- 두 work function 관계와 band bending이 contact의 I-V 특성을 결정한다.

| 접촉 | I-V | 의미 |
|---|---|---|
| Schottky contact | rectifying | 한 방향 전류가 우세한 barrier contact |
| Ohmic contact | linear `V = IR` | 양방향 carrier 이동이 쉬운 non-rectifying contact |

## Schottky contact

### n-type 기준 조건

- n-type semiconductor에서 `Phi_M > Phi_S`이면 Schottky contact가 형성된다.
- semiconductor 쪽에 electron depletion region이 생긴다.
- built-in potential에 의해 band bending이 나타난다.
- 열평형에서는 Fermi level이 평탄하고 net carrier flow는 0이다.

### Doping dependence

- doping이 증가해도 ideal equilibrium barrier height 자체는 크게 변하지 않는다.
- 그러나 depletion width와 barrier width가 감소한다.
- 매우 높은 doping, 예를 들어 `> 10^17 cm^-3` 수준에서는 barrier가 얇아져 tunneling이 가능해지고 양방향 carrier action이 나타날 수 있다.

### Schottky effect

- 외부 전기장이 conductor surface의 barrier를 낮추어 thermionic emission을 증가시키는 현상이다.

```text
Phi_eff = Phi - sqrt(e^3 E / (4 pi epsilon_0))
```

- 전기장 `E`가 커질수록 effective work function이 낮아져 방출이 쉬워진다.

### 응용

- Schottky diode는 forward voltage drop이 낮고 switching이 빠르다.
- 일반 PN diode의 `V_F`가 약 0.6-0.7 V라면 Schottky barrier diode는 약 0.2-0.3 V 수준으로 다뤄진다.
- reverse-biased Schottky photodiode는 빠른 photodetector로 쓰인다.
- metal-semiconductor depletion region에서 광생성 carrier를 빠르게 수집할 수 있다.

## Ohmic contact

### n-type 기준 조건

- n-type semiconductor에서 `Phi_M < Phi_S`이면 electron accumulation region이 생기며 barrier가 거의 없어 carrier 이동이 쉽다.
- MOSFET의 source, drain, gate contact는 carrier 주입과 추출을 위해 Ohmic contact로 설계한다.

### p-type과 n-type의 열전 응용

- 금속과 n-type 또는 p-type semiconductor의 Ohmic contact에서는 전류 방향에 따라 heat absorption과 heat dissipation 위치가 달라진다.
- Peltier effect를 이용한 thermoelectric device는 넓은 Ohmic contact 면적을 사용해 냉각과 방열 효율을 높인다.
- ceramic layer는 전기 절연과 열 전달을 동시에 만족시키기 위해 사용된다.

## PN junction 형성

### 전기적 중성

- p-type semiconductor는 acceptor ion과 hole 수가 같아 전체적으로 중성이다.
- n-type semiconductor는 donor ion과 free electron 수가 같아 전체적으로 중성이다.

### 접합 직후 변화

- p쪽의 hole은 n쪽으로 확산하고, n쪽의 electron은 p쪽으로 확산한다.
- 접합 근처에서 electron과 hole이 recombination한다.
- mobile carrier가 사라진 영역에는 고정된 ionized acceptor와 donor가 남는다.
- 이 공간전하 영역이 depletion region이다.

### Built-in electric field와 band diagram

- depletion region의 고정 전하는 n쪽에서 양전하, p쪽에서 음전하를 만들어 electric field를 형성한다.
- 이 field는 추가 diffusion을 막는 방향의 drift를 만든다.
- 열평형에서는 Fermi level이 전체 PN junction에서 일정하다.
- band diagram에서는 built-in potential `V_bi`에 해당하는 band bending이 나타난다.

## Depletion approximation

- depletion region 안에서는 mobile carrier 농도 `n`, `p`가 매우 작다고 가정한다.
- depletion region 밖의 quasi-neutral region에서는 charge density를 0으로 둔다.

```text
rho = -q N_A  for -x_p <= x <= 0
rho = +q N_D  for 0 <= x <= x_n
rho = 0 outside depletion region
```

- 이 근사를 쓰면 charge density, electric field, electrostatic potential의 closed-form solution을 구할 수 있다.

## 시험 포인트

- electron diffusion current와 hole diffusion current의 부호 차이.
- drift와 diffusion이 동시에 있을 때 total current를 해석하는 방법.
- 비균일 doping에서 built-in field가 생기는 이유.
- absorption coefficient로 band gap을 측정하는 직관.
- Schottky contact와 Ohmic contact의 work function 조건.
- Schottky diode가 PN diode보다 forward drop이 낮고 빠른 이유.
- PN junction에서 depletion region과 built-in potential이 형성되는 순서.
- depletion approximation의 가정.

