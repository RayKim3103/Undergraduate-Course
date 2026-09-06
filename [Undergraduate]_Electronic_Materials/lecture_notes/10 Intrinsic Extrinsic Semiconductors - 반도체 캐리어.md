# Intrinsic Extrinsic Semiconductors - 반도체 캐리어

tags: #ElectronicMaterials #Semiconductor #Doping #CarrierGeneration #Recombination

이전: [[09 Fermi Statistics Emission Phonons - 페르미 방출 포논]]  
다음: [[11 Diffusion Optical Absorption Contacts PN Junction - 확산 흡수 접촉 PN접합]]

## 핵심 요약

- 반도체는 금속과 절연체 사이의 band gap을 가지며, 온도, 빛, doping에 따라 carrier 농도가 크게 변한다.
- intrinsic semiconductor에서는 thermal 또는 optical excitation으로 electron-hole pair가 생성된다.
- n-type doping은 donor level을 통해 electron을 제공하고, p-type doping은 acceptor level을 통해 hole을 만든다.
- 열평형에서는 mass action law `np = n_i^2`가 성립한다.
- 온도에 따라 freeze-out, extrinsic, intrinsic 영역이 나타난다.
- recombination과 minority carrier lifetime은 photoconductivity와 photoresponse를 결정한다.

## Intrinsic semiconductor

### Si 결정과 band diagram

- Si 원자는 네 개의 sp3 hybrid orbital로 이웃 Si와 공유결합을 이룬다.
- 0 K에서는 valence band가 가득 차고 conduction band는 비어 있다.
- 유한 온도나 photon absorption으로 전자가 valence band에서 conduction band로 올라가면 electron-hole pair가 생긴다.

### Carrier generation

- photon energy가 band gap보다 크면 전자가 valence band에서 conduction band로 여기된다.

```text
h nu >= E_g
```

- photon이 Si-Si 결합을 깨면 자유전자와 정공이 동시에 생성된다.
- 열진동도 공유결합을 깨 electron-hole pair를 만들 수 있다.
- 정공은 실제 양전하 입자가 아니라, 이웃 결합 전자가 빈 결합 자리로 이동하면서 양전하처럼 이동하는 유효 carrier다.

### 전기장 아래의 전도

- 전기장이 걸리면 conduction band 전자와 valence band 정공이 drift하며 전류에 기여한다.
- 전위 `V(x)`가 변하면 전자의 electrostatic potential energy `-eV(x)`도 변해 band diagram이 공간적으로 기울어진다.

## 열평형과 mass action law

- 열평형에서 intrinsic, n-type, p-type 모두 다음 관계를 만족한다.

```text
n p = n_i^2
```

- intrinsic semiconductor에서는 `n = p = n_i`.
- n-type에서는 electron이 majority carrier, hole이 minority carrier다.
- p-type에서는 hole이 majority carrier, electron이 minority carrier다.

## Extrinsic semiconductor

### n-type doping

- As 같은 Group V 원소는 Si 자리에 치환되면 네 전자는 결합에 참여하고 다섯 번째 전자가 약하게 묶인다.
- 작은 에너지만으로 이 전자가 conduction band로 올라가 자유전자가 된다.
- donor level은 `E_C` 바로 아래에 존재한다.
- ionized donor는 양전하 `D+`로 남는다.

### p-type doping

- B 같은 Group III 원소는 Si보다 원자가전자가 하나 부족하다.
- 결합 하나에 전자가 비어 hole이 생긴다.
- acceptor level은 `E_V` 바로 위에 존재하며, valence band에서 전자를 받아 hole을 만든다.
- ionized acceptor는 음전하 `A-`로 남는다.

### 전압 공급과 band tilting

- 반도체에 전압을 연결하면 electron의 electrostatic potential energy가 위치에 따라 변한다.
- energy band 전체가 기울어지고, carrier drift가 발생한다.

## 온도 의존성

### Carrier concentration의 세 영역

| 온도 영역 | carrier 농도 지배 요인 | 특징 |
|---|---|---|
| 낮은 온도, freeze-out | donor 또는 acceptor ionization | dopant가 완전히 이온화되지 않음 |
| 중간 온도, extrinsic | dopant concentration | majority carrier 농도 ≈ dopant 농도 |
| 높은 온도, intrinsic | thermal generation across band gap | intrinsic carrier가 dopant carrier보다 많아짐 |

- `T_s`는 donor가 거의 모두 이온화되는 saturation temperature로 볼 수 있다.
- `T_i` 이상에서는 intrinsic carrier generation이 우세하다.

### Mobility의 온도 의존성

- 낮은 온도에서는 ionized impurity scattering이 중요하다.
- 높은 온도에서는 lattice vibration scattering이 커진다.
- doping 농도가 높을수록 impurity scattering이 커져 mobility가 낮아진다.

## 전도도의 온도 의존성

```text
sigma = q (n mu_n + p mu_p)
```

- 온도 상승은 carrier concentration을 증가시킬 수 있지만 mobility를 감소시킬 수도 있다.
- doped semiconductor의 전도도는 carrier 농도 변화와 mobility 변화가 함께 결정한다.
- intrinsic 영역에서는 carrier 농도 증가가 매우 커서 전도도가 급격히 증가한다.

## Degenerate semiconductor

- 매우 높은 doping에서는 donor level들이 서로 겹쳐 band처럼 되고 conduction band와 overlap할 수 있다.
- degenerate n-type에서는 Fermi level이 conduction band 안으로 들어갈 수 있다.
- degenerate p-type에서는 Fermi level이 valence band 안으로 들어갈 수 있다.
- 이 경우 반도체가 금속처럼 높은 carrier concentration과 낮은 저항을 보인다.

## Recombination과 trapping

### Direct recombination

- 전자와 정공이 직접 재결합하며 에너지를 photon 또는 lattice vibration으로 방출한다.
- GaAs처럼 direct band gap 재료에서는 conduction band minimum과 valence band maximum의 `k`가 같아 momentum conservation이 잘 맞는다.
- 그래서 radiative recombination과 LED, laser 응용에 유리하다.

### Trap-assisted recombination

- band gap 내부의 defect level이 전자나 정공을 포획해 재결합을 돕는다.
- dangling bond, impurity, grain boundary는 trap center로 작동할 수 있다.

## Low-level injection과 minority carrier

- n-type 반도체에 약한 빛을 비추면 excess electron `Delta n_n`과 excess hole `Delta p_n`이 생긴다.
- low-level injection에서는 majority carrier 변화가 평형 majority concentration보다 작다.

```text
Delta n_n < n_n0
```

- majority electron 농도는 거의 변하지 않지만, minority hole 농도는 상대적으로 크게 변한다.
- 따라서 광응답과 recombination dynamics는 주로 minority carrier lifetime에 민감하다.

## Photoresponse와 photocurrent

- 조명을 켜면 excess minority carrier concentration이 시간상수 `tau_h`로 steady state까지 증가한다.
- 조명을 끄면 같은 시간상수에 의해 equilibrium value로 지수적으로 감소한다.

```text
Delta p_n(t) rises or decays exponentially with lifetime tau_h
```

- photoconductor나 photodiode의 응답속도는 carrier lifetime과 transport time에 의해 제한된다.
- photocurrent는 광생성 carrier가 전기장에 의해 수집되며 생기는 전류다.

## 시험 포인트

- intrinsic semiconductor에서 electron-hole pair가 생기는 방식.
- donor level과 acceptor level의 band diagram 위치.
- `np = n_i^2`가 열평형에서 의미하는 것.
- freeze-out, extrinsic, intrinsic 온도 영역 구분.
- impurity scattering과 lattice scattering이 mobility에 주는 반대 경향.
- direct recombination과 trap-assisted recombination의 차이.
- low-level injection에서 minority carrier 변화가 중요한 이유.

