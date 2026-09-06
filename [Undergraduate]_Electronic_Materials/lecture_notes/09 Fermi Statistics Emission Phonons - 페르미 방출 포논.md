# Fermi Statistics Emission Phonons - 페르미 방출 포논

tags: #ElectronicMaterials #FermiDirac #Thermoelectric #ThermionicEmission #FieldEmission #Phonon

이전: [[08 Molecular Orbital and Energy Bands - 분자궤도 에너지밴드]]  
다음: [[10 Intrinsic Extrinsic Semiconductors - 반도체 캐리어]]

## 핵심 요약

- Fermi-Dirac 통계는 Pauli 배타 원리를 따르는 전자의 에너지 점유 확률을 설명한다.
- 전자 농도는 density of states `g(E)`와 점유 확률 `f(E)`의 곱을 에너지에 대해 적분해 얻는다.
- 서로 다른 금속을 접촉시키면 Fermi level이 정렬될 때까지 전자가 이동하고 contact potential이 생긴다.
- Seebeck effect는 온도구배가 전위차를 만드는 thermoelectric 현상이다.
- thermionic emission은 열에너지가 work function을 넘는 전자 방출이고, field emission은 강한 전기장이 장벽을 얇게 만들어 일으키는 터널링 방출이다.
- phonon은 격자진동의 양자화된 에너지이며, 비금속 열전도의 주요 carrier다.

## Fermi-Dirac statistics

- 고체 내 전자는 서로 구별되지 않는 fermion이며 Pauli 배타 원리를 따른다.
- 한 quantum state에는 스핀까지 포함해 허용된 수만큼만 전자가 들어갈 수 있다.
- Fermi-Dirac 함수 `f(E)`는 에너지 `E` 상태가 전자로 점유될 확률이다.

```text
f(E) = 1 / [1 + exp((E - E_F) / kT)]
```

- 0 K에서는 `E < E_F` 상태가 채워지고 `E > E_F` 상태가 비어 있다.
- 온도가 올라가면 `E_F` 근처의 일부 전자가 더 높은 에너지 상태로 열적으로 들뜬다.

## Free electron model

- density of states `g(E)`는 단위 부피, 단위 에너지당 가능한 전자 상태 수다.
- 실제 전자 에너지 분포는 가능한 상태 수와 점유 확률의 곱으로 표현한다.

```text
n_E(E) = g(E) f(E)
n = integral n_E(E) dE
```

- `g(E)`가 커도 `f(E)`가 작으면 실제 전자 수는 작다.
- 반대로 `f(E)`가 1에 가까워도 가능한 상태가 없으면 전자는 존재할 수 없다.
- 반도체 conduction band의 전자 농도도 같은 논리로 계산한다.

## Metal-metal contact potential

- 서로 다른 금속을 접촉시키면 work function과 Fermi level 차이 때문에 전자가 한쪽 표면으로 이동한다.
- 전자 이동은 전하 분리를 만들고 contact potential `Delta V`를 형성한다.
- 평형에서는 두 금속의 Fermi level이 정렬된다.
- 두 금속으로 닫힌 회로를 만들면 두 접촉부의 contact potential이 서로 반대 방향으로 작용해 순전류가 흐르지 않는다.

## Seebeck effect와 thermocouple

### Seebeck effect

- 도체에 온도구배가 있으면 뜨거운 쪽의 전자가 더 높은 평균 에너지와 더 긴 평균 자유행로를 가져 차가운 쪽으로 확산한다.
- 이 전하 재분포가 전기장을 만들고 전위차가 생긴다.
- 전위차의 크기는 재료의 Seebeck coefficient와 온도차에 의해 결정된다.

```text
Delta V = S Delta T
```

### Thermocouple

- 같은 금속선만 사용해 Seebeck voltage를 측정하면 양쪽 접촉 효과가 상쇄되어 net emf가 0이 될 수 있다.
- 서로 다른 두 금속 A, B를 접합하면 두 재료의 Seebeck coefficient 차이 때문에 온도를 전압으로 측정할 수 있다.
- 한 접점은 reference temperature, 다른 접점은 측정 온도에 둔다.

## Thermionic emission

- 금속 내부 전자가 열적으로 높은 에너지를 얻어 `E_F + Phi` 이상이 되면 표면 장벽을 넘어 진공으로 방출될 수 있다.
- 온도가 증가하면 Fermi-Dirac tail이 높은 에너지 쪽으로 확장되어 방출 전자 수가 증가한다.
- vacuum tube와 CRT의 cathode emission은 thermionic emission의 대표 예다.

```text
emission requires electron energy > E_F + Phi
```

## Field emission

- 강한 전기장을 금속 표면에 걸면 표면 포텐셜 장벽이 얇아지고 삼각형에 가까운 장벽이 된다.
- 전자는 Fermi energy 근처에서도 얇아진 장벽을 터널링해 방출될 수 있다.
- 날카로운 tip에서는 전기장이 집중되므로 field emission이 쉽게 발생한다.

### Fowler-Nordheim emission

- Fowler-Nordheim field emission은 높은 전기장에서 전자 터널링 방출을 설명한다.
- Spindt tip cathode와 field emission display는 이 원리를 이용한다.
- Fowler-Nordheim plot은 방출 전류가 field emission 메커니즘을 따르는지 확인하는 데 쓰인다.

### CNT emitter

- carbon nanotube는 길고 매우 얇으며 끝이 뾰족해 국부 전기장 강화가 크다.
- 거의 이상적인 field emitter 형태를 갖기 때문에 FED, 전자원, 센서 응용에서 중요하다.

## Phonon

### 격자진동의 양자화

- 원자를 평형 위치 주변의 harmonic oscillator로 보면 가능한 진동 에너지가 양자화된다.
- 결정의 많은 원자가 결합되어 움직이면 lattice wave가 생긴다.
- 이 격자진동의 양자화된 입자적 표현이 phonon이다.

### Longitudinal wave와 transverse wave

- longitudinal wave: 원자 변위가 파동 진행 방향과 평행하다.
- transverse wave: 원자 변위가 파동 진행 방향과 수직이다.

## Phonon density of states와 heat capacity

- phonon DOS는 주파수별 phonon mode 수를 나타낸다.
- Debye approximation은 실제 phonon DOS를 단순화해 전체 mode 수가 맞도록 근사한다.
- Debye temperature `T_D`는 고체의 phonon spectrum과 heat capacity 거동을 특징짓는다.
- Si의 경우 `T_D = 625 K`로 제시되며, 300 K에서 `T/T_D = 0.48`이므로 molar heat capacity가 고전 한계 `3R`에 완전히 도달하지 않는다.

## 비금속 열전도와 phonon scattering

- 비금속에서는 자유전자가 부족하므로 phonon이 열을 운반한다.
- 뜨거운 영역에서 생성된 phonon은 차가운 영역으로 이동하며 열 에너지를 전달한다.
- phonon-phonon anharmonic interaction은 열 흐름 반대 방향의 phonon을 만들어 열전도를 제한할 수 있다.
- 결함, grain boundary, isotope disorder도 phonon을 산란시켜 열전도도를 낮춘다.

## 시험 포인트

- `g(E)`, `f(E)`, `g(E)f(E)`의 차이.
- 금속 접촉에서 Fermi level 정렬과 contact potential의 관계.
- Seebeck effect와 thermocouple의 작동 원리.
- thermionic emission과 field emission의 장벽 통과 방식 차이.
- sharp tip과 CNT가 field emission에 유리한 이유.
- phonon이 비금속 열전도의 carrier인 이유.
- Debye temperature와 heat capacity 곡선의 의미.

