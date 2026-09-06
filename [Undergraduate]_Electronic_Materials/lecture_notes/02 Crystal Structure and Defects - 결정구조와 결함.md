# Crystal Structure and Defects - 결정구조와 결함

tags: #ElectronicMaterials #CrystalStructure #MillerIndex #Defects

이전: [[01 Atomic Model and Bonding - 원자모형과 결합]]  
다음: [[03 Crystalline Amorphous and Fabrication - 결정질 비정질 공정]]

## 핵심 요약

- 결정 구조는 `lattice + basis`로 표현한다.
- 단위격자는 전체 결정의 주기성과 대칭성을 담는 가장 편리한 반복 단위다.
- BCC, FCC, HCP, diamond cubic, zinc blende, NaCl, CsCl 구조는 재료의 충진율, 배위수, 결합 방향성, 전기적 특성을 좌우한다.
- Miller index는 결정 방향 `[uvw]`와 결정면 `(hkl)`을 정수로 표현하는 표기법이다.
- 실제 결정에는 점 결함과 선 결함이 존재하며, 결함은 확산, 전기저항, 기계적 강도, 결정 성장에 큰 영향을 준다.

## 열적으로 활성화된 과정

- 한 안정 상태에서 다른 안정 상태로 이동하려면 중간의 에너지 장벽 `E_A`를 넘어야 한다.
- 원자 확산, 불순물 이동, vacancy 이동은 대표적인 thermally activated process다.
- 온도가 올라갈수록 `E_A`를 넘는 입자 비율이 증가해 확산과 반응 속도가 빨라진다.

```text
probability-like dependence ~ exp(-E_A / kT)
```

- 결정 내부의 interstitial impurity가 한 빈 공간에서 이웃 빈 공간으로 이동하려면 주변 host atom을 밀어낼 충분한 에너지가 필요하다.

## 결정 상태

### Lattice, basis, unit cell

- lattice: 공간에 주기적으로 반복되는 기하학적 점 배열.
- basis: 각 lattice point에 붙는 동일한 원자 또는 분자 그룹.
- crystal structure: lattice에 basis를 배치한 실제 결정 구조.
- unit cell: 반복을 통해 전체 결정을 만들 수 있는 작은 셀.

```text
crystal structure = lattice + basis
```

### 격자 상수

- 일반 단위격자는 변 길이 `a, b, c`와 각도 `alpha, beta, gamma`로 표현한다.
- 결정계가 달라지면 변 길이와 각도 조건이 달라지고, 가능한 대칭성이 달라진다.

## 대표 결정 구조

| 구조 | 특징 | 충진율 또는 구조적 포인트 | 예 |
|---|---|---|---|
| BCC | 모서리 8개와 중심 1개 원자 | 약 68% 충진 | Fe |
| FCC | 모서리와 면 중심에 원자 | 약 74% 충진, close-packed | Cu |
| HCP | 육방 조밀 구조 | 약 74% 충진, FCC와 동일한 조밀 충진율 | Mg 계열 |
| Diamond cubic | 사면체 공유결합 네트워크 | 단위격자 8개 원자 | Si, Ge, diamond |
| Zinc blende | diamond cubic과 유사하되 두 종류 원자 교대 | III-V, II-VI 화합물 반도체 | GaAs, ZnS, InP |
| NaCl | 서로 침투한 두 FCC 이온 격자 | 반대 전하 이온의 이온 결정 | NaCl |
| CsCl | 한 이온이 정육면체 중심, 반대 이온이 모서리 | 배위수 8 | CsCl |

### 배위수

- 배위수는 중심 원자 또는 이온의 최근접 이웃 수다.
- 이온 결정에서는 양이온과 음이온 반지름 비가 가능한 배위수와 결정 구조를 제한한다.
- 반지름 비가 맞지 않으면 특정 위치에 이온이 안정적으로 들어갈 수 없어 다른 결정 구조가 선호된다.

## Miller 지수

### 결정 방향 `[uvw]`

1. 단위격자의 좌표축 x, y, z에 대한 방향 벡터 성분을 잡는다.
2. 분수가 있으면 가장 작은 정수비가 되도록 배수화한다.
3. 음수 성분은 bar 표기로 나타낸다.
4. 결정 방향은 대괄호 `[uvw]`로 쓴다.

### 방향족 `<uvw>`

- 대칭적으로 등가인 방향들의 집합은 `<uvw>`로 표현한다.
- 예를 들어 cubic 구조에서 `<111>`은 여러 등가 대각 방향을 포함한다.

### 결정면 `(hkl)`

1. 결정면이 x, y, z축과 만나는 절편을 구한다.
2. 각 절편의 역수를 취한다.
3. 가장 작은 정수비로 만든다.
4. 결정면은 소괄호 `(hkl)`로 쓴다.

### 면 원자 농도

- planar concentration은 특정 결정면 위에 중심이 놓인 원자 수를 그 면의 면적으로 나눈 값이다.

```text
planar concentration = number of atoms centered on plane / area of plane
```

- 같은 결정이라도 `(100)`, `(110)`, `(111)` 면은 원자 배열 밀도가 다르므로 표면 반응성, 식각 속도, 성장 특성이 달라질 수 있다.

## 동소체와 다형성

- allotropy 또는 polymorphism은 같은 물질이 둘 이상의 결정 구조를 가질 수 있는 성질이다.
- 탄소는 diamond, graphite, fullerene 등 다양한 구조를 갖는다.
- 구조가 달라지면 같은 원소라도 밀도, 탄성률, 전기전도성, 광학 특성이 크게 달라진다.

## 결정 결함

### 결함 분류

- 점 결함: vacancy, substitutional impurity, interstitial impurity, Schottky defect, Frenkel defect.
- 선 결함: edge dislocation, screw dislocation.
- 결함 주변에서는 결합 길이와 각도가 깨져 strain field가 생기며 전자와 phonon 산란이 증가한다.

### Vacancy

- 표면 원자가 충분한 에너지를 얻어 이동하면 빈 격자점이 생긴다.
- vacancy는 원자 확산과 함께 bulk 내부로 이동할 수 있다.
- vacancy 농도는 온도에 민감하고, 금속과 반도체의 확산 공정에서 중요하다.

### 불순물 점 결함

- substitutional impurity: 불순물 원자가 host atom 자리를 대체한다.
- interstitial impurity: 불순물 원자가 host atom 사이 빈 공간에 들어간다.
- 원자 크기 차이가 크면 주변 격자 왜곡이 커지고 산란이 증가한다.

### Schottky defect와 Frenkel defect

- Schottky defect: 이온 결정에서 내부 이온이 표면으로 이동해 vacancy를 남긴다.
- Frenkel defect: host ion이 interstitial 위치로 이동하면서 원래 위치에는 vacancy를 남긴다.
- 이온성 고체의 전기전도와 결함 화학을 이해할 때 중요하다.

### Edge dislocation

- 결정 내부에서 원자면 하나가 중간에 끝나며 생기는 선 결함이다.
- 추가 반평면 주변에는 위쪽 압축, 아래쪽 인장 형태의 strain field가 생긴다.
- 기계적 변형, 열처리, 결정 성장 과정에서 만들어질 수 있다.

### Screw dislocation

- 결정의 한 부분이 다른 부분에 대해 원자 간격만큼 전단되며 생기는 선 결함이다.
- 나선형 계단 구조가 표면에 생겨, 새 원자가 한 개 결합보다 두세 개 결합을 동시에 만들 수 있다.
- 그래서 screw dislocation은 결정 성장의 핵심 사이트가 될 수 있다.

## 시험 포인트

- `lattice`, `basis`, `unit cell`의 차이.
- BCC, FCC, HCP의 충진율과 구조적 차이.
- diamond cubic과 zinc blende가 반도체 재료에서 중요한 이유.
- Miller 방향과 Miller 면을 구하는 절차.
- 점 결함과 선 결함이 전기저항, 확산, 결정성에 주는 영향.
- screw dislocation이 결정 성장을 촉진하는 이유.

