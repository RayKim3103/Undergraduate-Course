# Light and Quantum Duality - 빛과 양자 이중성

tags: #ElectronicMaterials #QuantumPhysics #Diffraction #PhotoelectricEffect #Wavefunction

이전: [[03 Crystalline Amorphous and Fabrication - 결정질 비정질 공정]]  
다음: [[05 Tunneling Blackbody Laser - 터널링 흑체 레이저]]

## 핵심 요약

- 빛은 회절, 간섭, 결맞음 같은 파동 성질을 보인다.
- 광전효과와 Compton 산란은 빛을 에너지와 운동량을 가진 photon으로 보아야 설명된다.
- 전자도 Young 이중슬릿 실험에서 간섭 무늬를 만들며 파동성을 보인다.
- 파동함수 `psi(x)`는 전자의 물질파 상태를 나타내며, `|psi(x)|^2`는 전자 발견 확률밀도다.
- 무한 퍼텐셜 우물 안 전자는 경계조건 때문에 에너지가 양자화된다.

## 빛의 회절

### 회절의 의미

- 회절은 파동이 장애물을 만나거나 좁은 틈을 지날 때, 직진 경로 뒤쪽까지 퍼져 나가는 현상이다.
- 슬릿 폭 `a`가 파장 `lambda`에 가까워질수록 회절이 커진다.
- 큰 구멍에서는 직진성이 강하고, 작은 구멍에서는 퍼짐이 뚜렷하다.

### 회절 영역

- Kirchhoff region: 슬릿 근처의 일반 회절 영역.
- Fresnel region: 관측 거리가 슬릿 폭보다 충분히 큰 근거리 회절.
- Fraunhofer region: 관측 거리가 매우 멀어 평면파 근사가 가능한 원거리 회절.

```text
Fraunhofer condition: z >> pi a^2 / lambda
```

### 단일 슬릿 회절

- Fraunhofer 단일 슬릿 회절에서 어두운 무늬 조건은 다음과 같다.

```text
a sin(theta_n) = n lambda
theta_n ≈ n lambda / a  for small angle
```

- 같은 파장에서는 슬릿 폭 `a`가 작을수록 회절각이 커진다.
- 같은 슬릿 폭에서는 파장 `lambda`가 길수록 회절각이 커진다.

## 간섭과 결맞음

### 간섭

- 보강간섭은 두 파동의 위상이 맞아 진폭이 커지는 경우다.
- 상쇄간섭은 두 파동의 위상이 반대로 만나 진폭이 작아지는 경우다.

```text
constructive interference: path difference = n lambda
destructive interference: path difference = (n + 1/2) lambda
```

### 결맞음

- 두 광원이 같은 주파수와 일정한 위상 관계를 유지하면 coherent하다고 한다.
- stationary interference pattern을 만들려면 coherence가 필요하다.
- laser는 대표적인 coherent light source이고, 태양이나 일반 lamp는 incoherent source에 가깝다.

## 빛의 전자기파 관점

- 고전적으로 빛은 시간에 따라 변하는 전기장과 자기장이 서로 수직이고, 진행 방향에도 수직인 횡파다.
- 이 관점은 회절, 간섭, 편광, X-ray diffraction 같은 현상을 잘 설명한다.
- 결정에 X-ray를 조사하면 원자면에서 반사된 파동이 특정 방향에서 보강간섭을 일으켜 diffraction spot이나 ring을 만든다.

## Young 이중슬릿 실험

- 두 슬릿 `S1`, `S2`에서 나온 파동이 스크린의 점 `P`에서 만나 간섭한다.
- 보강간섭: `S1P - S2P = n lambda`.
- 상쇄간섭: `S1P - S2P = (n + 1/2) lambda`.
- 실제 이중슬릿 무늬는 단일 슬릿 회절 envelope 위에 이중슬릿 간섭무늬가 겹친 형태다.

## 광전효과

### 실험 결과

- 같은 파장에서 빛의 세기를 키우면 포화 전류가 커진다.
- 전자의 최대 운동에너지는 빛의 세기가 아니라 주파수에 의해 증가한다.
- 금속마다 threshold frequency가 다르며, 이는 work function이 다르기 때문이다.

```text
KE_max = h nu - h nu_0 = h nu - Phi
```

- `Phi`는 금속에서 전자를 꺼내기 위한 work function이다.
- `nu < nu_0`이면 아무리 세기가 커도 전자가 방출되지 않는다.

### 의미

- 빛 에너지는 연속적으로 전달되는 것이 아니라 photon 단위 `h nu`로 전달된다.
- 광전효과는 빛의 입자성을 보여주는 핵심 실험이다.

## Compton 산란

- X-ray photon이 도체의 거의 자유로운 전자와 충돌하면 산란된 photon의 파장이 길어진다.
- 산란각에 따라 파장 변화가 달라지며, 이는 photon이 운동량을 가진 입자처럼 행동함을 뜻한다.
- 광전효과가 에너지 양자화를 보인다면, Compton 산란은 photon momentum의 실재성을 보인다.

## 전자의 파동성

- 전자를 가속해 이중슬릿을 통과시키면 스크린에 간섭무늬가 나타난다.
- 전자는 입자처럼 한 점에서 검출되지만, 많은 전자를 누적하면 파동 간섭 분포를 만든다.
- 이 결과는 전자 상태를 궤적이 아니라 파동함수로 다루어야 함을 보여준다.

## 파동함수와 무한 퍼텐셜 우물

### 파동함수 조건

- 물리적으로 허용되는 `psi(x)`는 유한하고, 단일값이며, 정규화 가능해야 한다.
- 경계에서 불연속적이거나 무한대로 발산하는 함수는 허용되지 않는다.

### 확률 해석

```text
probability density = |psi(x)|^2
```

- 특정 위치에서 전자를 정확히 예측하기보다, 공간별 발견 확률을 계산한다.

### 무한 퍼텐셜 우물

- 1차원 무한 퍼텐셜 우물에서는 벽 밖으로 전자가 나갈 수 없으므로 경계에서 `psi = 0`이어야 한다.
- 이 경계조건 때문에 가능한 파장만 남고 에너지가 불연속적으로 양자화된다.
- 낮은 양자수 상태일수록 node가 적고 에너지가 낮다.

## 시험 포인트

- 회절각이 `lambda/a`에 비례한다는 해석.
- 보강간섭과 상쇄간섭의 path difference 조건.
- 광전효과에서 intensity와 frequency가 각각 saturation current와 electron kinetic energy에 주는 영향.
- work function과 stopping voltage의 의미.
- 전자 이중슬릿 실험이 전자의 파동성을 보여주는 방식.
- `|psi|^2` 확률 해석과 무한 우물의 에너지 양자화.

