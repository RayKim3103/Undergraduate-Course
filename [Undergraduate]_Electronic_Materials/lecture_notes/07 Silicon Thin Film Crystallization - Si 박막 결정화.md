# Silicon Thin Film Crystallization - Si 박막 결정화

tags: #ElectronicMaterials #SiliconThinFilm #LTPS #ELA #TFT

이전: [[06 Electrical and Thermal Conduction - 전기 열 전도]]  
다음: [[08 Molecular Orbital and Energy Bands - 분자궤도 에너지밴드]]

## 핵심 요약

- Si 박막 결정화는 비정질 Si를 poly-Si로 바꾸어 TFT 이동도와 전류 구동 능력을 높이는 공정이다.
- 대표 방법에는 furnace anneal, SPC, RTA, CGS, MILC, ELA, SLS가 있다.
- 생산 공정에서는 특히 ELA와 SLS가 중요하게 다루어진다.
- ELA는 excimer laser로 표면 a-Si만 빠르게 녹이고 재결정화해 glass substrate 손상을 줄인다.
- grain boundary는 TFT의 threshold voltage 상승, off current 증가, mobility 저하를 일으킨다.
- laser energy density는 partial melting, complete melting, super lateral growth를 결정한다.

## Si 박막 결정화 방법

| 방법 | 의미 | 핵심 특징 |
|---|---|---|
| Furnace anneal | 노 전체 가열 | 긴 시간, 높은 thermal budget |
| SPC | solid phase crystallization | 고상 상태에서 결정화 |
| RTA | rapid thermal annealing | 짧은 시간 고온 열처리 |
| CGS | continuous grain Si | 연속 grain 형성 |
| MILC | metal induced lateral crystallization | 금속 촉매로 lateral crystallization 유도 |
| ELA | excimer laser annealing | 표면만 순간 용융 후 재결정화 |
| SLS | sequential lateral solidification | lateral growth를 순차적으로 제어 |

## Rapid Thermal Annealing

- RTA는 dopant activation과 metal contact의 interfacial reaction에 쓰이는 반도체 공정이다.
- wafer를 상온에서 약 1000-1500 K까지 빠르게 가열한다.
- 목표 온도에서 몇 초만 유지한 뒤 빠르게 냉각한다.
- 긴 furnace anneal보다 diffusion broadening을 줄이면서 필요한 열처리 효과를 얻을 수 있다.

## Excimer Laser Annealing

### 공정 단계

1. glass substrate 위에 amorphous silicon layer를 증착한다.
2. pulsed rectangular UV laser beam을 스캔한다.
3. a-Si 표면층이 빠르게 녹는다.
4. 냉각 중 재결정화되어 poly-Si가 된다.
5. laser 에너지는 주로 표면 a-Si에서 흡수되므로 glass substrate는 상대적으로 영향을 덜 받는다.

### ELA의 장점과 문제

- 장점: 결정화 시간이 매우 짧고, 저온 기판에서도 poly-Si 형성이 가능하다.
- 장점: 여러 shot과 overlap 조건을 이용해 grain size를 키울 수 있다.
- 문제: OLED 또는 large-area display에서 균일도 문제가 생기기 쉽다.
- 예시 조건으로 95% overlap, 20 shots on a-Si 같은 다중 조사 조건이 다뤄진다.

## Polycrystalline Silicon TFT

### Poly-Si의 의미

- poly-Si는 여러 결정 grain이 모인 Si 박막이다.
- grain 내부는 결정질에 가깝지만, grain boundary에서는 결합 불완전성과 defect state가 많다.
- a-Si보다 carrier conduction이 좋아 TFT 성능을 높일 수 있지만, grain boundary 품질이 성능을 제한한다.

### Grain boundary 효과

- threshold voltage `V_th`가 증가할 수 있다.
- off current `I_off`가 증가할 수 있다.
- carrier mobility가 감소한다.
- trap state가 많아 transfer curve의 subthreshold 특성이 나빠질 수 있다.

### TFT transfer curve

- transfer curve는 drain voltage 조건을 고정하고 gate voltage를 변화시키며 drain current를 측정한 곡선이다.
- n-type TFT와 p-type TFT는 carrier 종류와 mobility 차이 때문에 전류 수준과 기울기가 다르다.

## ELA에서 energy density의 역할

### 분석 방법

- TEM은 ex-situ로 grain 구조를 관찰한다.
- TR, transient reflectance는 in-situ로 laser 조사 중 melting과 solidification dynamics를 본다.
- energy density에 따른 average grain radius와 melt duration을 비교하면 결정화 regime을 판단할 수 있다.

### 세 가지 결정화 regime

| Energy density | 상태 | 결과 grain |
|---|---|---|
| 낮음 | partial melting | vertical regrowth, small grains |
| 너무 높음 | complete melting | copious nucleation, fine grains |
| 적절함 | near complete melting | super lateral growth, large grains |

## Super Lateral Growth

- SLG는 거의 완전히 녹은 영역에서 남은 seed를 중심으로 lateral growth가 길게 진행되는 regime이다.
- seed가 가까우면 lateral growth가 서로 만나 continuous large-grained poly-Si를 만든다.
- seed가 너무 멀면 완전 용융 영역에서 copious nucleation이 먼저 발생해 isolated disk 또는 fine grain이 생길 수 있다.
- 따라서 energy density, pulse overlap, seed spacing 제어가 grain size와 균일도에 직접 연결된다.

## Multiple Pulse Irradiation

- 여러 번 laser pulse를 조사하면 melt-mediated grain growth가 반복되어 grain enlargement가 일어날 수 있다.
- 하지만 pulse마다 국부 용융과 재고화가 반복되므로 공정 윈도우가 좁고, 균일도 확보가 중요하다.

## Phase Transformation

### Heating

- laser pulse가 들어오면 a-Si의 온도가 급격히 상승한다.
- energy density가 충분하면 표면층이 녹고, 용융 깊이는 laser fluence와 흡수율에 따라 달라진다.

### Cooling

- pulse 이후 열이 기판과 주변으로 빠져나가며 재고화가 진행된다.
- 냉각 속도와 seed 존재 여부가 nucleation과 growth의 경쟁을 결정한다.
- recalescence는 결정화 중 방출되는 잠열 때문에 온도 변화가 일시적으로 완만해지는 현상으로 이해할 수 있다.

## Grain Boundary 위치 제어

- TFT channel 내부에 grain boundary가 놓이면 carrier 이동 경로가 trap과 barrier를 만나 성능 편차가 커진다.
- SLS나 artificially controlled SLG는 grain boundary 위치를 channel 밖으로 유도해 device uniformity를 높이려는 접근이다.
- source, drain, gate 배치와 grain boundary 위치의 관계가 TFT 특성 최적화에서 중요하다.

## 시험 포인트

- RTA와 ELA의 공정 차이.
- ELA가 glass substrate를 크게 손상시키지 않는 이유.
- grain boundary가 TFT의 `V_th`, `I_off`, mobility에 주는 영향.
- partial melting, complete melting, SLG regime의 차이.
- TR analysis로 melt duration과 energy density 관계를 보는 이유.
- grain boundary 위치 제어가 TFT 균일도에 중요한 이유.

