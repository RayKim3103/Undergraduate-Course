---
title: "11. Deep Learning Hardware and Quantization"
pages: 78
tags: [intelligent-system, lecture-note, AI-accelerator, CNN, dataflow, quantization]
---

# 11. Deep Learning Hardware and Quantization

> 이전: [[10 Computer Vision Neural Networks and CNN]]

## 학습 목표

Week11 자료는 딥러닝 연산을 CPU/GPU/전용 하드웨어에서 어떻게 실행하는지, CNN convolution을 행렬곱으로 바꾸는 방법, accelerator dataflow, low-precision quantization을 다룬다.

## CPU와 GPU의 차이

### CPU

- general-purpose processor
- Von Neumann architecture 기반
- 낮은 latency와 복잡한 control flow에 최적화
- multi-level cache로 single thread latency 감소
- out-of-order/speculative execution 등 control logic 비중이 큼

### GPU

- data-parallel throughput computation에 최적화
- 수많은 ALU를 통해 SIMD/SIMT 연산 수행
- 많은 thread context와 register를 보유
- memory latency를 많은 thread와 computation으로 숨김
- transistor를 computation에 더 많이 사용

요약:

| 관점 | CPU | GPU |
|---|---|---|
| 목표 | low latency | high throughput |
| 강점 | 복잡한 control, single-thread 성능 | 대규모 병렬 연산 |
| memory | cache hierarchy 중심 | shared memory/register 중심 |
| DL 연산 | 가능하지만 throughput 제한 | GEMM/conv에 적합 |

## GEMM과 Tiling

딥러닝의 FC layer와 convolution은 대부분 matrix multiplication(GEMM)으로 실행할 수 있다.

batch size가 $N$이면 matrix-vector operation이 matrix-matrix multiplication으로 바뀐다.

naive implementation은 temporal locality가 낮아 memory access가 비효율적이다. 이를 개선하기 위해 tiling을 사용한다.

tiling:

- 행렬을 작은 block으로 나눈다.
- tile이 cache/shared memory에 들어가도록 크기를 정한다.
- 같은 tile data를 여러 MAC 연산에 재사용한다.

## Fully-Connected Layer

FC layer:

$$
\mathbf{y}=W\mathbf{x}+\mathbf{b}
$$

batch 처리:

$$
Y=WX+B
$$

즉 행렬곱으로 표현된다. CPU/GPU library는 matrix shape에 따라 최적 GEMM kernel을 선택한다.

## Convolution을 Matrix Multiplication으로 변환

convolution layer도 Toeplitz matrix 또는 im2col 변환을 통해 GEMM으로 바꿀 수 있다.

아이디어:

1. input feature map의 local patch를 펼쳐 행렬 row/column으로 만든다.
2. filter weights를 펼쳐 행렬로 만든다.
3. GEMM을 수행한다.
4. 결과를 output feature map shape로 다시 reshape한다.

장점:

- 기존 GEMM accelerator/library 재사용 가능

단점:

- input data가 반복 저장되어 memory footprint가 커진다.

## Multi-Channel Convolution

CNN convolution의 차원:

- input feature map: $C \times H \times W$
- filter: $M \times C \times R \times S$
- output feature map: $M \times E \times F$

각 output element는 다음 합으로 계산된다.

$$
O[m,e,f]=
\sum_{c=0}^{C-1}\sum_{r=0}^{R-1}\sum_{s=0}^{S-1}
I[c,e+r,f+s]W[m,c,r,s]
$$

batch까지 포함하면 $N$ 차원이 추가된다.

## MAC Architecture

딥러닝 accelerator의 기본 연산은 multiply-and-accumulate이다.

$$
psum \leftarrow psum + activation \times weight
$$

convolution과 FC 모두 MAC 연산의 반복이다.

하드웨어 설계의 핵심:

- MAC unit 수
- PE(processing element) array 구조
- activation/weight/psum data movement
- local buffer와 global buffer 크기
- DRAM access 최소화

## Specialized Hardware와 Memory Bottleneck

CNN 연산은 병렬성이 매우 높지만 memory access가 병목이다.

자료 예시:

- AlexNet은 수억 MAC을 수행
- worst case로 모든 read/write가 DRAM이면 memory access energy가 커짐

핵심 목표:

```text
비싼 DRAM 접근을 줄이고, 가까운 local memory에서 data reuse를 늘린다.
```

memory hierarchy에서 멀고 큰 memory일수록 접근 에너지가 크다. 따라서 activation, weight, partial sum을 가능한 한 PE 근처에서 재사용해야 한다.

## Data Reuse

CNN에는 큰 data reuse 기회가 있다.

- 같은 weight가 여러 spatial location에 재사용된다.
- 같은 activation이 여러 filter/window에 재사용된다.
- partial sum은 여러 MAC 결과가 누적될 때 계속 갱신된다.

accelerator dataflow는 어떤 data를 stationary하게 둘 것인지에 따라 구분된다.

## Output Stationary

partial sum을 PE 내부 또는 local memory에 오래 유지한다.

목표:

- psum read/write energy 최소화
- local accumulation 최대화
- weights와 activations는 PE array에 broadcast/multicast

장점:

- partial sum을 DRAM이나 global buffer에 자주 쓰지 않아도 된다.

주의:

- output tile 크기와 local accumulator 용량이 중요하다.

## Weight Stationary

weight를 PE 내부에 고정하고 여러 activation에 재사용한다.

목표:

- weight read energy 최소화
- convolution/filter reuse 최대화
- activation을 broadcast하고 psum을 공간적으로 accumulate

적합한 상황:

- weight reuse가 큰 layer
- weight fetch 비용이 큰 경우

## Input Stationary

activation/input feature map을 PE 내부 또는 local buffer에 유지한다.

목표:

- activation read energy 최소화
- input feature map reuse 최대화
- weight를 공급하며 psum을 accumulate

## No Local Reuse

큰 global buffer를 shared storage로 사용한다.

특징:

- local reuse는 적지만 DRAM access는 줄일 수 있다.
- activation multicast, weight single-cast, psum accumulation 구조를 사용할 수 있다.

## Quantization이란

quantization은 continuous 또는 큰 범위의 값을 discrete set으로 제한하는 과정이다.

딥러닝 전통 구현은 FP32/FP64를 많이 쓰지만, 실제 inference에서는 8-bit 또는 16-bit precision으로도 충분한 경우가 많다.

## Low-Precision Computation의 장점

- memory에 더 많은 data 저장 가능
- cache/local buffer에 더 큰 model 저장 가능
- 초당 더 많은 number 전송 가능
- SIMD 병렬성 증가
- 계산 속도 향상
- energy 감소

단점:

- 표현 가능한 값 범위 제한
- full precision 값을 낮은 precision으로 저장할 때 quantization error 발생

## FP16

half-precision floating point는 보통 16 bit로 표현된다.

| field | bit |
|---|---:|
| sign | 1 |
| exponent | 5 |
| mantissa | 10 |

floating point는 넓은 dynamic range를 표현하기 좋지만 fixed point보다 hardware cost가 클 수 있다.

## Fixed Point

$(p+q+1)$ bit fixed-point:

| field | bit |
|---|---:|
| sign | 1 |
| integer part | p |
| fractional part | q |

예: -1과 1 사이 값을 표현하려면 integer bit 없이 fractional bit 중심으로 설계할 수 있다.

fixed point는 FPGA/ASIC에서 arbitrary bit-width를 정하기 쉽고 hardware cost가 낮다.

## Symmetric Scale Quantization

0 중심 real range를 정수 range에 대응시킨다.

개념:

$$
x_q = \operatorname{round}(s \cdot \operatorname{clip}(x,-\alpha,\alpha))
$$

- $s$: scaling factor
- $\alpha$: clipping threshold
- quantized range가 0을 중심으로 대칭

장점:

- zero-point 처리가 단순
- MAC hardware가 단순해질 수 있음

## Asymmetric Scale + Shift Quantization

0 중심이 아닌 real range를 더 효율적으로 표현한다.

개념:

$$
x_q = \operatorname{round}(s \cdot \operatorname{clip}(x,\beta,\alpha)+z)
$$

- $z$: zero point 또는 shift
- $\beta,\alpha$: clipping threshold

분포가 0 중심이 아닐 때 bit range를 더 효율적으로 사용할 수 있다.

## 어떤 연산을 Quantize할 것인가

quantize 대상:

- matrix multiply
- fully-connected layer
- convolution
- ReLU
- pooling 등 quantized space에서 처리 가능한 연산

quantize하지 않는 것이 보통 나은 대상:

- softmax
- tanh
- sigmoid
- GeLU
- 계산량이 작고 nonlinear한 layer

## Assignment 관점의 설계 포인트

CNN accelerator를 만들 때 확인해야 할 것:

- convolution loop order
- input/weight/output buffer 구조
- channel과 tile dimension
- MAC 병렬도
- psum accumulation 위치
- DRAM/BRAM access 횟수
- quantized data bitwidth
- overflow와 scaling
- control/status register로 configurable parameter 전달

## 체크포인트

- 딥러닝 layer는 GEMM/MAC 연산으로 볼 수 있다.
- convolution을 Toeplitz/im2col로 바꾸면 GEMM화할 수 있지만 data duplication이 생긴다.
- AI accelerator의 진짜 병목은 연산보다 data movement인 경우가 많다.
- output/weight/input stationary는 무엇을 PE 근처에 오래 둘지의 선택이다.
- quantization은 성능/전력/메모리 효율을 올리지만 accuracy와 overflow를 함께 관리해야 한다.
