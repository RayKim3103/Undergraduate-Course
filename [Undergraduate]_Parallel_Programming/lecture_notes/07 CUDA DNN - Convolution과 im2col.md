---
title: "07 CUDA DNN - Convolution과 im2col"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - dnn
  - convolution
  - im2col
---

# 07 CUDA DNN - Convolution과 im2col

이전: [[06 CUDA Transpose and Bank Conflict - Shared Memory 심화]]  
다음: [[08 CUDA Reduction - Parallel Reduction 최적화]]

## 핵심 요약

이 강의는 DNN의 기본 연산, convolution layer, direct convolution, im2col 기반 convolution을 다룬다. DNN은 fully connected layer와 convolution layer 같은 building block으로 구성되며, GPU 구현에서는 convolution의 7중 loop를 어떻게 dataflow로 배치하고 병렬화할지가 중요하다.

## DNN 기본

Deep neural network는 여러 layer를 쌓아 입력을 점점 더 높은 수준의 feature로 변환한다.

주요 구성:

- Fully connected layer
- Convolution layer
- Activation function, 예: ReLU
- Pooling
- 반복적인 layer stack

Training은 weight를 조정해 network가 표현하는 함수를 바꾸는 과정이며, backpropagation을 통해 filter와 weight를 학습한다.

## 2D Convolution

Convolution은 filter mask를 input image 위에서 sliding하며 output response map을 만든다. 여러 filter를 사용하면 여러 output feature map이 생성된다.

Output 크기:

```text
H_out = H - K + 1
W_out = W - K + 1
```

Padding/stride를 고려하지 않은 기본 valid convolution 기준이다.

## Convolution Layer Forward

대표 loop 차원은 다음 7개다.

| 차원 | 의미 |
|---|---|
| N | minibatch |
| M | output feature maps |
| H | input height |
| W | input width |
| C | input feature maps |
| K1 | weight height |
| K2 | weight width |

7중 loop dataflow:

```text
N -> M -> H -> W -> C -> K1 -> K2
```

CNN 가속은 이 loop nest를 어떤 순서로 돌리고, 어떤 차원을 block/thread/grid로 mapping하며, 어떤 데이터를 재사용할지 결정하는 문제다.

## Direct Convolution 병렬화

강의 전략:

- 각 block이 output pixel tile을 계산한다.
- Grid는 minibatch, output feature map, output tile 차원으로 구성된다.
- Thread block은 `TILE_WIDTH x TILE_WIDTH` output tile을 담당한다.

각 thread는 특정 `(n, m, h, w)` output element를 계산하고, C와 KxK 범위를 순회하며 MAC을 수행한다.

## Shared Memory Convolution

Convolution에서는 인접 output pixel들이 겹치는 input window를 많이 읽는다. 중복 global memory read가 많으므로 shared memory에 input tile을 올려 재사용할 수 있다.

주의점:

- Filter size 때문에 tile 주변 halo 영역이 필요하다.
- Tile load 영역과 실제 compute 영역이 다를 수 있다.
- Boundary check가 필요하다.

## im2col

im2col은 convolution을 matrix multiplication으로 바꾸는 기법이다.

```text
Input image patches -> X_col
Filter weights -> W_row
Y = W_row * X_col
```

장점:

- GPU에서 고도로 최적화된 GEMM을 활용할 수 있다.
- 복잡한 convolution loop를 matrix multiplication 문제로 바꾼다.

단점:

- X_col 생성으로 memory overhead가 생긴다.
- patch가 겹치므로 같은 input 값이 여러 번 복사될 수 있다.

## Pooling과 큰 네트워크

Pooling은 convolution보다 단순하며, 영역 내 max 또는 average를 계산한다. VGGNet 같은 큰 네트워크는 convolution, ReLU, pooling, fully connected layer를 반복적으로 구성한다.

## 추가 convolution 방법

| 방법 | 아이디어 |
|---|---|
| Direct convolution | 원래 loop를 GPU에 mapping |
| im2col + GEMM | convolution을 matrix multiplication으로 변환 |
| FFT convolution | time/spatial domain convolution을 frequency domain multiply로 변환 |
| Winograd | 곱셈 수를 줄이도록 중간값 재배치 |

## 정리

DNN 가속의 핵심은 convolution dataflow를 이해하고, direct convolution과 im2col의 tradeoff를 비교하는 것이다. Direct 방식은 memory overhead가 낮지만 최적화가 어렵고, im2col은 GEMM 최적화를 활용할 수 있지만 추가 memory movement가 생긴다.
