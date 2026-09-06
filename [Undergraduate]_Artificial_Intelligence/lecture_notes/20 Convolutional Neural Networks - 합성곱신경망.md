# Convolutional Neural Networks

tags: #artificial-intelligence #machine-learning #deep-learning #cnn #convolution #computer-vision

관련 노트: [[19 Neural Networks Part 3 - 실전 학습과 정규화]]

## 핵심 요약

이 강의는 computer vision에서 CNN이 왜 효과적인지 설명한다. 시각 피질의 receptive field와 hierarchy에서 출발해, 이미지가 숫자 행렬이라는 관점, hand-engineered feature의 한계, local connectivity, weight sharing, convolution layer의 spatial dimension 계산을 다룬다.

## 역사적 배경

| 흐름 | 의미 |
|---|---|
| Visual cortex 연구 | receptive field와 계층적 시각 처리 아이디어 |
| Neocognitron | simple/complex cell 구조를 닮은 초기 convolutional architecture |
| LeNet | gradient-based learning을 문서 인식에 적용 |
| AlexNet | ImageNet에서 deep CNN의 성능을 크게 보여준 계기 |

CNN은 현대 computer vision의 핵심 구조로 image classification, detection, segmentation 등에서 널리 쓰인다.

## Image는 숫자 행렬

컴퓨터에게 이미지는 pixel intensity matrix이다.

```text
grayscale image: H x W
color image: H x W x 3
pixel value: 보통 0-255
```

이미지 분류는 이 숫자 배열에서 class label을 예측하는 문제이다.

## Shallow Learning의 한계

과거 computer vision에서는 SIFT, HOG, bag of words 같은 hand-engineered feature를 사용했다. 이런 feature는 domain knowledge가 많이 필요하고, 새로운 task나 데이터에 맞게 확장하기 어렵다.

Deep learning은 feature extractor 자체를 데이터에서 학습한다.

```text
raw image -> learned features -> classifier
```

## Fully Connected Layer의 문제

이미지를 1차원 vector로 펴서 fully connected layer에 넣으면 공간 구조가 사라진다.

예:

```text
32 x 32 x 3 image -> 3072 x 1 vector
```

모든 pixel이 모든 neuron에 연결되므로 parameter 수가 많고, 인접 pixel의 지역 구조를 직접 활용하지 못한다.

## Local Connectivity

CNN은 한 neuron이 전체 이미지가 아니라 작은 patch만 본다. 이 patch를 receptive field라고 생각할 수 있다.

```text
local patch -> one neuron output
```

이미지에서는 가까운 pixel들이 의미 있는 local pattern을 이루므로 local connectivity가 자연스럽다.

## Weight Sharing

같은 filter를 이미지 전체 위치에 반복 적용한다. 같은 edge detector나 texture detector가 화면 어느 위치에서나 유용할 수 있기 때문이다.

효과:

- parameter 수 감소
- 위치 이동에 더 강한 feature 학습
- spatial structure 보존

## Convolution Operation

2D convolution은 filter를 이미지 위로 sliding하며 weighted sum을 계산한다.

```text
output[x,y] = sum_i sum_j image[x+i, y+j] * filter[i,j]
```

Convolution은 feature extractor로 작동한다. 여러 filter를 사용하면 서로 다른 activation map을 얻는다.

## Multiple Channels

Color image는 RGB 세 channel을 가진다. Convolution filter의 depth는 input depth와 같아야 한다.

예:

```text
input: 32 x 32 x 3
filter: 5 x 5 x 3
one spatial location output: scalar
```

Filter가 6개이면 output activation map도 6개가 된다.

```text
output depth = number of filters
```

## CNN Layer 구성

CNN은 convolution layer와 activation function을 반복해 feature hierarchy를 만든다.

```text
CONV -> ReLU -> CONV -> ReLU -> FC -> classifier
```

초기 layer는 edge나 color contrast 같은 low-level feature를 잡고, 깊은 layer는 object part나 class-level feature를 학습한다.

## Spatial Dimension 계산

Convolution output size는 input size, filter size, stride, padding에 의해 결정된다.

```text
W_out = (W_in - F + 2P) / S + 1
H_out = (H_in - F + 2P) / S + 1
```

| 기호 | 의미 |
|---|---|
| `F` | filter size |
| `S` | stride |
| `P` | zero padding |

Padding 없이 convolution을 반복하면 spatial size가 빠르게 줄어든다. Zero padding은 border를 보존하고 output size를 조절하는 데 사용한다.

## 예시 계산

입력이 `32 x 32 x 3`, filter가 `5 x 5 x 3`, stride가 1, padding이 2이면 spatial size는 유지된다.

```text
W_out = (32 - 5 + 2*2)/1 + 1 = 32
```

Filter가 10개이면 output은 다음 크기이다.

```text
32 x 32 x 10
```

## 1x1 Convolution

1x1 convolution은 spatial 위치마다 channel 방향의 linear combination을 수행한다. Spatial size는 유지하면서 channel 수를 바꾸거나 feature를 섞는 데 사용된다.

## CONV Layer의 Neuron 관점

Convolution layer도 결국 neuron들의 배열이다. 차이는 각 neuron이 전체 입력이 아니라 local receptive field만 보고, 같은 filter weight를 여러 위치에서 공유한다는 점이다.

## 시험ㆍ복습 체크포인트

- CNN이 fully connected network보다 이미지에 적합한 이유를 설명할 수 있어야 한다.
- Local connectivity와 weight sharing의 의미를 구분해야 한다.
- Convolution output spatial size 공식을 사용할 수 있어야 한다.
- Input channel 수와 filter depth가 같아야 하는 이유를 이해해야 한다.
- 여러 filter가 여러 activation map을 만든다는 점을 설명할 수 있어야 한다.

