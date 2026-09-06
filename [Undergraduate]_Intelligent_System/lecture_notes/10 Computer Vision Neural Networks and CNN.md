---
title: "10. Computer Vision, Neural Networks, and CNN"
pages: 53
tags: [intelligent-system, lecture-note, computer-vision, neural-network, CNN]
---

# 10. Computer Vision, Neural Networks, and CNN

> 이전: [[09 PS PL AXI PYNQ and ILA]]
> 다음: [[11 Deep Learning Hardware and Quantization]]

## 학습 목표

Week10-1 자료는 이미지 필터링에서 시작해 AI, neural network, MLP, CNN의 기본 구조까지 설명한다.

## Image란 무엇인가

2D digital image는 imaging system을 통해 얻은 공간 sampling과 quantization의 결과이다.

- grayscale image: 각 pixel이 intensity 하나를 가짐
- RGB image: 각 pixel이 `[R,G,B]` 세 값을 가짐
- 일반적으로 intensity는 1 byte, 즉 0-255 범위로 표현 가능

수학적으로 이미지는 위치 $(x,y)$에서 intensity를 반환하는 함수로 볼 수 있다.

$$
I(x,y)
$$

## Image Filtering

이미지에 operator를 적용해 성질을 바꾸거나 정보를 추출한다.

예:

- smoothing
- sharpening
- deblurring
- edge/corner extraction

대부분의 linear filtering은 작은 matrix인 mask/kernel/filter를 사용한다.

## Linear Filtering과 Convolution

kernel을 image 위로 이동시키며 주변 pixel과 가중합을 계산한다.

딥러닝에서는 엄밀한 convolution보다 cross-correlation을 사용하는 경우가 많다.

cross-correlation:

$$
Y[i,j]=\sum_m\sum_n X[i+m,j+n]K[m,n]
$$

convolution은 kernel을 뒤집는 차이가 있지만, CNN에서는 kernel weight가 학습되므로 보통 cross-correlation을 convolution layer라고 부른다.

## 대표 필터

| 필터 | 효과 |
|---|---|
| identity | 원본 유지 |
| shift kernel | 이미지를 한 방향으로 이동 |
| mean/blur | smoothing, low-pass |
| sharpening | high-frequency 강조 |
| Gaussian | 부드러운 low-pass filtering |

Gaussian filter:

$$
G_\sigma(x,y)=\frac{1}{2\pi\sigma^2}
e^{-\frac{x^2+y^2}{2\sigma^2}}
$$

$\sigma$가 클수록 더 넓게 blur된다.

## Edge와 Gradient

edge는 image intensity가 급격히 변하는 위치이다.

digital image의 미분은 finite difference로 근사한다.

$$
\frac{\partial I}{\partial x}
\approx I[x+1,y]-I[x,y]
$$

gradient:

$$
\nabla I =
\left[
\frac{\partial I}{\partial x},
\frac{\partial I}{\partial y}
\right]
$$

edge strength:

$$
\|\nabla I\|=
\sqrt{
\left(\frac{\partial I}{\partial x}\right)^2+
\left(\frac{\partial I}{\partial y}\right)^2}
$$

gradient direction:

$$
\theta=\tan^{-1}
\left(
\frac{\partial I/\partial y}{\partial I/\partial x}
\right)
$$

## AI, ML, DL

| 용어 | 의미 |
|---|---|
| AI | 기계가 인간의 지능적 행동을 수행 |
| ML | data에서 pattern을 학습해 예측 |
| DL | neural network 기반 ML |

예:

- AI: knowledge representation, fuzzy logic
- ML: decision tree, clustering
- DL: artificial/deep neural networks

## Artificial Neuron

perceptron은 neural network의 기본 연산 단위이다.

구성:

1. input
2. weight
3. weighted sum
4. bias
5. activation function

수식:

$$
u=\sum_i w_i x_i
$$

$$
y=\phi(u+b)
$$

bias는 affine transformation을 가능하게 한다.

## Single Layer Perceptron의 한계

single layer perceptron은 linearly separable function만 표현할 수 있다.

가능:

- AND
- OR

불가능:

- XOR

XOR은 하나의 직선 decision boundary로 분리되지 않는다.

## Multi Layer Perceptron

hidden layer를 추가하면 입력 공간을 변환해 XOR 같은 nonlinear 문제를 풀 수 있다.

구조:

```text
input -> hidden layer -> output
```

hidden layer는 feature extractor처럼 동작한다.

## Deep Neural Network

deep learning은 여러 hidden layer로 구성된 neural network를 사용한다.

전통적 접근:

```text
Pattern -> hand-crafted feature extractor -> classifier -> label
```

deep learning:

```text
Raw pattern -> learned feature extractor + classifier -> label
```

초기 layer는 edge/corner 같은 low-level feature를, 깊은 layer는 object part나 semantic feature 같은 high-level feature를 학습한다.

## MLP for Images

이미지를 MLP에 넣으려면 2D matrix를 1D vector로 flatten한다.

예: MNIST

- handwritten digit 0-9
- 28 x 28 image
- pixel value 0-255
- train 55,000, test 10,000 examples

전처리:

- 0-255 integer를 0-1 floating point로 normalization
- normalization은 data variance를 줄여 학습 수렴을 빠르게 한다.

## MLP의 이미지 처리 한계

1. flatten 과정에서 spatial relationship이 손실된다.
2. high-resolution image에서는 fully-connected weight 수가 폭증한다.
3. 같은 pattern이 위치만 바뀌어도 별도 weight로 학습해야 한다.

예:

- MNIST 28x28: 784 input neurons
- Full HD 1920x1080: 2,073,600 input neurons

## CNN으로의 전환

CNN은 locally-connected layer와 weight sharing을 사용한다.

Fully-connected:

- hidden unit이 전체 image를 봄
- parameter 수 많음

Convolutional layer:

- 작은 local receptive field만 봄
- 동일 kernel을 모든 위치에 공유
- 공간 구조 보존

## CNN 구성 요소

일반적인 CNN:

```text
Convolution -> Activation -> Pooling -> ... -> Fully Connected -> Output
```

### Convolution Layer

- learnable kernel을 image 또는 feature map에 적용
- local pattern을 검출
- multiple input/output channels 처리 가능

### Activation

비선형성을 부여한다. ReLU 같은 modern activation이 널리 쓰인다.

### Pooling

feature map의 spatial dimension을 줄인다.

- max pooling: window 안 최댓값
- average pooling: window 평균

pooling은 위치 변화에 대한 민감도를 줄인다.

## Translation Equivariance와 Invariance

### Equivariance

입력이 shift되면 출력 feature도 같은 방식으로 shift된다.

주로 convolution layer가 제공한다.

### Invariance

입력 object의 위치가 달라도 같은 class로 인식한다.

pooling과 deeper layers가 invariance 형성에 도움을 준다.

## Padding과 Stride

### Padding

입력 주변에 0 등을 추가해 boundary를 확장한다.

효과:

- output size 조절
- edge 주변 정보 손실 완화

### Stride

kernel 이동 간격이다.

효과:

- stride가 커지면 output spatial size 감소
- 계산량 감소
- down-sampling 효과

## Multi-Channel Convolution

input channel이 $C$개이면 kernel도 $C$개 channel을 가진다.

하나의 output channel:

$$
Y_o[i,j]=
\sum_{c=0}^{C-1}
\sum_m\sum_n
X_c[i+m,j+n]K_{o,c}[m,n]
$$

output channel이 여러 개이면 서로 다른 filter set이 여러 feature map을 만든다.

## 체크포인트

- image filtering의 핵심 연산은 convolution/cross-correlation이다.
- edge는 intensity 변화가 큰 지점이며 gradient로 측정한다.
- MLP는 이미지를 flatten하므로 공간 구조와 parameter 효율성이 약하다.
- CNN은 local connection과 weight sharing으로 이미지에 적합하다.
- padding, stride, channel 수는 CNN output shape와 hardware cost를 결정한다.
