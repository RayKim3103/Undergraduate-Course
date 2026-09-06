# Neural Networks Part 1: 다층신경망 기초

tags: #artificial-intelligence #machine-learning #neural-network #perceptron #activation #backpropagation

관련 노트: [[16 Dimension Reduction - PCA와 LDA]], [[18 Neural Networks Part 2 - Backpropagation]]

## 핵심 요약

이 강의는 linear classifier의 한계를 출발점으로 neural network를 도입한다. Perceptron과 sigmoid neuron, hidden layer, activation function, XOR 문제, forward propagation과 backpropagation의 기본 아이디어를 다룬다.

## Linear Classifier의 한계

Linear classifier는 feature 공간에서 직선 또는 hyperplane decision boundary를 만든다.

```text
score(x) = w^T x
y_hat = sign(score(x))
```

데이터가 선형으로 분리되지 않으면 단순 linear classifier만으로는 좋은 boundary를 만들 수 없다. 원형이나 XOR 형태의 결정경계가 대표적인 예이다.

## Polynomial Feature와 Feature Transform

비선형 boundary를 만들기 위해 입력을 변환할 수 있다.

```text
model(x,w) = w0 + w1*x1^2 + w2*x2^2
```

이렇게 feature transform을 사람이 설계하면 linear model도 원형 boundary를 만들 수 있다. Neural network는 이러한 nonlinear feature를 데이터에서 학습하도록 확장한 모델로 볼 수 있다.

## Perceptron

Perceptron은 하나의 neuron으로 볼 수 있다.

```text
z = w0 + sum_j w_j*x[j]
output = g(z)
```

입력 edge는 feature 값을 전달하고, weight는 각 feature의 중요도를 나타낸다. `x[0]=1`로 두면 intercept를 weight로 포함할 수 있다.

## Sigmoid Neuron

Sigmoid neuron은 activation function으로 sigmoid를 사용한다.

```text
g(z) = 1 / (1 + exp(-z))
```

Sigmoid는 출력이 0과 1 사이여서 확률처럼 해석할 수 있고, 미분 가능하므로 gradient 기반 학습에 사용할 수 있다.

## Boolean Function과 XOR 문제

Perceptron은 AND, OR처럼 linearly separable한 Boolean function은 표현할 수 있다. 하지만 XOR는 선형 decision boundary 하나로 분리할 수 없다.

```text
XOR = (x1 AND NOT x2) OR (x2 AND NOT x1)
```

Hidden layer를 추가하면 중간 feature를 만들 수 있고, XOR 같은 비선형 함수를 표현할 수 있다.

## Hidden Layer

Hidden layer는 입력을 새로운 representation으로 바꾼다.

```text
h_k = g(w_k^T x)
output = g(v^T h)
```

각 hidden unit은 입력 공간에서 다른 feature detector처럼 동작한다. 여러 hidden unit을 조합하면 복잡한 decision boundary를 만들 수 있다.

## Activation Function의 중요성

Neural network에서 activation function은 반드시 nonlinear이어야 한다. 모든 layer가 linear activation만 사용하면 여러 linear transform을 곱한 결과도 결국 하나의 linear transform이기 때문이다.

대표 activation:

- Step function
- Sigmoid
- tanh
- ReLU

## General Neural Network

Neural network는 linear transformation과 nonlinear activation을 반복적으로 쌓은 구조이다.

```text
input -> linear -> activation -> linear -> activation -> output
```

Layer가 깊어질수록 입력에서 점점 더 추상적인 feature를 학습할 수 있다.

## Forward Propagation

Forward propagation은 입력에서 출력까지 값을 계산하는 과정이다.

1. Input layer에 feature를 넣는다.
2. 각 layer에서 weighted sum을 계산한다.
3. Activation function을 적용한다.
4. 마지막 layer에서 prediction을 만든다.

## Backpropagation의 기본 아이디어

Backpropagation은 loss에 대한 각 parameter의 gradient를 효율적으로 계산하는 알고리즘이다. 출력 layer에서 시작해 chain rule을 이용해 gradient를 뒤쪽 layer로 전달한다.

```text
forward pass: prediction 계산
backward pass: gradient 계산
gradient descent: weight update
```

## 시험ㆍ복습 체크포인트

- Linear classifier가 XOR를 표현하지 못하는 이유를 설명할 수 있어야 한다.
- Hidden layer가 feature transform 역할을 한다는 점을 이해해야 한다.
- Activation function이 nonlinear이어야 하는 이유를 말할 수 있어야 한다.
- Forward propagation과 backpropagation의 역할을 구분할 수 있어야 한다.

