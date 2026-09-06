# Neural Networks Part 2: Backpropagation

tags: #artificial-intelligence #machine-learning #neural-network #backpropagation #chain-rule #vectorization

관련 노트: [[17 Neural Networks Part 1 - 다층신경망 기초]], [[19 Neural Networks Part 3 - 실전 학습과 정규화]]

## 핵심 요약

이 강의는 computational graph에서 chain rule을 적용해 gradient를 계산하는 backpropagation을 자세히 설명한다. Scalar 예제에서 시작해 gate별 local gradient, sigmoid layer, vector/matrix 연산, modular forward/backward API로 확장한다.

## Computational Graph

복잡한 함수는 작은 연산 노드들의 graph로 표현할 수 있다.

예:

```text
f(x,y,z) = (x + y) * z
```

Graph는 덧셈 노드와 곱셈 노드로 나뉜다. Forward pass에서는 각 노드의 출력값을 계산하고, backward pass에서는 최종 loss가 각 입력에 얼마나 민감한지 계산한다.

## Chain Rule

Backpropagation의 핵심은 chain rule이다.

```text
dL/dx = dL/dz * dz/dx
```

각 노드는 upstream gradient `dL/dout`을 받아 local gradient와 곱해 downstream gradient를 만든다.

```text
downstream gradient = upstream gradient * local gradient
```

## Gate별 Gradient Pattern

### Add Gate

덧셈은 gradient를 그대로 분배한다.

```text
z = x + y
dL/dx = dL/dz
dL/dy = dL/dz
```

### Multiply Gate

곱셈은 상대 입력값을 곱해 gradient를 전달한다.

```text
z = x*y
dL/dx = dL/dz * y
dL/dy = dL/dz * x
```

### Max Gate

Max gate는 forward에서 선택된 입력으로만 gradient를 보낸다.

```text
z = max(x,y)
gradient flows to argmax input
```

## Sigmoid Layer

Sigmoid 함수의 미분은 출력값으로 간단히 표현된다.

```text
sigmoid(x) = 1 / (1 + exp(-x))
d sigmoid / dx = sigmoid(x) * (1 - sigmoid(x))
```

Forward pass에서 sigmoid output을 저장해 두면 backward pass에서 효율적으로 gradient를 계산할 수 있다.

## Flat Code와 Modular Code

초기 예제는 모든 연산과 미분을 한 코드에 직접 쓸 수 있다. 그러나 neural network는 layer와 parameter가 많아지므로 각 operation을 module로 만들고 `forward()`와 `backward()` API를 제공하는 구조가 필요하다.

```text
forward(input) -> output, cache
backward(upstream_gradient, cache) -> input_gradient, parameter_gradient
```

Cache에는 backward 계산에 필요한 forward 중간값을 저장한다.

## Vector Derivatives

Neural network 구현에서는 scalar보다 vector와 matrix가 기본이다.

중요한 원칙:

```text
어떤 변수에 대한 loss gradient는 그 변수와 같은 shape를 가진다.
```

예를 들어 `W`가 `N x M` matrix이면 `dL/dW`도 `N x M`이다.

## ReLU Backpropagation

ReLU는 elementwise로 적용된다.

```text
f(x) = max(0, x)
```

Backward:

```text
x > 0이면 gradient 통과
x <= 0이면 gradient 0
```

## Matrix Multiplication Backpropagation

Matrix multiplication에서도 chain rule을 shape에 맞게 적용한다.

```text
Y = XW
dL/dX = dL/dY * W^T
dL/dW = X^T * dL/dY
```

이 식은 fully connected layer의 backward pass에서 핵심이다.

## Vectorization

여러 data point를 한 번에 matrix로 처리하면 반복문보다 효율적이다.

```text
X: batch input
W: weight matrix
Y = XW
```

Batch 단위 backpropagation은 모든 sample의 gradient contribution을 matrix 연산으로 모아 계산한다.

## 시험ㆍ복습 체크포인트

- Upstream gradient, local gradient, downstream gradient의 관계를 설명할 수 있어야 한다.
- Add, multiply, max gate의 backward rule을 이해해야 한다.
- Sigmoid와 ReLU의 derivative를 쓸 수 있어야 한다.
- Matrix multiplication backward 식의 shape를 검산할 수 있어야 한다.
- Forward/backward API에서 cache가 필요한 이유를 말할 수 있어야 한다.

